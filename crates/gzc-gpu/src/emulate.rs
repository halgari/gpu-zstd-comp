//! Test aid: rewrite a shader so that, on the machine at hand (an NVIDIA card), it behaves as it
//! would on another GPU wherever WGSL or its backends leave the behaviour to the implementation.
//! `GpuOptions::emulate` turns it on for a context (`GZC_EMULATE_*` through
//! `GpuOptions::from_env`). Every module then goes through `Emulation::rewrite`
//! (`GpuContext::wgsl_module`).
//!
//! The rewrite works on naga's WGSL output: the source is parsed and validated by naga and written
//! back, which gives one statement per line, every binary expression fully parenthesized as
//! `(left op right)` and every call or load that matters bound to a `let`, so the textual passes
//! below only need bracket matching.
use anyhow::anyhow;

/// Which foreign semantics to emulate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Emulation {
    /// `GZC_EMULATE_SHIFT_MOD32`: every `x << n` / `x >> n` shifts by `n % 32`, as Apple and AMD
    /// GPUs do (NVIDIA yields 0 / the sign for a shift by 32 or more). Catches shift amounts that
    /// can reach 32.
    pub shift_mod32: bool,
    /// `GZC_EMULATE_VEC_RMW`: a store to one component of a vector in workgroup memory
    /// (`v[i] = x`, `a[j][i] = x`, `a[j].x = x`) becomes a read-modify-write of the whole vector,
    /// which is how Apple's Metal compiler lowers it. Threads of one SIMD group that store
    /// different components of the same vector in one instruction then keep only one thread's
    /// component, as on an Apple GPU.
    pub vector_rmw: bool,
    /// `GZC_EMULATE_SKEW`: timing skew, for races a slow or preempted GPU would expose. Every
    /// invocation stalls for a pseudo-random time (a dependent ALU chain of up to 512 steps, one call
    /// in eight) at entry, after every barrier and `workgroupUniformLoad`, before every subgroup
    /// operation, and (in modules that use subgroup operations; one call in four, up to 64 steps) at
    /// the start of every `if` / `else` / `switch` case body (naga's output, so also the `if`s it
    /// lowers `&&` / `||` to), so that the phases between barriers start at very
    /// different times across the workgroup, the lanes of a subgroup arrive at their collectives
    /// apart, and lanes that took a branch fall behind the ones that did not: a collective that
    /// counts on the lanes having reconverged after a divergent branch sees them apart. Output must
    /// not change. (Slower: a test aid.)
    pub skew: bool,
}

impl Emulation {
    /// Nothing emulated.
    pub const NONE: Self = Self { shift_mod32: false, vector_rmw: false, skew: false };
    /// Every other GPU's semantics (not the timing skew, which only slows things down).
    pub const ALL: Self = Self { shift_mod32: true, vector_rmw: true, skew: false };

    /// True when any part is on.
    pub fn any(self) -> bool {
        self.shift_mod32 || self.vector_rmw || self.skew
    }

    /// Both parts that either of `self` and `other` has.
    pub fn or(self, other: Self) -> Self {
        Self {
            shift_mod32: self.shift_mod32 || other.shift_mod32,
            vector_rmw: self.vector_rmw || other.vector_rmw,
            skew: self.skew || other.skew,
        }
    }

    /// `src` (complete WGSL) rewritten as described on the fields.
    pub(crate) fn rewrite(self, src: &str) -> anyhow::Result<String> {
        let module = naga::front::wgsl::parse_str(src).map_err(|e| anyhow!("{}", e.emit_to_string(src)))?;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .map_err(|e| anyhow!("{e:?}"))?;
        let mut out = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())?;
        if self.shift_mod32 {
            out = shift_amounts_mod32(&out);
        }
        if self.vector_rmw {
            out = vector_stores_rmw(&out);
        }
        if self.skew {
            out = skew_timing(&out)?;
        }
        Ok(out)
    }
}

/// The index of the bracket closing the group `s` is inside of (the first `)`/`]` at depth -1).
fn close_of(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (j, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth -= 1;
                if depth < 0 {
                    return Some(j);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every ` << right)` / ` >> right)` of naga's output becomes ` << ((right) % 32u))`. A shift
/// amount is `u32` or `vecN<u32>`, and `%` takes a vector and a scalar.
fn shift_amounts_mod32(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 8);
    let mut rest = s;
    loop {
        let Some(i) = [rest.find(" << "), rest.find(" >> ")].into_iter().flatten().min() else {
            out.push_str(rest);
            return out;
        };
        let amount = &rest[i + 4..];
        let end = close_of(amount).expect("naga writes every binary expression parenthesized");
        out.push_str(&rest[..i + 4]);
        out.push_str(&format!("(({}) % 32u)", shift_amounts_mod32(&amount[..end])));
        rest = &amount[end..];
    }
}

/// Every statement line `NAME<accessors><component> = value;` of naga's output whose `NAME` is
/// a workgroup variable of vector or array-of-vector type, `<component>` being `[i]` or `.x`
/// (etc.), becomes `{ var rmw = NAME<accessors>; rmw<component> = value; NAME<accessors> = rmw; }`.
/// The accessors are side-effect free in naga's output (calls and loads are bound to `let`s), so
/// evaluating them twice is harmless.
fn vector_stores_rmw(s: &str) -> String {
    // `var<workgroup> NAME: vecN<..>;` or `var<workgroup> NAME: array<vecN<..>, M>;`
    let vec_vars: Vec<(String, bool)> = s
        .lines()
        .filter_map(|l| l.trim().strip_prefix("var<workgroup> "))
        .filter_map(|l| l.split_once(": "))
        .filter(|(_, ty)| ty.starts_with("vec") || ty.starts_with("array<vec"))
        .map(|(name, ty)| (name.to_string(), ty.starts_with("array")))
        .collect();
    let mut out = String::with_capacity(s.len());
    for line in s.lines() {
        out.push_str(&rmw_line(line, &vec_vars).unwrap_or_else(|| line.to_string()));
        out.push('\n');
    }
    out
}

fn rmw_line(line: &str, vec_vars: &[(String, bool)]) -> Option<String> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let t = line.trim();
    let (name, is_array) =
        vec_vars.iter().find(|(n, _)| t.starts_with(n.as_str()) && matches!(t.as_bytes().get(n.len()), Some(b'[' | b'.')))?;
    // Accessors after the name (`[..]` groups and `.field`s) up to ` = `; their start offsets.
    let mut pos = name.len();
    let mut starts = Vec::new();
    let b = t.as_bytes();
    while pos < b.len() {
        match b[pos] {
            b'[' => {
                starts.push(pos);
                pos += 1 + close_of(&t[pos + 1..])? + 1;
            }
            b'.' => {
                starts.push(pos);
                pos += 1;
                while pos < b.len() && (b[pos] as char).is_ascii_alphanumeric() {
                    pos += 1;
                }
            }
            _ => break,
        }
    }
    let value = t[pos..].strip_prefix(" = ")?.strip_suffix(';')?;
    // A component store: one accessor into a vector, an element index then one into an array.
    if starts.len() != 1 + usize::from(*is_array) || (*is_array && b[starts[0]] != b'[') {
        return None;
    }
    let last = *starts.last()?;
    let (base, component) = (&t[..last], &t[last..pos]);
    Some(format!("{indent}{{ var rmw = {base}; rmw{component} = {value}; {base} = rmw; }}"))
}

/// The helpers `skew_timing` appends: a per-invocation LCG state and the stall.
const SKEW_WGSL: &str = "
var<private> gzc_seed: u32;
var<workgroup> gzc_spin: atomic<u32>;
fn gzc_hash(x: u32) -> u32 {
    var h = x * 0x9E3779B1u;
    h ^= h >> 15u;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13u;
    return h;
}
fn gzc_skew() {
    gzc_seed = gzc_seed * 1664525u + 1013904223u;
    let r = gzc_seed >> 24u;
    if (r < 32u) {
        // A dependent ALU chain of up to 512 steps; the atomic on its (practically never 0)
        // result keeps it from being optimized away.
        var x = gzc_seed | 1u;
        let n = (r + 1u) * 16u;
        for (var i = 0u; i < n; i++) { x = (x ^ (x >> 13u)) * 0x5BD1E995u; }
        if (x == 0u) { atomicAdd(&gzc_spin, 1u); }
    }
}
// The lighter stall at the start of branch bodies (one call in four, up to 64 steps): a full
// gzc_skew in every branch of K3's coop parse loop runs its dispatches past the driver's
// preemption timeout.
fn gzc_skew_branch() {
    gzc_seed = gzc_seed * 1664525u + 1013904223u;
    let r = gzc_seed >> 24u;
    if (r < 64u) {
        var x = gzc_seed | 1u;
        let n = ((r & 15u) + 1u) * 4u;
        for (var i = 0u; i < n; i++) { x = (x ^ (x >> 13u)) * 0x5BD1E995u; }
        if (x == 0u) { atomicAdd(&gzc_spin, 1u); }
    }
}
";

/// The parameter name of `@builtin(name)` in an entry point's parameter list, if any.
fn builtin_param(params: &str, name: &str) -> Option<String> {
    let at = params.find(&format!("@builtin({name})"))?;
    let rest = params[at..].split_once(')')?.1.trim_start();
    Some(rest.split(':').next()?.trim().to_string())
}

/// Whether naga's output line `t` (trimmed) opens the body of a branch: `if .. {`, `} else {`,
/// `case ..: {`, `default: {`.
fn opens_branch(t: &str) -> bool {
    t.ends_with('{')
        && (t.starts_with("if ") || t.starts_with("if(") || t.starts_with("} else") || t.starts_with("case ") || t.starts_with("default"))
}

/// The timing skew (`Emulation::skew`) on naga's output: every compute entry point gets the
/// `local_invocation_index` and `workgroup_id` built-ins (when it lacks them), seeds `gzc_seed`
/// from them and stalls once; every `workgroupBarrier();` / `storageBarrier();` statement and
/// every `workgroupUniformLoad` binding is followed by a stall, every statement calling a
/// subgroup built-in preceded by one, and, in a module that calls subgroup built-ins, every
/// branch body (`if` / `else` / `case` / `default`) starts with the lighter `gzc_skew_branch`.
/// Only there, and lighter: full stalls in every branch run K2's and K3's dispatches past the
/// driver's preemption timeout (NVIDIA Xid 109), and outside those modules nothing depends on
/// reconvergence.
fn skew_timing(s: &str) -> anyhow::Result<String> {
    let mut out = String::with_capacity(s.len() * 2);
    let mut entry = false;
    let calls_subgroup = |t: &str| t.contains("subgroup") && !t.starts_with('@') && !t.starts_with("fn ");
    let branch_stalls = s.lines().any(|l| calls_subgroup(l.trim()));
    for line in s.lines() {
        let t = line.trim();
        if t.starts_with("@compute") {
            entry = true;
        }
        if entry && t.starts_with("fn ") {
            entry = false;
            let open = t.find('(').ok_or_else(|| anyhow!("entry point without parameters: {t}"))?;
            let close = t.rfind(')').ok_or_else(|| anyhow!("entry point line not on one line: {t}"))?;
            anyhow::ensure!(t.ends_with('{'), "entry point line not on one line: {t}");
            let params = &t[open + 1..close];
            let mut extra = Vec::new();
            let lii = builtin_param(params, "local_invocation_index").unwrap_or_else(|| {
                extra.push("@builtin(local_invocation_index) gzc_lii: u32");
                "gzc_lii".into()
            });
            let wid = builtin_param(params, "workgroup_id").unwrap_or_else(|| {
                extra.push("@builtin(workgroup_id) gzc_wid: vec3<u32>");
                "gzc_wid".into()
            });
            let sep = if params.trim().is_empty() || extra.is_empty() { "" } else { ", " };
            out.push_str(&format!("{}{sep}{}{}\n", &t[..close], extra.join(", "), &t[close..]));
            out.push_str(&format!(
                "    gzc_seed = gzc_hash({lii} ^ gzc_hash({wid}.x ^ ({wid}.y << 16u) ^ ({wid}.z << 24u)));\n    gzc_skew();\n"
            ));
            continue;
        }
        if calls_subgroup(t) {
            out.push_str("gzc_skew();\n");
        }
        out.push_str(line);
        out.push('\n');
        if t == "workgroupBarrier();" || t == "storageBarrier();" || t.contains("workgroupUniformLoad(") {
            out.push_str("gzc_skew();\n");
        } else if branch_stalls && opens_branch(t) {
            out.push_str("gzc_skew_branch();\n");
        }
    }
    out.push_str(SKEW_WGSL);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_amounts_are_taken_mod_32() {
        let src = "fn f(x: u32, n: u32) -> u32 { var y = x << n; y >>= (n + 1u); return y + (x << 3u); }";
        let out = Emulation { shift_mod32: true, ..Emulation::NONE }.rewrite(src).unwrap();
        assert!(out.contains("(x << ((n) % 32u))"), "{out}");
        assert!(out.contains(">> (((n + 1u)) % 32u))"), "{out}");
        assert!(out.contains("(x << ((3u) % 32u))"), "{out}");
    }

    #[test]
    fn workgroup_vector_component_stores_become_rmw() {
        let src = "var<workgroup> s4: array<vec4<u32>, 16>;\n\
                   var<workgroup> v: vec2<u32>;\n\
                   var<workgroup> plain: array<u32, 4>;\n\
                   @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lid: u32) {\n\
                   s4[lid >> 2u][lid & 3u] = lid; s4[lid >> 2u].y = 1u; v[lid & 1u] = 2u; plain[lid & 3u] = 3u;\n\
                   s4[lid & 15u] = vec4(0u); v = vec2(0u); }";
        let out = Emulation { vector_rmw: true, ..Emulation::NONE }.rewrite(src).unwrap();
        assert!(out.contains("{ var rmw = s4_[(lid >> 2u)]; rmw[(lid & 3u)] = lid; s4_[(lid >> 2u)] = rmw; }"), "{out}");
        assert!(out.contains("{ var rmw = s4_[(lid >> 2u)]; rmw.y = 1u; s4_[(lid >> 2u)] = rmw; }"), "{out}");
        assert!(out.contains("{ var rmw = v; rmw[(lid & 1u)] = 2u; v = rmw; }"), "{out}");
        assert!(out.contains("plain[(lid & 3u)] = 3u;"), "{out}");
        assert!(out.contains("s4_[(lid & 15u)] = vec4(0u);"), "{out}");
        assert!(out.contains("v = vec2(0u);"), "{out}");
    }

    #[test]
    fn skew_seeds_entry_points_and_stalls_around_barriers() {
        let src = "var<workgroup> a: array<u32, 64>;\n\
                   @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lid: u32) {\n\
                   a[lid] = lid; workgroupBarrier(); let x = workgroupUniformLoad(&a[0]); a[lid] = x; }\n\
                   @compute @workgroup_size(64) fn other() { storageBarrier(); }";
        let out = Emulation { skew: true, ..Emulation::NONE }.rewrite(src).unwrap();
        assert!(out.contains("gzc_seed = gzc_hash(lid ^ gzc_hash(gzc_wid.x"), "{out}");
        assert!(out.contains("fn other(@builtin(local_invocation_index) gzc_lii: u32, @builtin(workgroup_id) gzc_wid: vec3<u32>)"), "{out}");
        assert_eq!(out.matches("gzc_skew();").count(), 2 + 3, "{out}");
        naga::front::wgsl::parse_str(&out).unwrap_or_else(|e| panic!("{}\n{out}", e.emit_to_string(&out)));
    }

    #[test]
    fn skew_stalls_inside_divergent_branches() {
        // Entry and the subgroup call get a full stall; `a && b` (lowered to an `if` with an
        // `else`), the explicit `if` / `else` and the `switch`'s two bodies a branch stall each.
        let src = "\
                   var<workgroup> a: array<u32, 64>;\n\
                   @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lid: u32) {\n\
                   var x = 0u; if (lid < 3u) { x = 1u; } else { x = 2u; }\n\
                   let c = (lid > 1u) && (x == 2u);\n\
                   switch (lid & 1u) { case 0u: { x += 1u; } default: { x += 2u; } }\n\
                   a[lid] = subgroupAdd(select(x, 0u, c)); }";
        let out = Emulation { skew: true, ..Emulation::NONE }.rewrite(src).unwrap();
        assert_eq!(out.matches("gzc_skew();").count(), 1 + 1, "{out}");
        assert_eq!(out.matches("gzc_skew_branch();").count(), 2 + 2 + 2, "{out}");
        naga::front::wgsl::parse_str(&out).unwrap_or_else(|e| panic!("{}\n{out}", e.emit_to_string(&out)));
        // Without subgroup calls the branches stay as they are.
        let plain = src.replace("subgroupAdd(select(x, 0u, c))", "select(x, 0u, c)");
        let out = Emulation { skew: true, ..Emulation::NONE }.rewrite(&plain).unwrap();
        assert_eq!(out.matches("gzc_skew();").count(), 1, "{out}");
        assert_eq!(out.matches("gzc_skew_branch();").count(), 0, "{out}");
    }
}
