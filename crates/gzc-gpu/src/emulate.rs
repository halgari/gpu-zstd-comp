//! Test aid: rewrite a shader so that, on the machine at hand (an NVIDIA card), it behaves as it
//! would on another GPU wherever WGSL or its backends leave the behaviour to the implementation.
//! `GpuOptions::emulate` / the `GZC_EMULATE_*` variables turn it on for a context; every module
//! then goes through `Emulation::rewrite` (`GpuContext::wgsl_module`).
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
    /// component, as on an Apple GPU (the M4 Pro frame corruption: K4's prefix-sum scratch).
    pub vector_rmw: bool,
}

impl Emulation {
    pub const NONE: Self = Self { shift_mod32: false, vector_rmw: false };
    pub const ALL: Self = Self { shift_mod32: true, vector_rmw: true };

    /// The parts the `GZC_EMULATE_*` variables turn on (any value but `0`).
    pub fn from_env() -> Self {
        Self {
            shift_mod32: crate::context::env_on("GZC_EMULATE_SHIFT_MOD32"),
            vector_rmw: crate::context::env_on("GZC_EMULATE_VEC_RMW"),
        }
    }

    pub fn any(self) -> bool {
        self.shift_mod32 || self.vector_rmw
    }

    /// Both parts that either of `self` and `other` has.
    pub fn or(self, other: Self) -> Self {
        Self { shift_mod32: self.shift_mod32 || other.shift_mod32, vector_rmw: self.vector_rmw || other.vector_rmw }
    }

    /// `src` (complete WGSL) rewritten as described on the fields.
    pub fn rewrite(self, src: &str) -> anyhow::Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_amounts_are_taken_mod_32() {
        let src = "fn f(x: u32, n: u32) -> u32 { var y = x << n; y >>= (n + 1u); return y + (x << 3u); }";
        let out = Emulation { shift_mod32: true, vector_rmw: false }.rewrite(src).unwrap();
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
        let out = Emulation { shift_mod32: false, vector_rmw: true }.rewrite(src).unwrap();
        assert!(out.contains("{ var rmw = s4_[(lid >> 2u)]; rmw[(lid & 3u)] = lid; s4_[(lid >> 2u)] = rmw; }"), "{out}");
        assert!(out.contains("{ var rmw = s4_[(lid >> 2u)]; rmw.y = 1u; s4_[(lid >> 2u)] = rmw; }"), "{out}");
        assert!(out.contains("{ var rmw = v; rmw[(lid & 1u)] = 2u; v = rmw; }"), "{out}");
        assert!(out.contains("plain[(lid & 3u)] = 3u;"), "{out}");
        assert!(out.contains("s4_[(lid & 15u)] = vec4(0u);"), "{out}");
        assert!(out.contains("v = vec2(0u);"), "{out}");
    }
}
