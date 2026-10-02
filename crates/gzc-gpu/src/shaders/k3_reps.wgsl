// The repeat-offset history and its helpers, shared by the sequential parse (k3_parse.wgsl, and
// the k3_lazy / k3_coop code appended to it) and the segmented fix-up (k3_fixup.wgsl). The host
// prepends this to both (`K3_REPS_WGSL`, `K3_FIXUP_WGSL`).

// Repeat-offset history (gzc_core::seq::Reps).
var<private> r0: u32;
var<private> r1: u32;
var<private> r2: u32;

// == gzc_core::seq::off_base_for
fn off_base_for(offset: u32, lit_len: u32) -> u32 {
    if (lit_len > 0u) {
        if (offset == r0) { return 1u; }
        if (offset == r1) { return 2u; }
        if (offset == r2) { return 3u; }
    } else {
        if (offset == r1) { return 1u; }
        if (offset == r2) { return 2u; }
        if (r0 > 1u && offset == r0 - 1u) { return 3u; }
    }
    return offset + 3u;
}

// == gzc_core::seq::apply_off_base (repeat-history update only).
fn apply_off_base(off_base: u32, lit_len: u32) {
    if (off_base > 3u) {
        r2 = r1;
        r1 = r0;
        r0 = off_base - 3u;
        return;
    }
    let idx = off_base - 1u + select(0u, 1u, lit_len == 0u);
    var off: u32;
    switch (idx) {
        case 0u: { off = r0; }
        case 1u: { off = r1; }
        case 2u: { off = r2; }
        default: { off = r0 - 1u; } // wrapping, as in the reference
    }
    if (idx > 0u) {
        if (idx > 1u) { r2 = r1; }
        r1 = r0;
        r0 = off;
    }
}
