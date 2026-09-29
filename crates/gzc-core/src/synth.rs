//! Seeded synthetic test corpora: uniform, text-like, DDS-like and NIF-like data.
use crate::config::BLOCK_SIZE;

/// Small, fast, deterministic PRNG (splitmix64).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// All-zero buffer of the given length.
pub fn zeros(len: usize) -> Vec<u8> {
    vec![0u8; len]
}

/// Uniform random bytes, deterministic per seed.
pub fn random(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        out.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    out.truncate(len);
    out
}

const WORDS: [&str; 64] = [
    "dragon", "whiterun", "guard", "arrow", "knee", "jarl", "septim", "skooma", "nord", "imperial",
    "stormcloak", "thalmor", "mead", "sweetroll", "riverwood", "solitude", "windhelm", "markarth",
    "falkreath", "dawnstar", "winterhold", "morthal", "riften", "shield", "sword", "bow", "mace",
    "axe", "dagger", "armor", "helmet", "potion", "scroll", "spell", "mage", "warrior", "thief",
    "assassin", "bandit", "draugr", "skeleton", "troll", "giant", "mammoth", "wolf", "bear",
    "saber", "cat", "horse", "cart", "road", "mountain", "cave", "ruin", "temple", "shrine",
    "altar", "talos", "akatosh", "sithis", "daedra", "aetherius", "oblivion", "soul",
];

/// Text-like data: words from a fixed Skyrim-flavored list, space-separated, newline every ~12 words.
pub fn text(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len + 32);
    let mut word_count = 0u32;
    while out.len() < len {
        let idx = (rng.next_u64() % WORDS.len() as u64) as usize;
        out.extend_from_slice(WORDS[idx].as_bytes());
        word_count += 1;
        if word_count % 12 == 0 {
            out.push(b'\n');
        } else {
            out.push(b' ');
        }
    }
    out.truncate(len);
    out
}

fn write_u32_le(dst: &mut [u8], v: u32) {
    dst.copy_from_slice(&v.to_le_bytes());
}

/// DDS-texture-like data: a plausible 128-byte DDS/DXT1 header followed by
/// 8-byte BC1 blocks drawn mostly from a small repeating palette.
pub fn dds_like(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len.max(128));

    let mut header = [0u8; 128];
    header[0..4].copy_from_slice(b"DDS ");
    write_u32_le(&mut header[4..8], 124); // dwSize
    write_u32_le(&mut header[8..12], 0x0002_1007); // dwFlags: CAPS|HEIGHT|WIDTH|PIXELFORMAT|LINEARSIZE
    const DIMS: [u32; 5] = [64, 128, 256, 512, 1024];
    let height = DIMS[(rng.next_u64() % DIMS.len() as u64) as usize];
    let width = DIMS[(rng.next_u64() % DIMS.len() as u64) as usize];
    write_u32_le(&mut header[12..16], height); // dwHeight
    write_u32_le(&mut header[16..20], width); // dwWidth
    write_u32_le(&mut header[20..24], (width * height) / 2); // dwPitchOrLinearSize (BC1)
    write_u32_le(&mut header[24..28], 0); // dwDepth
    write_u32_le(&mut header[28..32], 1); // dwMipMapCount
    // header[32..76] dwReserved1 stays zero
    write_u32_le(&mut header[76..80], 32); // ddspf.dwSize
    write_u32_le(&mut header[80..84], 0x4); // ddspf.dwFlags = DDPF_FOURCC
    header[84..88].copy_from_slice(b"DXT1"); // ddspf.dwFourCC
    // ddspf bit masks and dwCaps2..4/dwReserved2 stay zero
    write_u32_le(&mut header[108..112], 0x1000); // dwCaps = DDSCAPS_TEXTURE
    out.extend_from_slice(&header);

    let palette: Vec<[u8; 8]> = (0..48)
        .map(|_| {
            let mut blk = [0u8; 8];
            for b in blk.iter_mut() {
                *b = (rng.next_u64() & 0xFF) as u8;
            }
            blk
        })
        .collect();

    while out.len() < len {
        if rng.next_u64() % 100 < 70 {
            let idx = (rng.next_u64() % palette.len() as u64) as usize;
            out.extend_from_slice(&palette[idx]);
        } else {
            let mut blk = [0u8; 8];
            for b in blk.iter_mut() {
                *b = (rng.next_u64() & 0xFF) as u8;
            }
            out.extend_from_slice(&blk);
        }
    }
    out.truncate(len);
    out
}

const NIF_STRINGS: [&str; 6] = [
    "NiNode",
    "BSTriShape",
    "BSLightingShaderProperty",
    "BSShaderTextureSet",
    "NiAlphaProperty",
    "Scene Root",
];

/// NIF-mesh-like data: a Gamebryo header, a cycling string table, then
/// repeating vertex + triangle-index records on a smooth grid.
pub fn nif_like(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len + 64);

    out.extend_from_slice(b"Gamebryo File Format, Version 20.2.0.7\n");
    out.extend_from_slice(&(NIF_STRINGS.len() as u32).to_le_bytes()); // num strings
    out.extend_from_slice(&100u32.to_le_bytes()); // user version

    for s in NIF_STRINGS {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    let mut i: u32 = 0;
    while out.len() < len {
        let px = (i % 32) as f32;
        let py = ((i / 32) % 32) as f32;
        let pz = ((px + py) * 0.1).sin() * 5.0;
        out.extend_from_slice(&px.to_le_bytes());
        out.extend_from_slice(&py.to_le_bytes());
        out.extend_from_slice(&pz.to_le_bytes());
        let u = (i as u16).wrapping_mul(37);
        let v = (i as u16).wrapping_mul(53);
        out.extend_from_slice(&u.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
        for _ in 0..4 {
            out.push((rng.next_u64() & 0xFF) as u8); // packed normal
        }
        for _ in 0..4 {
            out.push((rng.next_u64() & 0xFF) as u8); // vertex colour
        }
        out.extend_from_slice(&(i as u16).to_le_bytes());
        out.extend_from_slice(&((i + 1) as u16).to_le_bytes());
        out.extend_from_slice(&((i + 2) as u16).to_le_bytes());
        i = i.wrapping_add(1);
    }
    out.truncate(len);
    out
}

/// Named synthetic corpus entries covering block-size edge cases.
pub fn test_cases() -> Vec<(&'static str, Vec<u8>)> {
    let half = BLOCK_SIZE / 2;
    vec![
        ("zeros", zeros(BLOCK_SIZE)),
        ("random", random(1, BLOCK_SIZE)),
        ("text", text(2, 3 * BLOCK_SIZE / 2)),
        ("dds", dds_like(3, 2 * BLOCK_SIZE)),
        ("nif", nif_like(4, 2 * BLOCK_SIZE)),
        ("one_byte", vec![0x42]),
        ("exact_block", text(5, BLOCK_SIZE)),
        ("block_plus_one", text(6, BLOCK_SIZE + 1)),
        ("period3", {
            let mut v = Vec::with_capacity(BLOCK_SIZE);
            while v.len() < BLOCK_SIZE {
                v.extend_from_slice(&[1u8, 2, 3]);
            }
            v.truncate(BLOCK_SIZE);
            v
        }),
        ("mixed", {
            let mut v = text(7, half);
            v.extend_from_slice(&random(8, BLOCK_SIZE - half));
            v
        }),
        ("strided", {
            let mut v = Vec::with_capacity(BLOCK_SIZE);
            let mut i: u32 = 0;
            while v.len() < BLOCK_SIZE {
                v.extend_from_slice(&i.to_le_bytes());
                v.extend_from_slice(&[0xAB; 20]);
                i = i.wrapping_add(1);
            }
            v.truncate(BLOCK_SIZE);
            v
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_are_deterministic_and_sized() {
        for f in [random, text, dds_like, nif_like] {
            assert_eq!(f(9, 1000), f(9, 1000));
            assert_eq!(f(9, 1000).len(), 1000);
            assert_ne!(f(9, 4096), f(10, 4096));
        }
    }

    #[test]
    fn test_cases_include_edge_sizes() {
        let names: Vec<_> = test_cases().iter().map(|(n, _)| *n).collect();
        for n in [
            "zeros", "random", "text", "dds", "nif", "one_byte", "exact_block", "block_plus_one",
            "period3", "mixed", "strided",
        ] {
            assert!(names.contains(&n), "missing {n}");
        }
    }
}
