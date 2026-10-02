//! `GpuOptions::try_from_env`: every `GZC_*` variable either takes its value or is refused by
//! name, and an empty value counts as unset. One test in its own binary, because it changes the
//! process environment.
use gzc_gpu::{CompressorOptions, Emulation, Error, GpuOptions, K3Kernel, Level};

const VARS: [&str; 17] = [
    "GZC_NO_SUBGROUPS",
    "GZC_DIRECT_UPLOAD",
    "GZC_TRANSFER_QUEUE",
    "GZC_NO_TIMESTAMPS",
    "GZC_SORTED",
    "GZC_UPLOAD_THREADS",
    "GZC_K1_GROUPS",
    "GZC_K3_MODE",
    "GZC_K3_W",
    "GZC_K3_FORCE_FALLBACK",
    "GZC_CHECKED_SHADERS",
    "GZC_EMULATE_SHIFT_MOD32",
    "GZC_EMULATE_VEC_RMW",
    "GZC_EMULATE_SKEW",
    "GZC_POISON",
    "GZC_POISON_SEED",
    "GZC_DUMP_WGSL",
];

fn set(vars: &[(&str, &str)]) {
    // SAFETY: the only test of this binary, so no other thread reads the environment.
    unsafe {
        VARS.iter().for_each(|v| std::env::remove_var(v));
        vars.iter().for_each(|(k, v)| std::env::set_var(k, v));
    }
}

#[test]
fn try_from_env_takes_or_refuses_every_variable() {
    set(&[]);
    assert_eq!(GpuOptions::try_from_env().unwrap(), GpuOptions::default());
    assert_eq!(GpuOptions::from_env(), GpuOptions::default());
    assert_eq!(CompressorOptions::try_from_env(Level::Zstd9).unwrap().gpu, GpuOptions::default());

    set(&[
        ("GZC_NO_SUBGROUPS", "1"),
        ("GZC_DIRECT_UPLOAD", "0"),
        ("GZC_TRANSFER_QUEUE", "0"),
        ("GZC_NO_TIMESTAMPS", "yes"),
        ("GZC_SORTED", "0"),
        ("GZC_UPLOAD_THREADS", "2"),
        ("GZC_K1_GROUPS", "256"),
        ("GZC_K3_MODE", "coop"),
        ("GZC_K3_W", "16"),
        ("GZC_K3_FORCE_FALLBACK", "1"),
        ("GZC_CHECKED_SHADERS", "1"),
        ("GZC_EMULATE_SHIFT_MOD32", "1"),
        ("GZC_EMULATE_VEC_RMW", "1"),
        ("GZC_EMULATE_SKEW", "1"),
        ("GZC_POISON", "1"),
        ("GZC_POISON_SEED", "42"),
        ("GZC_DUMP_WGSL", "/tmp/wgsl"),
    ]);
    let want = GpuOptions {
        subgroups: false,
        direct_upload: Some(false),
        transfer_queue: false,
        timestamps: false,
        sorted_finder: false,
        upload_threads: Some(2),
        k1_groups: Some(256),
        k3_kernel: Some(K3Kernel::Coop),
        k3_width: Some(16),
        k3_force_fallback: true,
        checked_shaders: true,
        emulate: Emulation { shift_mod32: true, vector_rmw: true, skew: true },
        poison: true,
        poison_seed: Some(42),
        dump_wgsl: Some("/tmp/wgsl".into()),
    };
    assert_eq!(GpuOptions::try_from_env().unwrap(), want);

    // An empty value is "unset", for every variable.
    let empty: Vec<(&str, &str)> = VARS.iter().map(|&v| (v, "")).collect();
    set(&empty);
    assert_eq!(GpuOptions::try_from_env().unwrap(), GpuOptions::default());
    for &var in &VARS {
        set(&[(var, "")]);
        assert_eq!(GpuOptions::try_from_env().unwrap(), GpuOptions::default(), "{var}=");
    }

    // Switches: `0` is the other state, whichever the default.
    set(&[("GZC_NO_SUBGROUPS", "0"), ("GZC_TRANSFER_QUEUE", "1"), ("GZC_K3_FORCE_FALLBACK", "0"), ("GZC_DIRECT_UPLOAD", "1")]);
    assert_eq!(GpuOptions::try_from_env().unwrap(), GpuOptions { direct_upload: Some(true), ..GpuOptions::default() });
    set(&[("GZC_K3_MODE", "seq")]);
    assert_eq!(GpuOptions::try_from_env().unwrap().k3_kernel, Some(K3Kernel::Seq));

    // A value a variable does not take is refused, naming the variable. None is ignored.
    for (key, value) in [
        ("GZC_DIRECT_UPLOAD", "yes"),
        ("GZC_UPLOAD_THREADS", "four"),
        ("GZC_UPLOAD_THREADS", "0"),
        ("GZC_K1_GROUPS", "-1"),
        ("GZC_K1_GROUPS", "0"),
        ("GZC_K3_MODE", "bogus"),
        ("GZC_K3_W", "x"),
        ("GZC_K3_W", "7"),
        ("GZC_POISON_SEED", "1.5"),
    ] {
        set(&[(key, value)]);
        let e = GpuOptions::try_from_env().err().unwrap_or_else(|| panic!("{key}={value} was accepted"));
        assert!(matches!(&e, Error::InvalidInput(m) if m.contains(key)), "{key}={value}: {e:?}");
        let e = CompressorOptions::try_from_env(Level::Zstd3).expect_err("the same error");
        assert!(matches!(&e, Error::InvalidInput(m) if m.contains(key)), "{key}={value}: {e:?}");
        let panic = std::panic::catch_unwind(GpuOptions::from_env).expect_err("from_env panics");
        let msg = panic.downcast_ref::<String>().expect("a message");
        assert!(msg.contains(key), "{key}={value}: {msg}");
    }
    set(&[]);
}
