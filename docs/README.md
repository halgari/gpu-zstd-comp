# Documentation

- [`reference.md`](reference.md): the benchmark CLI, the corpus download tool, the library API,
  the presets and the environment variables. This is the current documentation.
- [`results/`](results/): measurements, one file per machine or measurement round, plus the logs
  they were taken from.
- [`design/`](design/): specs, plans and research notes.
- [`img/`](img/): the README's chart and the script that draws it.

The files in `results/` and `design/` are kept as written at the time. They name presets,
environment variables and code that have since changed or gone, so read them as history, and
check `reference.md` for how things work now.

## `design/`

- `specs/`: the design each stage was built to.
- `plans/`: the implementation plans for those stages.
- `m5/`: the optimal parse: its design (`m5-opt-design.md`, which `gzc_core::opt` follows), what
  drives the ratio, and the K3opt performance study.
- `m6/`: research for a faster optimal parse, and the subgroup-uniformity audit of the kernels
  (`subgroup-audit.md`).
- `m7/`: research into ideas beyond the current kernels.
- `speed/`: research and designs for the throughput work.
