# nodepilot

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/OortCloudd/nodepilot/blob/main/LICENSE)
[![Dependencies](https://img.shields.io/badge/deps-stdlib%20%2B%20pyyaml-lightgrey.svg)](https://github.com/OortCloudd/nodepilot/blob/main/pyproject.toml)

A single-node, daemonless, OOM-safe job orchestrator for one big machine. You
declare a queue of compute jobs in YAML; nodepilot runs them back-to-back with
kernel-enforced per-job memory containment, NUMA-local core pinning, dependency
ordering, and a crash-safe resume loop — no cluster, no scheduler daemon to
operate, no `sbatch`.

nodepilot's value is not a novel mechanism. cgroups v2, `numactl`, and
`oom_score_adj` already exist and are well documented. What this project
contributes is an *opinionated synthesis* of those primitives, and the
field-earned operating knowledge of how they actually behave under load — which
of the obvious approaches silently fail, and which blunt one holds. That
knowledge is written down in [`GOTCHAS.md`](GOTCHAS.md) (symptom → root cause →
the exact guard in the source) and in
[`docs/architecture.md`](docs/architecture.md). Read those if you want to judge
whether the design choices are sound; this README is the operating manual.

```
$ nodepilot run queue.yaml
2026-01-01 12:00:00 INFO  slice active slice=nodepilot.slice used_gb=0.4 cap_gb=200.0
2026-01-01 12:00:00 INFO  queue loaded jobs=3 pending=3 cores=64 ram_budget_gb=200 nodes=2
2026-01-01 12:00:00 INFO  launch job=prep cores=4 ram_gb=8 cpu=0-3 node=0 mem=membind:0
2026-01-01 12:00:11 INFO  done job=prep hours=0.00
2026-01-01 12:00:11 INFO  launch job=solve cores=16 ram_gb=64 cpu=0-15 node=0 mem=membind:0
```

---

## Design principles

Three principles, each paid for by a failure documented in
[`GOTCHAS.md`](GOTCHAS.md), are the spine of the design. They explain why some
parts that look over-engineered are deliberate.

1. **Measure, don't trust the request.** Declared `ram_gb` is a starting point,
   not a safety mechanism — real footprints of long compute jobs overshoot and
   undershoot by tens of percent. Admission reads the live cgroup
   `memory.current` of the slice and believes the kernel over the YAML whenever a
   cgroup is present. (GOTCHAS §2, §9.)

2. **One placer, not two.** Whoever sets the cpuset must be the only thing that
   binds. An outer `numactl --membind` is defeated if an MPI launcher re-binds
   ranks on top of it: each rank's first-touch pages then fault in on whatever
   node it started on, and first-touch is permanent — moving the thread back
   later does not move the page, so you pay remote-memory latency for the life of
   the job. The fix is to tell the launcher to stand down
   (`--bind-to none --map-by node`) and let the outer `numactl` own all
   placement. (GOTCHAS §1, §4, §5.)

3. **Blunt and bypass-proof beats clever and fragile.** A global concurrency cap
   and a single slice `MemoryMax` hold regardless of how any one job distributes
   its pages; clever per-node RAM accounting gets routed around (by memory
   interleaving, by jobs started outside the scheduler) and is therefore kept
   opt-in and off the hot path. (GOTCHAS §3, §6, §7, §10.)

The architecture document expands on each: [`docs/architecture.md`](docs/architecture.md).

---

## Where this fits

nodepilot occupies a narrow, otherwise-empty cell: it is the only daemonless,
pip-installable tool that runs *unmodified* shell commands and combines all
three of —

- NUMA-local `--membind` placement,
- a kernel-enforced per-job cgroup `MemoryMax` with OOM-victim biasing, and
- a crash-safe dependency-DAG queue.

If you do not need all three of those together on a single box, an existing tool
is very likely a better fit. See
[Alternatives / when not to use this](#alternatives--when-not-to-use-this) for an
honest comparison; nodepilot is not more capable than the tools below, only
narrower and zero-ops.

---

## The problem

You have a fat box — a 2-socket workstation or a single fat server — and a pile
of long-running compute jobs to push through it. Your two obvious options both
hurt:

- **SLURM / a real cluster scheduler is overkill.** It's a multi-daemon system
  built for many nodes and many users. Installing, configuring, and babysitting
  `slurmctld` + `slurmd` + a database + cgroup plugins on *one* machine is a
  week of yak-shaving for a problem you don't have.

- **The bare OS scheduler hurts you in two specific ways:**
  1. **It OOM-kills the wrong thing.** Launch a couple of memory-hungry jobs at
     once, one overshoots its expected footprint, and the kernel's OOM killer
     reaps *whatever it likes* — often a different innocent job, or a system
     service. One greedy job takes down your whole batch.
  2. **It scatters threads across sockets.** Left to the default scheduler, a
     job's threads migrate across every socket/CCD, a couple of complexes
     absorb the work while the rest idle, and first-touch memory pages land on a
     remote NUMA node — a durable 1.5–1.8× latency tax for the life of the job.

nodepilot is the thin layer in between: a **queue + admission control + cgroups
v2 memory containment + NUMA pinning + a watchdog loop**, in one `pip install`,
with stdlib + PyYAML as its only dependencies.

> **Example workloads, not dependencies.** nodepilot is engine-agnostic: it only
> ever sees a shell command and a resource request. ORCA, CP2K, VASP, Gaussian,
> and PyTorch are mentioned throughout this README purely as *examples* of the
> kind of back-to-back compute jobs people run this way. nodepilot does not
> bundle, wrap, or know anything about them. Every runnable example below uses
> `sleep` / `echo` / `hostname` / `stress-ng` so you can try it on any box.

---

## Install

Requires **Python 3.10+** (Linux; cgroups v2 + sysfs NUMA are Linux features).

```bash
git clone https://github.com/OortCloudd/nodepilot.git
cd nodepilot
pip install -e .
```

That installs the `nodepilot` package and the `nodepilot` console command. The
only third-party dependency is **PyYAML**; everything else is the standard
library (no `psutil`, no compiled extensions).

> nodepilot runs on a laptop or in a container too — it just degrades
> gracefully. Without sysfs NUMA nodes it sees one synthetic node; without a
> cgroup slice it falls back to declarative RAM accounting (see
> [the mental model](#mental-model)). The OOM containment and per-node pinning
> are what you gain on a real multi-socket host.

---

## 60-second quickstart

**1. Write a queue** (or copy [`examples/queue.simple.yaml`](examples/queue.simple.yaml)):

```yaml
# queue.yaml
global:
  max_concurrent: 2          # at most 2 jobs running at once
  ram_budget_gb: 16          # don't commit more than 16 GiB across jobs
  memory_slice: ""           # "" = declarative-only; see cgroup setup below
  state_path: ./nodepilot_state.json

jobs:
  - id: hello
    command: "echo 'hello from nodepilot' && sleep 2"
    cores: 1
    ram_gb: 1

  - id: warmup
    command: "hostname && sleep 3"
    cores: 2
    ram_gb: 2

  - id: report
    command: "echo 'all done'"
    depends_on: [hello, warmup]   # waits for both to finish
    priority: 50                  # lower number = runs sooner
```

**2. Run it.** The loop launches jobs as resources free up, respects
dependencies, then exits when the queue drains:

```bash
nodepilot run queue.yaml
```

**3. Check status** from another shell (reads the live state file):

```bash
nodepilot status queue.yaml
```

```
ID                       STATUS     PRIO CORES  RAM_GB CPU          REASON
-------------------------------------------------------------------------------
report                   pending      50     1       4 -
hello                    running     100     1       1 0
warmup                   running     100     2       2 1-2
-------------------------------------------------------------------------------
total=3  pending=1 running=2
```

**4. Kill a running job** by id (it's marked failed and frees its resources):

```bash
nodepilot kill queue.yaml warmup
```

**5. Reset** to discard saved progress and start the queue fresh from the YAML:

```bash
nodepilot reset queue.yaml
```

That's the whole surface. `run` resumes automatically from the state file if one
exists — kill the process, reboot the box, start it again, and finished jobs stay
finished while interrupted ones are reconciled (see
[crash-safe resume](#crash-safe-resume)).

### Try it without writing anything

```bash
nodepilot run examples/queue.simple.yaml --max-ticks 20
```

`--max-ticks N` stops the loop after N ticks regardless of queue state — handy
for a quick smoke test or a dry run.

---

## Mental model

nodepilot is a small **state machine driven by a periodic tick**. Understanding
the tick and the three guard rails is enough to use it well.

### The tick

Every `poll_interval` seconds the orchestrator runs one pass, always in this
order:

```
        ┌─────────────────────────────────────────────────────────────┐
        │  reap  →  enforce pins  →  launch (priority, then id)  → save │
        └─────────────────────────────────────────────────────────────┘
```

1. **Reap.** For each running job whose process has exited, classify the outcome
   — `done` (exit 0), `oom_killed` (SIGKILL / kernel OOM hint), `signal_N`,
   or `exit_N`. An OOM kill arms a cooldown that pauses new launches so memory
   actually frees before retrying.
2. **Enforce pins.** Re-`taskset` any job process that has drifted off its
   assigned cores (MPI ranks love to escape their cpuset).
3. **Launch.** Walk pending jobs in priority order; for each whose dependencies
   are met *and* that admission control admits, find a NUMA-local core block,
   wrap the command, and start it.
4. **Persist.** If anything changed, atomically snapshot the job list to JSON.

The loop **exits when nothing is pending and nothing is running.**

### Guard rail 1 — cgroups v2 memory containment (the real OOM fix)

The declarative `ram_gb` sum is *not* the safety mechanism — real footprints
overshoot and undershoot by tens of percent. The reliable guard is the kernel:

- Every job runs inside a **systemd scope** with a hard `MemoryMax = ram_gb`.
  When a job overshoots, the kernel OOM-kills **that scope**, not the host and
  not a random neighbour.
- All scopes live under one parent **slice** (`nodepilot.slice` by default)
  whose own `MemoryMax` sits *below* physical RAM. The slice's `memory.current`
  is the single source of truth for committed memory — admission reads it
  directly rather than trusting declared values.
- The orchestrator sets its own `oom_score_adj` negative (it must survive a
  memory storm to clean up) and each job's positive (jobs die first).

This requires a one-time slice setup — see
[the cgroup prerequisite](#cgroup-slice-setup-recommended). Without it,
nodepilot still runs and falls back to declarative + live-`/proc/meminfo`
accounting; you just lose the kernel-enforced ceiling.

### Guard rail 2 — NUMA-aware placement

Each job is handed a **contiguous block of physical cores on a single NUMA
node** and pinned with `numactl --physcpubind=<cores> --membind=<node>`, so its
CPUs and its memory stay local. Large jobs (over `interleave_threshold_gb`) get
`--interleave=all` instead, to avoid one job saturating a single node's memory.
If `numactl` isn't installed, nodepilot falls back to `taskset` (CPU pinning
only). Topology is auto-discovered from sysfs, or you can declare it explicitly
(`numa_nodes:`).

### Guard rail 3 — admission control

Before any job starts, a stack of **simple, blunt checks** runs cheapest-first
and short-circuits on the first failure. Blunt-but-reliable beats clever: fancy
per-node accounting gets bypassed (by memory interleaving, by jobs started
outside the scheduler), a few hard limits don't.

| # | Check | What it does |
|---|-------|--------------|
| 1 | **Pause sentinel** | A `.nodepilot.pause` file freezes all launches (manual brake; `touch` to pause, `rm` to resume). |
| 2 | **OOM cooldown** | After a system OOM, no launches for `oom_cooldown_seconds` so memory truly frees. |
| 3 | **Exclusive mutex** | An `exclusive: true` job runs alone — nothing starts beside it, it doesn't start beside anything. |
| 4 | **Concurrency cap** | Hard ceiling of `max_concurrent` simultaneous jobs. |
| 5 | **Core budget** | Running cores + this job ≤ `core_budget`. |
| 6 | **RAM guard** | If the cgroup slice is live: project `slice.memory.current + ram_gb` against `slice.max − ram_safety_gb`. Otherwise: declared-RAM sum **and** a live `MemAvailable` check. |
| 7 | **maxcore sanity** | *Advisory only.* Warns (never blocks) when `ram_gb` looks too small for `maxcore × nprocs × 1.3`. |

When a job can't start, it simply waits and is retried next tick; the reason is
logged so you can see *why* it's waiting.

---

## What it does

- **YAML queue, two-line minimum.** A job needs only `id` and `command`;
  everything else defaults. A bare list of jobs is accepted too.
- **Dependencies and priorities.** `depends_on` gates a job on others reaching
  `done`; `priority` orders the ready set (lower runs first, ties broken by id
  for determinism).
- **Exclusive jobs.** A memory- or bandwidth-bound phase marked
  `exclusive: true` runs alone.
- **Hard memory containment.** Per-job `MemoryMax`, a per-slice ceiling below
  physical RAM, and `oom_score_adj` biasing so the scheduler outlives a storm.
  See [`docs/cgroups_setup.md`](docs/cgroups_setup.md).
- **NUMA-local pinning.** Contiguous core blocks, memory bound to the node, with
  optional memory interleaving for very large jobs.
- **MPI binding helpers.** Keep launchers from scattering ranks across sockets
  and spilling first-touch pages cross-node. See
  [MPI binding](#mpi-binding-stopping-cross-socket-spill).
- **Crash-safe resume.** Atomic JSON state after every change; on restart,
  finished jobs stay finished and crashed-but-`running` jobs are reconciled.
- **Pluggable runner.** `subprocess` (default; detached child tracked by PID) or
  `tmux` (each job in its own attachable session for live output).
- **Greppable structured logs.** One line per event:
  `launch job=solve cores=16 ram_gb=64 cpu=0-15 node=0 mem=membind:0`.
- **Small and auditable.** Pure Python, stdlib + PyYAML only — no `psutil`, no
  compiled extensions.

---

## YAML schema reference

A queue file is a mapping with a `global:` block and a `jobs:` list. (A bare
list of jobs is also accepted — then all globals take their defaults.)

### `global:` — scheduler tunables

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `max_concurrent` | int | `4` | Max jobs running simultaneously. |
| `ram_budget_gb` | float | `0` → 85% of host RAM | RAM the scheduler may commit across running jobs (the declarative pre-check; the cgroup slice is the real cap). |
| `core_budget` | int | `0` → host cores | Total physical cores the scheduler may hand out. |
| `ram_safety_gb` | float | `20.0` | Margin kept free below the slice cap so the kernel never pre-emptively OOM-kills the slice. |
| `memory_slice` | str | `"nodepilot.slice"` | systemd slice that contains every job scope. `""` disables cgroup containment (declarative-only). |
| `numa_nodes` | map | `{}` → auto-detect | `{node: "core-range"}`, e.g. `{0: "0-15", 1: "16-31"}`. Empty = discover from sysfs. |
| `reserved_cores` | str | `""` | Cores never handed to jobs (SMT siblings, an OS/GPU lane), e.g. `"96-191"`. |
| `interleave_threshold_gb` | float | `0.0` | Spread a job's memory across all nodes (`--interleave=all`) once its `ram_gb` reaches this. `0` disables. |
| `orchestrator_oom_score_adj` | int | `-800` | `oom_score_adj` for the scheduler itself (negative = protect). |
| `job_oom_score_adj` | int | `500` | `oom_score_adj` for each job (positive = sacrifice first). |
| `oom_cooldown_seconds` | int | `300` | Freeze new launches this long after a system OOM. |
| `runner` | str | `"subprocess"` | Execution backend: `"subprocess"` or `"tmux"`. |
| `poll_interval` | int | `10` | Seconds between ticks. |
| `state_path` | str | `"nodepilot_state.json"` | Where the JSON state snapshot is written (resume on restart). |
| `log_path` | str | `"nodepilot.log"` | Structured log file. `""` = stderr only. |
| `pause_file` | str | `".nodepilot.pause"` | Touch this file to pause all new launches. |

### `jobs:` — one entry per job

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `id` | str | **required** | Unique job id. |
| `command` | str | **required** | Shell command, run via `bash -lc` (pipes / `&&` / env expansion all work). |
| `cores` | int | `1` | Physical cores to reserve (a contiguous NUMA-local block when possible). |
| `ram_gb` | float | `4.0` | Memory budget; enforced as the scope's hard `MemoryMax`, and used by admission. |
| `maxcore` | int | `0` | Optional per-process memory **hint** in MiB (mirrors the `%maxcore`-style knob of some codes). Used only by the advisory sanity rule; never enforced. |
| `nprocs` | int | `0` → `cores` | MPI ranks the job will spawn. Drives the sanity rule and per-rank binding plan. |
| `depends_on` | list[str] | `[]` | Job ids that must reach `done` before this job is eligible. |
| `priority` | int | `100` | **Lower runs first.** Ties broken by id. |
| `exclusive` | bool | `false` | Run alone — admitted only when nothing else runs, blocks others while running. |
| `workdir` | str | `""` → CWD | Directory the command runs from. |
| `env` | map | `{}` | Extra environment variables merged into the child's environment. |

> Unknown keys are tolerated (so you can keep `notes:` or custom tags inline),
> and runtime fields (`status`, `cpu_list`, `start_time`, …) are filled in by
> the scheduler — don't set them by hand.

A fuller annotated example lives in
[`examples/queue.simple.yaml`](examples/queue.simple.yaml).

---

## Cgroup slice setup (recommended)

The kernel-enforced OOM ceiling needs a **systemd slice whose `MemoryMax` sits
below physical RAM**. This is a one-time, per-user setup. Pick a ceiling that
leaves headroom for the OS and anything else on the box (e.g. on a 256 GiB host,
cap the slice around 200 GiB).

Create `~/.config/systemd/user/nodepilot.slice`:

```ini
[Unit]
Description=nodepilot job containment slice

[Slice]
# Hard ceiling for ALL nodepilot jobs combined. Keep it BELOW physical RAM
# so an overshoot kills a job scope, never the host. Tune to your machine.
MemoryMax=200G
MemorySwapMax=0
```

Then start it (and enable lingering so it survives logout, if you run detached):

```bash
systemctl --user daemon-reload
systemctl --user start nodepilot.slice
loginctl enable-linger "$USER"     # optional: keep the user manager alive
```

Confirm nodepilot sees it — `run` logs `slice active … cap_gb=200.0` at startup
(and `cgroup slice inactive …` if it doesn't, meaning it fell back to
declarative accounting). You can change the ceiling live with
`systemctl --user set-property nodepilot.slice MemoryMax=180G`.

To run **without** cgroups (a laptop, a container, a quick test), set
`memory_slice: ""` in the YAML — admission then uses declarative + live
`MemAvailable` accounting instead.

Full walkthrough, troubleshooting, and how the cgroup directories are located:
**[`docs/cgroups_setup.md`](docs/cgroups_setup.md)**.

---

## MPI binding (stopping cross-socket spill)

When you pin a job with `numactl` and then launch MPI *inside* it, the MPI
launcher's own affinity logic can override the parent cpuset — ranks scatter
across sockets and their first-touch pages land on a remote node, costing a
durable latency penalty and risking a single-node OOM. nodepilot ships
launcher-agnostic helpers (in `nodepilot.mpi`) to defer all binding to the outer
`numactl`:

- `openmpi_no_bind_env()` / `openmpi_no_bind_flags()` — make the launcher *not*
  bind ranks and map them by NUMA node (`--bind-to none --map-by node`), so the
  outer `numactl` owns all placement.
- `rank_binding_plan()` / `openmpi_rankfile()` — lay out explicit per-rank core
  assignments when you *do* want per-rank pinning.
- `omp_thread_env()` — `OMP_NUM_THREADS` + close/cores binding for hybrid
  MPI+OpenMP jobs.

The practical recipe (which env vars to set, the `--bind-to none` rationale, and
the gotchas) is in **[`GOTCHAS.md`](GOTCHAS.md)**.

---

## Library use

The CLI is a thin shell over a clean Python API — drive the scheduler directly
if you'd rather:

```python
from nodepilot import Orchestrator

orch = Orchestrator.from_queue("queue.yaml")   # resumes from state if present
orch.run()                                     # blocks until the queue drains
```

Lower-level pieces are exported too — `Config`, `Job`, `JobStatus`,
`load_queue`, `AdmissionController`, `Decision`, `Placement`, `allocate`,
`detect`, `placement_prefix` — so you can build queues programmatically, query
NUMA topology, or test an admission decision in isolation:

```python
from nodepilot import detect, allocate

topology = detect()                            # {0: "0-15", 1: "16-31"} or auto
placement = allocate(8, occupied=set(), numa_nodes=topology)
print(placement)        # Placement(cpu='0-7', membind=0, contig=True)
```

---

## Gotchas & operational notes

The sharp edges — resume semantics when you edit the YAML mid-run, the `tmux`
runner's exit-code blind spot, MPI binding, SMT/`reserved_cores`, choosing a
slice ceiling — are collected in **[`GOTCHAS.md`](GOTCHAS.md)**. A couple worth
flagging up front:

### Crash-safe resume

`run` writes the full job list to `state_path` after every tick that changes
something, atomically (temp file + `os.replace`, so a crash mid-write never
truncates it). On restart it **loads jobs from the state file**, not the YAML:
finished jobs stay finished, and a job left `running` by a crash but with no
live process/scope is **reconciled** — marked failed so it frees its cores/RAM
and stops blocking admission.

> **Editing the YAML mid-run.** Because resume reads the state file, edits to the
> queue YAML are ignored until you `reset` (which deletes the state file).
> nodepilot logs a warning when it notices the YAML is newer than the saved
> state, so you're not surprised.

### `subprocess` vs `tmux` runner

The default `subprocess` runner tracks each job by PID and recovers its exit
code precisely. The `tmux` runner gives you `tmux attach` for live output, but
the wrapper's exit status isn't recoverable — so a `tmux` job's success is
inferred (it falls back to the kernel-log OOM hint, else assumes success). Use
`subprocess` when you care about exact exit-code classification.

---

## How it fits together

```
        queue.yaml ──► load_queue ──► Config + [Job, …]
                                          │
                                          ▼
                                   Orchestrator.run()
                                          │  every poll_interval:
            ┌─────────────────────────────┼──────────────────────────────┐
            ▼                             ▼                               ▼
          reap                      enforce_pins                       launch
   (classify exits,            (re-taskset drifted          (admission ► NUMA allocate
    arm OOM cooldown)            MPI ranks back on            ► build cgroup-scoped,
            │                    their cores)                  numactl-pinned command
            │                                                  ► start)
            └──────────────────────────────┬──────────────────────────────┘
                                           ▼
                                  atomic JSON state  ◄──── resume on restart
```

Each box is one small, independently-testable module: `config`, `numa`,
`cgroups`, `admission`, `mpi`, `runner`, `state`, `logs`, `orchestrator`, `cli`.

---

## Alternatives / when not to use this

nodepilot is narrow on purpose. For most jobs one of these is the right tool,
and several of them are probably already installed on your box.

- **SLURM.** SLURM has every mechanism nodepilot has and many it does not
  (cgroup containment, NUMA affinity, dependency chains, fair-share, preemption,
  accounting, multi-node). It is strictly more capable. If you already run
  SLURM, or your workload will outgrow one machine, use SLURM — it complements
  and precedes nodepilot rather than competing with it. nodepilot's only edge is
  zero-ops on a single box: there is no `slurmctld`/`slurmd`/database/cgroup
  plugin to install, configure, and babysit. That edge disappears the moment you
  need a second node.

- **GNU `parallel`.** `parallel --memfree <N>` already does measured admission
  against live free RAM and will delay, kill, and requeue jobs under memory
  pressure — so "memory awareness" alone is not a reason to choose nodepilot, and
  you very likely already have `parallel`. nodepilot's honest advantage over it
  is narrower and concrete: a **hard, kernel-enforced per-job cgroup ceiling**
  (a job that overshoots is OOM-killed in its own scope, not throttled by a soft
  free-RAM heuristic), **NUMA-local placement**, and a **dependency DAG**. If you
  need none of those three, `parallel --memfree` is simpler.

- **`task-spooler` (`tsp`) / `nq`.** These give you the serial/parallel queue
  with almost no setup, but no memory containment, no NUMA placement, and no OOM
  biasing. Use them when the queue is all you want and the jobs are well-behaved
  on memory.

- **`systemd-run --scope -p MemoryMax=…`.** This is exactly the containment
  primitive nodepilot wraps (see [`GOTCHAS.md`](GOTCHAS.md) §2), and you can
  drive it by hand. What it does not give you is a scheduler: no queue, no
  dependencies, no admission control, no placement, no resume. nodepilot is the
  scheduler on top of it.

---

## Caveats / verification

Two honest limitations to know before you rely on the safety mechanisms.

- **The headline mechanisms are manually verified, not covered by CI.** The unit
  suite is green and exercises the pure logic (allocation, admission decisions,
  state, parsing), but the cgroup containment, NUMA `--membind` placement, and
  `oom_score_adj` paths are **not** exercised by CI — they require a systemd user
  bus and a multi-socket host, which the CI environment does not have. They have
  been checked by hand on a real machine. Treat them as verified-by-hand, and
  re-verify on your own hardware — a copy-paste procedure for all four checks is
  in [`docs/acceptance.md`](docs/acceptance.md).

- **Verify `MemoryMax` is actually *enforced* on your systemd, once.** On some
  systemd versions a `MemoryMax` set on a transient `--user --scope` is
  *accepted* (no error) but **not enforced** — the central guarantee then
  silently degrades to a soft limit, and an overshooting job will not be
  OOM-killed in its scope. Confirm enforcement with one deliberate-overshoot test
  before trusting it (e.g. a `stress-ng --vm-bytes` job whose footprint exceeds
  its `ram_gb`, which should be killed promptly). The procedure and the
  cgroup-path checks are in
  [`docs/cgroups_setup.md`](docs/cgroups_setup.md).

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).
