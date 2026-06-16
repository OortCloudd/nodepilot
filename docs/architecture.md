# nodepilot architecture

How the pieces fit together: the modules and their responsibilities, the
scheduling **tick loop**, the inside-out **command layering**, the job
**state machine**, and the one invariant everything else leans on — *the cgroup
slice's `memory.current` is the source of truth for committed RAM, not the sum
of declared budgets.*

nodepilot runs a queue of compute jobs back-to-back on a single big machine
without a cluster scheduler. It is deliberately small: standard library plus
PyYAML, no `psutil`, Python 3.10+. It sees nothing but a shell command string
and a resource request — ORCA, CP2K, VASP, Gaussian and PyTorch appear in this
document only as *example* workloads; the runnable examples below use
`sleep` / `echo` / `hostname`.

---

## 1. Module map

Eleven modules, each with one job. Arrows mean "imports / calls".

```
                         cli  ──────────────┐
                          │  build_parser   │ status / kill / reset
                          ▼  main           │
                   orchestrator  ◀──────────┘
                   (the tick loop)
       ┌───────────┬───────┴───────┬───────────┬──────────┐
       ▼           ▼               ▼           ▼          ▼
   admission     runner          numa        state      logs
   can_launch    build_command   detect      save/load  get_logger
       │         start/reap      allocate    (atomic)   kv
       │         enforce_pin     placement_      │
       ▼              │          prefix          │
   cgroups  ◀─────────┘             │            │
   SliceMonitor                     ▼            ▼
   wrap_scope_command              mpi         config
   oom_score                  rank_binding_   Job / Config
       │                      plan, no-bind   JobStatus
       ▼                           env        load_queue
   /sys/fs/cgroup,            (helpers only,  (YAML 1:1)
   systemd-run                 not wired into
                               the core loop)
```

| Module | Responsibility | Key public surface |
|---|---|---|
| **`config`** | The data model. `Job` and `Config` dataclasses, the `JobStatus` string constants, and YAML loading. The YAML `global:` keys map 1:1 onto `Config` fields. | `Job`, `Config`, `JobStatus`, `load_queue(path) -> (Config, list[Job])`, `job_to_dict` |
| **`numa`** | Topology discovery and core/memory placement. Parses CPU-list strings, finds a NUMA-local contiguous core block, and builds the `numactl` prefix. | `detect()`, `allocate(...) -> Placement\|None`, `Placement`, `placement_prefix(placement) -> list[str]`, `parse_cpu_list`, `format_cpu_list`, `node_memory_gb`, `has_numactl` |
| **`cgroups`** | The OOM guard. Wraps a command in a `systemd-run --user --scope` with a hard `MemoryMax`, reads the parent slice's live memory accounting, and sets `oom_score_adj`. | `wrap_scope_command(...)`, `SliceMonitor`, `scope_unit_name`, `scope_is_active`, `set_oom_score_adj`, `set_self_oom_score_adj`, `systemd_run_available`, `journal_reports_oom` |
| **`mpi`** | Launcher-agnostic binding *data*. Lays MPI ranks across a core block and builds "keep your hands off" Open MPI environments/flags so the outer `numactl` owns all placement. Pure helpers — builds dicts/strings, executes nothing. | `rank_binding_plan(...)`, `RankBinding`, `openmpi_no_bind_env()`, `openmpi_no_bind_flags()`, `openmpi_rankfile`, `omp_thread_env` |
| **`admission`** | The gate. A stack of cheap checks deciding whether a pending job may start *now*. | `AdmissionController.can_launch(job, jobs) -> Decision`, `Decision`, `trigger_oom_cooldown`, `in_cooldown`, `running_cores`, `running_ram_gb`, `running_count`, `maxcore_sane` |
| **`runner`** | Execution + supervision. Builds the final argv (inside-out), starts it (`subprocess` or `tmux`), checks liveness, classifies the exit, and re-pins drifted processes. | `build_command(job, placement, config) -> list[str]`, `Runner`, `reap(job, runner) -> Outcome`, `enforce_pin(job) -> int`, `Outcome` |
| **`state`** | Crash-safe persistence. Atomic JSON snapshot of the job list (`write tmp` → `os.replace`), tolerant load. | `save_state(path, jobs)`, `load_state(path) -> list[Job]`, `state_exists(path)` |
| **`logs`** | Dependency-free structured logging. One greppable `key=value` line per event. Module is `logs.py` (not `logging.py`) to avoid shadowing the stdlib. | `get_logger(log_path=None) -> Logger`, `kv(**pairs) -> str` |
| **`orchestrator`** | The loop that ties it together. Owns the job list, runs the tick, persists on change, reconciles zombies on startup. | `Orchestrator(config, jobs)`, `Orchestrator.from_queue(path)`, `.run(max_ticks=None)`, `.reset()`, `.kill(job_id)` |
| **`cli`** | The `nodepilot` command: `run` / `status` / `kill` / `reset`, each taking a positional `queue` path. | `main(argv=None) -> int`, `build_parser()` |
| **`__init__`** | The public API re-exports. | `Orchestrator`, `Config`, `Job`, `JobStatus`, `load_queue`, `AdmissionController`, `Decision`, `Placement`, `allocate`, `detect`, `placement_prefix`, `__version__` |

The dependency graph is acyclic and shallow. `config` is the leaf everyone
imports. `cgroups` and `numa` sit above it with no dependency on each other.
`admission` and `runner` compose `cgroups` + `numa`. `orchestrator` sits on top
of all of them, and `cli` drives `orchestrator`. `mpi` is a self-contained set
of helpers for hand-built MPI launch commands — it is **not** called by the core
loop, so wiring it in is opt-in (see [§6](#6-where-mpi-fits)).

---

## 2. The tick loop

`Orchestrator.run()` arms its own OOM protection
(`cgroups.set_self_oom_score_adj(config.orchestrator_oom_score_adj)`, default
`-800`), logs startup, reconciles zombies *once*, then loops: run a `_tick()`,
persist if anything changed, stop when the queue is drained, otherwise
`sleep(poll_interval)` and go again.

A single tick runs three phases in a fixed order — **reap → enforce pins →
launch** — and returns whether any persistable state changed.

```
                          ┌─────────────────────────────────────────┐
                          │                 TICK                     │
                          └─────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │ 1. REAP FINISHED                              (_reap_finished)         │
   │    for each RUNNING job:                                               │
   │        runner.is_alive(job)?  ── yes ─▶ leave it running               │
   │             │ no                                                       │
   │             ▼                                                          │
   │        outcome = runner.reap(job, runner)   # done / failed / oom      │
   │        job.status      = outcome.status                                │
   │        job.failure_reason / exit_code / end_time = …                   │
   │        if outcome.reason == "oom_killed":                              │
   │            admission.trigger_oom_cooldown()   # freeze new launches    │
   │        changed = True                                                  │
   └──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ 2. ENFORCE PINS                               (_enforce_pins)          │
   │    for each RUNNING job with a cpu_list:                               │
   │        runner.enforce_pin(job)   # re-taskset drifted PIDs             │
   │    side-effect only — NOT counted as a state change to persist         │
   └──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ 3. LAUNCH PENDING                             (_launch_pending)        │
   │    pending = PENDING jobs sorted by (priority, id)   # low prio first  │
   │    for job in pending:                                                 │
   │        if not deps_satisfied(job):          continue   # wait on deps  │
   │        decision = admission.can_launch(job, jobs)                      │
   │        if not decision.ok:                  continue   # gated; reason  │
   │        placement = numa.allocate(job.cores, occupied, nodes, …)        │
   │        if placement is None:                continue   # defer: no fit  │
   │        job.cpu_list, job.numa_node = placement.cpu_list, placement.node │
   │        argv = runner.build_command(job, placement, config)             │
   │        runner.start(job, argv)                                         │
   │        job.status = RUNNING ; job.start_time = now                     │
   │        changed = True                                                  │
   └──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ 4. PERSIST IF CHANGED                         (run loop)               │
   │    if changed: state.save_state(state_path, jobs)   # atomic JSON      │
   │    if drained (no PENDING and no RUNNING): exit                        │
   └──────────────────────────────────────────────────────────────────────┘
```

Why this order:

- **Reap before launch.** A job that just finished frees cores and RAM. Reaping
  first means those resources are visible to admission and placement *in the
  same tick*, so the freed slot can be refilled immediately instead of one tick
  later.
- **Enforce pins in the middle.** Pin drift only matters for jobs that are
  already running; doing it after reap means we never waste a `taskset` on a job
  that just exited, and doing it before launch keeps the running set tidy before
  we add to it.
- **Launch last, priority-ordered.** Pending jobs are sorted by
  `(priority, id)` — **lower `priority` runs first**, ties broken by `id` for
  determinism. Each candidate must clear three independent gates in sequence:
  dependencies satisfied → admission says yes → a NUMA block is free. Any gate
  failing just `continue`s to the next job; the blocked job is retried on the
  next tick.

Note the asymmetry: a reap or a launch is a *state change* and triggers a
persist; pin enforcement is a pure side-effect on the live OS and is never
persisted. So `_tick()` returns `reaped or launched`.

The loop terminates when `_is_drained()` is true — no job is `PENDING` or
`RUNNING`. `done`, `failed`, and `dormant` jobs do not keep the loop alive.

---

## 3. The command layering (inside-out)

When a job is launched, `runner.build_command(job, placement, config)` assembles
the final argv by wrapping the user's command in successive layers. Read it
**inside-out**: the user's command is the kernel, and each layer outside it adds
one capability.

```
   ┌───────────────────────────────────────────────────────────────────┐
   │ systemd-run --user --scope  --slice=<memory_slice>                 │   OUTER:
   │   -p MemoryMax=<ram_gb>G  -p MemorySwapMax=0  -p MemoryHigh=<…>G    │   OOM guard
   │   --unit=nodepilot-<job-id>.scope  --collect --no-block            │   (cgroups)
   │ ┌───────────────────────────────────────────────────────────────┐ │
   │ │ numactl --physcpubind=<cpu_list> --membind=<node>             │ │   MIDDLE:
   │ │   (or --interleave=all when the placement interleaves;        │ │   CPU + mem
   │ │    or `taskset -c <cpu_list>` when numactl is absent)         │ │   placement
   │ │ ┌───────────────────────────────────────────────────────────┐ │ │   (numa)
   │ │ │ bash -lc "<job.command>"                                  │ │ │   INNER:
   │ │ │   the user's command, exactly as written in the YAML      │ │ │   user cmd
   │ │ └───────────────────────────────────────────────────────────┘ │ │
   │ └───────────────────────────────────────────────────────────────┘ │
   └───────────────────────────────────────────────────────────────────┘
```

Built from the inside out, in code:

1. **Inner — the user command.**
   `inner = ["bash", "-lc", job.command]`. Running through a login shell is
   deliberate: it makes pipelines, `&&`, redirection, and environment expansion
   in the YAML command behave exactly as a user would expect from a terminal.

2. **Middle — placement.**
   `placed = [*placement_prefix(placement), *inner]`.
   `numa.placement_prefix` returns `numactl --physcpubind=<cpu_list>
   --membind=<node>` (or `--interleave=all` if `placement.interleave` is set),
   falling back to `taskset -c <cpu_list>` when `numactl` is not installed, and
   to `[]` (no prefix) when there is nothing to pin. This binds both the CPUs
   and the memory of the job to one NUMA node.

3. **Outer — containment.**
   Only if `config.memory_slice` is set *and* `cgroups.systemd_run_available()`,
   `placed` is wrapped by `cgroups.wrap_scope_command(...)` into a
   `systemd-run --user --scope` that joins the slice and sets `MemoryMax`,
   `MemorySwapMax=0`, and a `MemoryHigh` a few GiB under the max (throttle
   before the hard kill). If cgroup containment is unavailable or disabled,
   `build_command` returns `placed` unwrapped and the system runs in
   *declarative-only* mode.

The result is one argv that `Runner.start` executes. The ordering matters:
the cgroup scope is **outermost** so the kernel accounts for *everything* the
job spawns (every `numactl` child, every MPI rank) against the same `MemoryMax`;
`numactl` is **inside** the scope so placement applies to the whole shell
subtree; `bash -lc` is **innermost** so the command text is interpreted last,
as written.

A concrete (synthetic) expansion for a 4-core, 8 GiB job named `crunch` placed
on cores `0-3` of node `0`:

```
systemd-run --user --scope --quiet --collect --no-block \
  --slice=nodepilot.slice --unit=nodepilot-crunch.scope \
  -p MemoryMax=8G -p MemorySwapMax=0 -p MemoryHigh=4G \
  numactl --physcpubind=0-3 --membind=0 \
  bash -lc "echo start && sleep 30 && echo done"
```

---

## 4. The job state machine

Every job carries a `status` string drawn from `config.JobStatus`. The states
are plain strings (not an `Enum`) so they round-trip through the JSON state file
with no custom encoder.

```
                         (declared in YAML with status: dormant)
                                        │
                                        ▼
                                  ┌───────────┐
                                  │  DORMANT  │   ignored by the scheduler until
                                  └───────────┘   something flips it to PENDING
                                        │ (manual edit / external hook)
                                        ▼
   (default for a new job) ─────▶ ┌───────────┐
                                  │  PENDING  │ ◀── eligible; retried every tick
                                  └───────────┘     until all gates pass
                                        │
                deps satisfied  +  admission ok  +  NUMA block free
                                        │  (runner.start succeeds)
                                        ▼
                                  ┌───────────┐
                                  │  RUNNING  │ ── holds cores + RAM; re-pinned
                                  └───────────┘    each tick if it drifts
                                        │
                       process exits / killed / crash-at-restart
                                        │
                       runner.reap classifies the exit status
                ┌───────────────────────┴───────────────────────┐
                ▼                                                ▼
          ┌───────────┐                                    ┌───────────┐
          │   DONE    │   exit 0                            │  FAILED   │   exit≠0,
          └───────────┘                                    └───────────┘   signal,
        terminal; frees                                  terminal; frees   OOM, or
        resources, unblocks                              resources; an     zombie-
        dependents                                       OOM also arms the at-restart
                                                         cooldown
```

State semantics, as the code uses them:

- **`pending`** — the default for a freshly loaded job (`Job.status` defaults to
  `JobStatus.PENDING`). The launch phase considers only pending jobs.
- **`running`** — set by `_launch` after `runner.start` succeeds. A running job
  is *active*: `Job.is_active()` is true, and it is counted by
  `running_cores` / `running_ram_gb` / `running_count` and by `_occupied_cores`.
  Only running jobs are reaped and pin-enforced.
- **`done`** — terminal success (`runner.reap` saw exit code 0).
  `Job.is_terminal()` is true. Dependents that name this job in `depends_on`
  become eligible.
- **`failed`** — terminal failure. `runner.reap` maps exits to a `failure_reason`
  code: `0 → done`; `137` or a kernel-log OOM hint → `oom_killed`; `>128` →
  `signal_<N>`; anything else → `exit_<N>`. The orchestrator also writes
  `failed` for a launch error (`launch_error: …`), a user kill
  (`killed_by_user`), and a zombie reconciled at startup (`zombie_at_restart`).
- **`dormant`** — declared but not yet eligible. The scheduler never touches a
  dormant job; an external action (a hook, a manual YAML/state edit) flips it to
  `pending` when it should join the queue. Useful for staging jobs you are not
  ready to release.

`is_active()` returns true for `running`; `is_terminal()` returns true for
`done` and `failed`. Only those four-plus-dormant strings are ever assigned.

### Dependencies

A pending job is eligible only when **every** id in its `depends_on` list refers
to a job currently in `done` (`_deps_satisfied`). A missing or not-yet-`done`
upstream keeps the job pending. There is no automatic failure propagation: if an
upstream job *fails*, its dependents simply never become eligible and remain
`pending` until the upstream is re-run to `done` or the dependency is removed.

### Crash recovery and zombies

State is snapshotted after every changed tick, so a restart resumes exactly
where it stopped. `Orchestrator.from_queue` loads the **state file** in
preference to the YAML job list when one exists (the YAML's `global:` block is
still honoured for tunables, and an edit to the YAML after the last save is
flagged with a warning — those edits are ignored until `reset`).

On startup, `_reconcile_zombies` walks jobs the state file calls `running` and
checks whether they are *really* alive — both `runner.is_alive(job)` (PID or
tmux session) **and** `cgroups.scope_is_active(job.id, slice)` (the scope cgroup
directory still exists). If neither is alive, the job died while the
orchestrator was down: it is marked `failed` with reason `zombie_at_restart`,
its `cpu_list` is cleared so the allocator can reuse those cores, and the
correction is persisted. This is what stops a crashed-mid-run job from holding
phantom resources forever.

(Jobs the runner launched *this* process are tracked by a retained `Popen`
handle for an accurate exit code. A job resumed from the state file has no such
handle, so its liveness is probed by PID/scope and its exit is best-effort —
exactly the case zombie reconciliation exists to clean up.)

---

## 5. Source of truth for RAM: `memory.current`, not the declared sum

This is the single most important design decision, so it gets its own section.

Declared `ram_gb` values are *estimates*. Real footprints overshoot or undershoot
by tens of percent, and jobs sometimes get launched outside the scheduler
entirely. So nodepilot never trusts the arithmetic sum of declared budgets as
its real memory guard. Two mechanisms work together:

**The kernel enforces, per job.** Each job's `systemd-run` scope carries a hard
`MemoryMax`. An overshoot oom-kills *that scope* — the job — not the host. Jobs
are given a positive `oom_score_adj` (`job_oom_score_adj`, default `+500`) and
the orchestrator a negative one (`-800`), so under genuine pressure a job dies
before the scheduler does, and the scheduler survives to reap it.

**The slice reports, in aggregate.** Every scope is nested under one parent
*slice* (`config.memory_slice`, default `nodepilot.slice`). `SliceMonitor` reads
that slice's `memory.current` from `/sys/fs/cgroup` — the kernel's live tally of
all committed memory across every job — and `memory.max` as the kernel-enforced
ceiling (which an operator may raise/lower at runtime with
`systemctl --user set-property`).

`AdmissionController._check_ram` uses that tally as primary truth:

```
if slice memory.current is readable (cgroup is live):
        cap        = slice memory.max  (or ram_budget_gb if the slice has no Max)
        safe_cap   = cap - ram_safety_gb
        projected  = memory.current + job.ram_gb
        admit only if projected <= safe_cap
else:   # declarative fallback — no live cgroup accounting
        admit only if (sum of running jobs' declared ram_gb) + job.ram_gb <= ram_budget_gb
        AND /proc/meminfo MemAvailable >= job.ram_gb + ram_safety_gb
```

The fallback path is defence-in-depth for hosts without cgroup v2 / systemd: it
combines the declarative sum *and* a live `MemAvailable` check, so even a job
whose neighbours under-declared cannot push the box into swap. But where the
slice is live — the normal case — the number that gates admission is the one the
kernel actually measured, with a `ram_safety_gb` margin kept free below the cap
so the kernel is never forced into a pre-emptive slice kill.

Per-node RAM gating works on the same principle one level down: when
`numa.allocate` is given a `node_ram_cap_gb`, it reads each node's *anonymous*
memory via `node_memory_gb` (from `/sys/.../nodeN/meminfo`, excluding
reclaimable page cache) and skips a node that the job would over-commit — the
guard against a single node filling up while its sibling sits free.

---

## 6. Where MPI fits

`mpi` is intentionally **not** wired into the default launch path — the core
loop pins jobs with `numactl` and lets the job's own launcher run inside that
cpuset. The module exists for the case where you build an MPI launch command
yourself and want it to *cooperate* with that outer pin instead of fighting it.

The failure it prevents: an MPI launcher's own affinity logic overrides the
parent cpuset, scatters ranks across sockets, and lands their first-touch pages
on a remote node — a lasting cross-socket memory penalty that even later
re-pinning cannot undo. The fix is to tell the launcher to keep its hands off
and let the outer `numactl` own placement:

- `openmpi_no_bind_env()` → `{OMPI_MCA_hwloc_base_binding_policy: none,
  OMPI_MCA_rmaps_base_mapping_policy: node, OMPI_MCA_rmaps_base_ranking_policy:
  core}` — merge into the job's `env:`.
- `openmpi_no_bind_flags()` → `["--bind-to", "none", "--map-by", "node"]` — the
  command-line equivalent.
- `rank_binding_plan(cpu_list, nprocs)` lays ranks across the job's core block
  (rank *i* gets a contiguous slice; the last rank absorbs any remainder), and
  `openmpi_rankfile(plan)` renders it for `mpirun --rankfile` when you *do* want
  explicit per-rank pinning. `omp_thread_env(threads)` adds
  `OMP_NUM_THREADS` / `OMP_PROC_BIND=close` / `OMP_PLACES=cores` for hybrid jobs.

In practice you put the `--bind-to none --map-by node` (or the MCA env) directly
in the job's `command` / `env` in the YAML; nodepilot's outer `numactl` does the
rest. `runner.enforce_pin` is the safety net for ranks that still drift — it
re-`taskset`s any of a job's PIDs whose affinity escaped the assigned block, and
deliberately **skips the shared tmux server** (which `pgrep` matches by session
name but must never be narrowed to one job's cores).

---

## 7. End-to-end: one job's life

Putting it together with a minimal synthetic queue (`queue.yaml`):

```yaml
global:
  max_concurrent: 2
  ram_budget_gb: 32
  memory_slice: nodepilot.slice
  state_path: ./nodepilot_state.json
jobs:
  - id: prep
    command: "echo preparing && sleep 5"
    cores: 2
    ram_gb: 4
  - id: crunch
    command: "echo crunching && sleep 20"
    cores: 4
    ram_gb: 8
    depends_on: [prep]
    priority: 50
```

```
nodepilot run queue.yaml      # start (or resume) the loop
nodepilot status queue.yaml   # print the job table
nodepilot kill  queue.yaml crunch
nodepilot reset queue.yaml    # discard saved state, start fresh next run
```

What happens:

1. `Orchestrator.from_queue("queue.yaml")` parses the YAML (`load_queue`), and —
   since no state file exists yet — schedules the YAML jobs as written. It arms
   its own OOM protection and logs the slice status.
2. **Tick 1.** `prep` is `pending` with no dependencies; admission admits it (the
   slice `memory.current` + 4 GiB is well under the cap); `numa.allocate` returns
   a 2-core block; `build_command` wraps `bash -lc "echo preparing && sleep 5"`
   in `numactl` and a `MemoryMax=4G` scope; `runner.start` launches it; `prep`
   → `running`. `crunch` is skipped — its dependency `prep` is not `done`. State
   is persisted.
3. **Subsequent ticks.** `prep` is re-pin-checked. When it exits 0, reap marks it
   `done`. Now `crunch`'s dependency is satisfied; it is admitted, placed on a
   4-core block, launched → `running`.
4. **Drain.** When `crunch` exits 0 and nothing is pending or running, the loop
   logs `queue complete done=2 failed=0` and stops.

If at any point a job overshoots its `MemoryMax`, the kernel oom-kills just that
scope; reap classifies it `failed` / `oom_killed`, arms the admission cooldown,
and the next eligible job waits out the grace period before launching — so one
runaway never cascades into a host-wide OOM.

---

## 8. Design principles in one place

- **The kernel is the guard, not the spreadsheet.** Hard per-job `MemoryMax` +
  a slice ceiling below physical RAM + `memory.current` as the admission input.
  Declared budgets are a pre-filter, never the last line of defence.
- **Pin CPU *and* memory, together, on one NUMA node.** Scattered threads and
  remote first-touch pages are the quiet throughput killers; `numactl
  --physcpubind --membind` and a re-pinning watchdog keep a job where it was
  placed.
- **Simple, ordered admission beats clever bookkeeping.** Pause → cooldown →
  exclusive → concurrency → cores → RAM → maxcore-advisory, cheapest first,
  short-circuit on the first failure. Elaborate per-node accounting gets bypassed
  in the real world; a few blunt limits hold.
- **Crash-safe by construction.** Atomic state writes, resume from state over
  YAML, and zombie reconciliation so a crash never strands resources.
- **Engine-agnostic.** The orchestrator only ever sees a shell command and a
  resource request. What that command *is* — a quantum-chemistry run, a training
  job, or `sleep 30` — is none of its business.
