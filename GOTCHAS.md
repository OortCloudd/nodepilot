# GOTCHAS — field notes from running a fat box without SLURM

These are the failures that shaped nodepilot's design. Each one cost a run (or a
night) before the fix landed. They are written **symptom → root cause → fix**, and
each maps to the exact code that now defends against it.

Nothing here is hypothetical: every gotcha corresponds to a guard you can read in
the source. If you are extending nodepilot, read these before "simplifying" the
parts that look over-engineered — they look that way *because* of what is below.

The running scenario throughout: one large multi-socket machine, several
NUMA nodes, a queue of long memory-hungry MPI jobs (think ORCA / CP2K / VASP /
Gaussian, or a multi-GPU PyTorch run) launched back to back. The OS scheduler and
a hand-written `for job in queue; do ...; done` loop are not enough. The examples
use `sleep` / `stress-ng` / `hostname` so they run anywhere.

Index:

1. [`mpi_affinity` — ranks scatter across CCDs and sockets](#1-mpi_affinity)
2. [`cgroup_pin` — declared RAM lies; the cgroup does not](#2-cgroup_pin)
3. [`numa_per_node` — single-node OOM while the other node is free](#3-numa_per_node)
4. [`mpi_bindto_none` — the launcher overrides your numactl](#4-mpi_bindto_none)
5. [`tmux_exclude` — re-pinning the shared tmux server breaks every session](#5-tmux_exclude)
6. [`oom_cooldown` — relaunching straight back into the OOM](#6-oom_cooldown)
7. [`zombie_restart` — a crash leaves "running" jobs that never run](#7-zombie_restart)
8. [`queue_after_state` — editing the YAML changes nothing](#8-queue_after_state)
9. [`no_silent_swap` — MemorySwapMax=0, or the OOM hides in swap](#9-no_silent_swap)
10. [`oom_score` — protect the scheduler, sacrifice the jobs](#10-oom_score)
11. [`smt_siblings` — "32 cores" that are really 16](#11-smt_siblings)

---

## 1. `mpi_affinity`

**Symptom.** Launch four MPI jobs on a multi-socket box, each asking for, say, 16
ranks. Aggregate throughput is far below 4× a single job. `htop` shows work piled
onto two or three core complexes (CCDs) while whole CCDs sit idle; per-job
wall-clock is wildly uneven run to run. Letting the kernel "balance" the threads
made it *worse*, not better.

**Root cause.** With no per-job CPU placement, the kernel is free to schedule any
job's threads on any core, and it does. Threads from different jobs land on the
same CCD and contend for its L3; a single job's threads spread across sockets and
pay cross-socket latency on every shared-memory access. There is no notion of "this
job owns these cores" anywhere, so nothing keeps them apart.

**Fix.** Hand each job a *contiguous block of physical cores on one NUMA node* and
pin both CPU and memory there. `numa.allocate(n_cores, occupied, numa_nodes)`
walks each node, prefers the largest contiguous run of free cores
(`_largest_contiguous_run`), and falls back to any N free cores on that node if no
run is long enough — but always NUMA-local. The orchestrator passes the union of
cores already held by running jobs (plus `reserved_cores`) as `occupied`
(`Orchestrator._occupied_cores`), so two jobs never get the same core.
`numa.placement_prefix(placement)` then builds

```
numactl --physcpubind=<cpu_list> --membind=<node>
```

prepended to the command in `runner.build_command`. Contiguity is preferred in the
allocator's sort key (`not contiguous` is the first tuple element, so contiguous
wins), which keeps a rank's cores on the same CCD.

```python
from nodepilot.numa import allocate, placement_prefix

# Node 0 = cores 0-15, node 1 = 16-31. Nothing running yet.
nodes = {0: "0-15", 1: "16-31"}
p = allocate(8, occupied=set(), numa_nodes=nodes)
print(p)                       # Placement(cpu='0-7', membind=0, contig=True)
print(placement_prefix(p))     # ['numactl', '--physcpubind=0-7', '--membind=0']
#                              # (numactl present; taskset -c 0-7 is the fallback)
```

A contiguous block on one socket, memory bound to that socket. No scatter. The
allocator's tie-break also spreads load: when one node is partly busy and another
is free, the *emptier* node wins (the `-len(free)` term in the sort key), which
keeps a fresh job off a node that's already filling — anti-fragmentation for free:

```python
# Cores 0-7 on node 0 are busy. A new 8-core job lands on the empty node 1,
# not on node 0's remaining 8-15, because node 1 has more headroom.
p = allocate(8, occupied=set(range(8)), numa_nodes=nodes)
print(p)                       # Placement(cpu='16-23', membind=1, contig=True)

# When no NUMA-local block can fit, allocate returns None and the orchestrator
# defers the job to the next tick rather than scattering it:
print(allocate(8, occupied=set(range(32)), numa_nodes=nodes))   # None
```

> Code: `numa.allocate`, `numa._largest_contiguous_run`, `numa.placement_prefix`,
> `runner.build_command`, `Orchestrator._occupied_cores`.

---

## 2. `cgroup_pin`

**Symptom.** Admission control did its arithmetic perfectly — sum of declared
`ram_gb` across running jobs stayed comfortably under budget — and the host still
OOM-killed something. Conversely, jobs that declared a huge `ram_gb` "to be safe"
blocked the queue while the machine sat half-empty. The declared number and the
real footprint disagreed by tens of percent in both directions.

**Root cause.** `ram_gb` is a *request*, not a measurement. Real resident memory of
a long compute job overshoots or undershoots the declared value by 30–50%
routinely (integral accuracy, grid size, the phase of the job). Summing requests is
accounting fiction; the kernel charges you for actual pages.

**Fix.** Two layers, both anchored to the kernel rather than the request.

1. **Hard per-job ceiling.** Every job runs inside a systemd scope with
   `MemoryMax=<ram_gb>G` (`cgroups.wrap_scope_command`, layered on by
   `runner.build_command`). If a job overshoots its declared RAM, the kernel
   oom-kills *that scope* — the job, not the host. The request becomes an enforced
   limit instead of a hope.

2. **Truth for admission.** The scopes live under one parent slice
   (`Config.memory_slice`, default `nodepilot.slice`). Its `memory.current` is the
   real committed total. `AdmissionController._check_ram` reads it via
   `cgroups.SliceMonitor` and projects `slice.memory.current + job.ram_gb` against
   `slice.memory.max - ram_safety_gb`. Declared sums are only the *fallback* when no
   live cgroup is present.

```python
from nodepilot.cgroups import wrap_scope_command, SliceMonitor

argv = wrap_scope_command(["bash", "-lc", "stress-ng --vm 1 --vm-bytes 4G -t 30s"],
                          job_id="vm-test", ram_gb=8, slice_name="nodepilot.slice")
# systemd-run --user --scope --slice=nodepilot.slice
#   -p MemoryMax=8G -p MemorySwapMax=0 -p MemoryHigh=4G  bash -lc "..."

mon = SliceMonitor("nodepilot.slice")
mon.used_gb()   # live committed RAM of all jobs, or None if the slice isn't up
mon.max_gb()    # the kernel-enforced ceiling, or None if unset
```

When `SliceMonitor.used_gb()` returns a number, admission trusts the cgroup over
the declared sum. That single substitution is what stopped the "the math was right
but it OOM'd anyway" failures.

> Code: `cgroups.wrap_scope_command`, `cgroups.SliceMonitor`,
> `admission.AdmissionController._check_ram`, `runner.build_command`.

---

## 3. `numa_per_node`

**Symptom.** A job died with a NUMA-policy memory error
(`CONSTRAINT_MEMORY_POLICY` in the kernel log) — an *out-of-memory on one node*
while the other node still had tens of GiB free. Total free RAM looked fine; the
job was just bound to a node that filled up.

**Root cause (and the trap).** The obvious fix is per-node RAM accounting: track
how much each node is committed and refuse to place a job on a node that would
overflow. We built exactly that. It then got bypassed: a large job that crosses the
interleave threshold has its pages spread across *all* nodes (`--interleave=all`),
so per-node bookkeeping no longer describes where its memory actually is, and jobs
launched outside the scheduler don't show up in our per-node tally at all. Clever
NUMA accounting is fragile precisely because it assumes it sees every allocation.

**Fix — the lesson.** Prefer a **blunt global cap over clever per-node
accounting**. The real guards are the concurrency cap and the slice's single
`MemoryMax` (gotchas 2 and 6): they hold regardless of how any one job's pages are
distributed. Per-node RAM gating is kept as an **opt-in** in `numa.allocate`
(`node_ram_cap_gb` / `node_ram_safety_gb`, both default 0 = disabled) and is
deliberately *not* wired into the orchestrator's hot path. When enabled it reads
**anonymous** pages only (`numa.node_memory_gb` → `Active(anon) + Inactive(anon)`),
so a node merely holding file cache doesn't look full — and it is **skipped
entirely for jobs that will interleave**, because the per-node number is
meaningless for them:

```python
# numa.allocate, gating branch
will_interleave = interleave_threshold_gb > 0 and ram_gb >= interleave_threshold_gb
gate_ram = node_ram_cap_gb > 0 and not will_interleave        # off unless asked
safe_cap = node_ram_cap_gb - node_ram_safety_gb
...
if gate_ram:
    anon, _ = node_memory_gb(node)
    if anon is not None and anon + ram_gb > safe_cap:
        continue   # this node would over-commit; try another
```

Reach for the per-node cap only on a box where one node genuinely fills before the
others and you've accepted its limits. The default configuration leans on the blunt
caps that cannot be bypassed.

> Code: `numa.allocate` (`node_ram_cap_gb` path), `numa.node_memory_gb`,
> `Config.interleave_threshold_gb`; contrast with `admission` (concurrency + slice
> cap) which is always on.

---

## 4. `mpi_bindto_none`

**Symptom.** Even *with* the outer `numactl --physcpubind --membind` from gotcha 1,
MPI jobs still spilled memory cross-node and ran 1.5–1.8× slower than a clean
single-node run. Re-pinning the threads after launch (gotcha 5) fixed the CPU
placement but **not** the latency — the pages were already remote and stayed
remote.

**Root cause.** The MPI launcher applies its *own* affinity and mapping on top of
whatever cpuset it inherits. It re-binds ranks, scattering them across sockets, and
each rank's **first-touch** allocations land wherever that rank happened to start.
First-touch is permanent: once a page is faulted in on a remote node, moving the
thread back later doesn't move the page. So you pay remote-memory latency for the
entire life of the job, no matter how aggressively a watchdog re-pins CPUs
afterwards.

**Fix.** Tell the launcher to keep its hands off and let the outer `numactl` own
all placement. `mpi.openmpi_no_bind_env()` returns the Open MPI MCA settings that
disable rank binding and map by NUMA node, so first-touch pages stay local:

```python
from nodepilot.mpi import openmpi_no_bind_env, openmpi_no_bind_flags

openmpi_no_bind_env()
# {'OMPI_MCA_hwloc_base_binding_policy': 'none',     # don't bind ranks
#  'OMPI_MCA_rmaps_base_mapping_policy': 'node',     # distribute by node
#  'OMPI_MCA_rmaps_base_ranking_policy': 'core'}

openmpi_no_bind_flags()      # ['--bind-to', 'none', '--map-by', 'node']
```

Put the env dict into the job's `env:` (it merges into the child environment in
`runner.start`), or add the flags to your `mpirun` line. The principle generalises
beyond Open MPI: **whoever sets the cpuset must be the only thing that binds.** Two
binders fighting is how you get remote pages. If you *do* want explicit per-rank
pinning instead, build it yourself with `mpi.rank_binding_plan` +
`mpi.openmpi_rankfile` and feed `mpirun --rankfile` — but don't also let it
`--bind-to` something else.

```yaml
# examples/queue.yaml — make Open MPI defer to the outer numactl
jobs:
  - id: mpi-job
    command: "mpirun $NP_NOBIND -np 4 hostname"   # placeholder; real run = your solver
    cores: 4
    nprocs: 4
    env:
      OMPI_MCA_hwloc_base_binding_policy: "none"
      OMPI_MCA_rmaps_base_mapping_policy: "node"
      OMPI_MCA_rmaps_base_ranking_policy: "core"
```

> Code: `mpi.openmpi_no_bind_env`, `mpi.openmpi_no_bind_flags`,
> `mpi.rank_binding_plan`, `mpi.openmpi_rankfile`, `runner.start` (env merge).

---

## 5. `tmux_exclude`

**Symptom.** Using the `tmux` runner, the pin-enforcement pass (which re-`taskset`s
any process that drifted off its assigned cores) started wrecking *unrelated* jobs.
A job would suddenly be confined to a handful of cores it was never assigned, and
sessions for completely different jobs would seize up at the same time. The more
jobs running, the worse the collateral damage.

**Root cause.** With the tmux backend, `_pids_for_job` finds a job's processes with
`pgrep -f <job.id>`. But **tmux runs all sessions under one shared server process**,
and that server's command line contains every session name — so `pgrep -f <job.id>`
*matches the shared server*. The enforcement loop then ran
`taskset -cp <one job's cpu_list> <server pid>`, narrowing the single process that
hosts **every** session to one job's cores. Every other session inherited that
cpuset. One job's pin enforcement throttled the whole machine's tmux.

**Fix.** Never re-pin the shared tmux server. `runner.enforce_pin` reads each
candidate PID's `comm` and **skips any process whose `comm` starts with `tmux`**:

```python
# runner.enforce_pin
for pid in _pids_for_job(job):
    comm = Path(f"/proc/{pid}/comm").read_text(encoding="ascii").strip()
    if comm.startswith("tmux"):
        continue                      # the shared server — never narrow it
    current = _cpus_allowed(pid)
    if current is None:
        continue
    if not current or not current.issubset(expected):
        subprocess.run(["taskset", "-cp", job.cpu_list, str(pid)], ...)
        repinned += 1
```

The real job processes still get re-pinned when their affinity drifts (MPI ranks
love to escape via `sched_setaffinity` after launch); only the shared server is
exempt. Note the subtlety: the substring match on the **command line** is what
catches the server, but the guard keys off `comm` (the executable name), which is
the reliable signal that this PID *is* tmux rather than a job that merely mentions
tmux. The same `comm.startswith("tmux")` skip is why the orchestrator can run
`enforce_pin` every tick without fear.

> Code: `runner.enforce_pin`, `runner._pids_for_job`,
> `Orchestrator._enforce_pins`.

---

## 6. `oom_cooldown`

**Symptom.** A job OOM-died, freed its slot, the scheduler immediately launched the
next pending job into the just-vacated memory — which OOM-died too. Repeat. A
single memory spike turned into a burst of back-to-back OOM kills as the queue
thrashed against a host that hadn't actually recovered yet.

**Root cause.** OOM-killed pages are not reclaimed instantly, and the cgroup
`memory.current` lags reality for a beat. Reaping a job and launching the next in
the *same tick* meant admitting against stale free-memory numbers, straight into
the same wall.

**Fix.** An OOM kill arms a cooldown. When `runner.reap` classifies a death as
`oom_killed`, the orchestrator calls `admission.trigger_oom_cooldown()`, which
blocks **all** launches for `Config.oom_cooldown_seconds` (default 300):

```python
# Orchestrator._reap_finished
if outcome.reason == "oom_killed":
    self.admission.trigger_oom_cooldown()
    self.log.warning("oom cooldown armed", ...)

# admission.AdmissionController.can_launch — checked second, before any placement
remaining = self.in_cooldown()
if remaining > 0:
    return Decision(False, f"OOM cooldown ({int(remaining)}s remaining)")
```

The cooldown is checked *early* in `can_launch` (right after the pause sentinel, so
nothing gets placed during it) and gives memory time to actually free before the
next attempt. It's a deliberate anti-thrash brake: better to idle for a few minutes
than to machine-gun the OOM killer.

> Code: `admission.AdmissionController.trigger_oom_cooldown` / `in_cooldown` /
> `can_launch`, `runner.reap` (the `oom_killed` classification),
> `Orchestrator._reap_finished`, `Config.oom_cooldown_seconds`.

---

## 7. `zombie_restart`

**Symptom.** Restart the orchestrator (a deploy, a host reboot, a crash) and the
queue would wedge: jobs marked `running` in the state file held cores and RAM in
the admission accounting, so nothing new could launch — but the processes those
jobs supposedly were had died with the previous orchestrator. Phantom jobs blocked
real ones forever.

**Root cause.** State is snapshotted as `running` while a job is live. If the
orchestrator dies between a job ending and the next state write — or if the whole
machine went down — the persisted status stays `running` for a process that no
longer exists. On resume, admission control faithfully reserves resources for
ghosts.

**Fix.** Reconcile zombies at startup, *before* the first tick.
`Orchestrator._reconcile_zombies` walks every job the state file calls `running`
and checks two independent liveness signals — the tracked process
(`runner.is_alive`) **and** the cgroup scope
(`cgroups.scope_is_active(job.id, slice)`). If **neither** is alive, the job is a
zombie: mark it `failed` with reason `zombie_at_restart`, clear its `cpu_list` so
the allocator can reuse the cores, and persist:

```python
# Orchestrator._reconcile_zombies
alive_proc  = self.runner.is_alive(job)
alive_scope = cgroups.scope_is_active(job.id, self.config.memory_slice) \
              if self.config.memory_slice else False
if not alive_proc and not alive_scope:
    job.status = JobStatus.FAILED
    job.failure_reason = "zombie_at_restart"
    job.cpu_list = ""          # release the core reservation
```

Two signals matter because a job resumed from state has **no retained `Popen`
handle** — `runner.is_alive` can only probe it by PID, which is ambiguous after a
reboot (PIDs get reused). The cgroup scope is the corroborating evidence: a scope
directory under the slice means the job is genuinely still there. Requiring *both*
to be dead before declaring a zombie avoids killing a job that survived the
orchestrator restart (e.g. a detached `setsid` child or a `--no-block` scope that
outlived its parent). State writes are atomic (`state.save_state` writes `*.tmp`
then `os.replace`), so a crash mid-write can never produce the *other* failure mode
— a truncated state file.

> Code: `Orchestrator._reconcile_zombies`, `runner.Runner.is_alive`,
> `runner._pid_alive`, `cgroups.scope_is_active`, `state.save_state` (atomic).

---

## 8. `queue_after_state`

**Symptom.** Edit `queue.yaml` — bump a job's `ram_gb`, add a job, change a
priority — `nodepilot run` again, and the edits did nothing. The change "didn't
take" and there was no obvious reason why.

**Root cause.** Resume semantics. `Orchestrator.from_queue` loads the **persisted
job list** from the state file when one exists, so that progress (done jobs, run
history) is not lost on restart. The YAML's `global:` block is still honoured for
tunables, but the per-job edits in the YAML are ignored until you `reset`. This is
correct — you don't want a restart to forget which jobs already finished — but it's
surprising the first time it bites.

**Fix.** Make the silent behaviour loud. `_warn_if_queue_newer` compares mtimes and
logs a warning when the YAML was edited after the last state save:

```python
# orchestrator._warn_if_queue_newer
if Path(queue_path).stat().st_mtime > Path(state_path).stat().st_mtime:
    log.warning("queue edited after last state save",
                queue=queue_path, hint="changes ignored until 'reset'")
```

The fix for the *user* is `nodepilot reset <queue.yaml>` (deletes the state file)
followed by `nodepilot run` to start fresh from the edited YAML. The warning exists
so you find that out from a log line instead of from an hour of confusion.

> Code: `Orchestrator.from_queue`, `orchestrator._warn_if_queue_newer`,
> `Orchestrator.reset`, `cli` (`reset` subcommand).

---

## 9. `no_silent_swap`

**Symptom.** A job that should have been killed for blowing its memory budget
instead crawled — the whole machine went sluggish, every job slowed down, and the
"OOM" never came. The misbehaving job had quietly spilled into swap and dragged the
host into thrash instead of dying cleanly.

**Root cause.** A cgroup `MemoryMax` with swap still available lets a job exceed its
RAM ceiling by paging to disk. Instead of a clean, attributable kill you get
silent, machine-wide slowdown that's much harder to diagnose — and the offending
job keeps running, starving its better-behaved neighbours of I/O and memory
bandwidth.

**Fix.** Set `MemorySwapMax=0` on every scope so there is no swap escape hatch: a
job that hits its `MemoryMax` is oom-killed promptly and visibly.
`cgroups.wrap_scope_command` always emits it, and also sets `MemoryHigh` a few GiB
below `MemoryMax` to throttle *before* the hard kill:

```python
# cgroups.wrap_scope_command
"-p", f"MemoryMax={ram_gb:g}G",
"-p", f"MemorySwapMax={swap_max}",     # default 0 -> no swap, no silent thrash
"-p", f"MemoryHigh={high_gb:g}G",      # soft throttle before the hard ceiling
```

A clean kill you can attribute to one job beats a host-wide slowdown you have to
hunt for. (If you genuinely want a swap allowance, pass `swap_max=<bytes>` — but the
default of zero is the right one for compute nodes.)

> Code: `cgroups.wrap_scope_command` (`MemorySwapMax`, `MemoryHigh`).

---

## 10. `oom_score`

**Symptom.** Under a real memory storm the kernel OOM killer reached past the jobs
and killed the **orchestrator itself** (or a system service). With the scheduler
dead, the jobs it should have sacrificed kept running, nothing got reaped, and the
queue was leaderless until someone noticed.

**Root cause.** By default everything is a roughly equal OOM candidate; the kernel
picks by memory footprint, not by role. The long-lived, low-footprint orchestrator
is a perfectly valid victim in that calculus — exactly the wrong one to lose,
because it's the thing that's supposed to clean up.

**Fix.** Bias the kernel's victim selection explicitly. The orchestrator lowers its
own `oom_score_adj` to a strongly negative value at startup, and every job's scope
gets a positive one so jobs are sacrificed first:

```python
# Orchestrator.run, once at startup
cgroups.set_self_oom_score_adj(cfg.orchestrator_oom_score_adj)   # default -800

# runner._start_subprocess, per job
cgroups.set_oom_score_adj(proc.pid, self.config.job_oom_score_adj)  # default +500
```

So when memory runs out the kernel kills a *job* (which nodepilot then reaps,
classifies as `oom_killed`, and follows with the cooldown of gotcha 6), and the
scheduler survives to do that bookkeeping. `set_oom_score_adj` clamps to the valid
`[-1000, 1000]` range and returns `False` (never raises) if the write isn't
permitted — on a host where you can't set it, you simply don't get the protection,
but nothing breaks. Tune both numbers via `Config.orchestrator_oom_score_adj` and
`Config.job_oom_score_adj`.

> Code: `cgroups.set_self_oom_score_adj`, `cgroups.set_oom_score_adj`,
> `Orchestrator.run`, `runner._start_subprocess`,
> `Config.orchestrator_oom_score_adj` / `job_oom_score_adj`.

---

## 11. `smt_siblings`

**Symptom.** A box advertising "32 cores" gave disappointing throughput when fully
packed, and memory-bandwidth-bound jobs in particular ran much slower at 32-wide
than scaling from 16-wide predicted. The scheduler thought it had 32 independent
cores to hand out; it had 16 physical cores with two hardware threads each.

**Root cause.** `numa.detect` reads `/sys/devices/system/node/nodeN/cpulist`, which
lists **logical** CPUs — SMT siblings included. nodepilot cannot reliably tell a
physical core from its sibling thread from sysfs alone (no hardware topology probe,
by design — stdlib + PyYAML only), so it treats every logical CPU as allocatable. On
a bandwidth-bound code, packing both siblings of a core buys you little and can hurt.

**Fix.** Exclude the sibling threads via `Config.reserved_cores`. Those cores are
added to the `occupied` set in `Orchestrator._occupied_cores` and never handed to
any job, so the scheduler only places work on the cores you kept:

```yaml
# global: keep only the "first" SMT thread of each core on a 16-core / 32-thread box
# (sibling layout is machine-specific: check /sys/devices/system/cpu/cpuN/topology/
#  thread_siblings_list before copying this verbatim)
global:
  reserved_cores: "16-31"   # siblings live in the upper range on this host
  core_budget: 16           # match the budget to the physical cores you actually use
```

This is a deliberate "you know your hardware, tell us" knob rather than a fragile
auto-detect that would guess wrong on exotic topologies. The same mechanism reserves
cores for the OS or a GPU feeder lane. `numa.detect`'s docstring calls this out
explicitly so the logical-vs-physical assumption is never a silent surprise.

> Code: `Config.reserved_cores`, `numa.detect` (logical-CPU note),
> `Orchestrator._occupied_cores`, `numa.parse_cpu_list`.

---

## The through-line

Three principles fall out of the list above, and they're the spine of the whole
design:

- **Measure, don't trust the request.** Declared `ram_gb` is a starting point; the
  cgroup `memory.current` is the truth (gotchas 2, 9). Admission believes the kernel
  over the YAML whenever the kernel is available.
- **One placer, not two.** Whoever sets the cpuset must be the only thing that binds.
  `numactl` owns placement; the MPI launcher is told to stand down (gotchas 1, 4),
  and pin enforcement is careful not to touch shared infrastructure (gotcha 5).
- **Blunt and bypass-proof beats clever and fragile.** A global concurrency cap and a
  single slice ceiling hold no matter how memory is distributed; per-node accounting
  is opt-in because it can be routed around (gotcha 3). Survive the storm
  (gotchas 6, 7, 10), and make the surprising-but-correct behaviour loud
  (gotcha 8).

If a future change makes one of these guards look unnecessary, re-read its symptom
first. They are all here because the simpler version didn't survive contact with a
real machine.
