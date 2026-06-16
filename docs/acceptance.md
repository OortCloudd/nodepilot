# Manual acceptance checks

The automated test suite covers the pure logic (allocator, admission, config,
state, MPI/cgroup argument construction, and real-subprocess orchestration). It
does **not** exercise the four mechanisms that are the point of the tool, because
they require a systemd user session and a multi-socket host that CI does not
have:

1. systemd `MemoryMax` is actually **enforced** (not silently accepted);
2. a memory overshoot is **contained** to its own scope (the host survives);
3. each job's threads are **pinned** to its assigned NUMA-local core block;
4. the OOM-score bias is **written** (jobs are killed before the scheduler).

Run the checks below once on a target host before relying on the guarantees.
They are copy-paste shell; each prints what to look for. Adjust `nodepilot.slice`
if you set a different `memory_slice`.

## 0. Prerequisite — is `MemoryMax` enforced on a `--user --scope`?

On some systemd versions a `MemoryMax` set on a transient `--user --scope` is
accepted but not enforced (a delegation/older-systemd issue); the containment
guarantee then degrades to a soft limit. Verify enforcement directly:

```bash
# Allocate 2 GiB inside a scope capped at 256 MiB. Expected: the process is
# OOM-killed inside the scope (exit via SIGKILL), NOT allowed to reach 2 GiB.
systemd-run --user --scope -p MemoryMax=256M -p MemorySwapMax=0 \
  python3 -c 'b=bytearray()
while True: b += bytearray(50_000_000)'
echo "exit=$?"     # expect 137 (128+SIGKILL); 'Killed' on stderr
```

If the process instead grows past 256 MiB and the host swaps or OOMs something
else, `MemoryMax` is **not** enforced on this host — fix the systemd/cgroup
delegation (see [`cgroups_setup.md`](cgroups_setup.md)) before trusting nodepilot's
containment. All later checks assume this one passed.

## 1. OOM containment — one greedy job does not take down the batch

```bash
cat > accept_oom.yaml <<'YAML'
global:
  memory_slice: nodepilot.slice
  ram_budget_gb: 8
jobs:
  - id: greedy
    command: "python3 -c 'b=bytearray()\nwhile True: b += bytearray(100_000_000)'"
    cores: 2
    ram_gb: 1          # hard cap: scope MemoryMax=1G
  - id: innocent
    command: "sleep 30; echo innocent-survived"
    cores: 2
    ram_gb: 1
YAML
nodepilot run accept_oom.yaml
```

Expected: `greedy` is reaped `oom_killed` (exit 137), the OOM cooldown is armed,
and `innocent` completes (`innocent-survived`). The host stays responsive; the
kernel OOM killer never reaps an unrelated process.

## 2. NUMA pinning — threads stay on the assigned block

While a job is running, confirm its processes are confined to the cores nodepilot
assigned (the `launch ... cpu=<list> node=<n>` log line) and that memory is bound
to that node:

```bash
nodepilot status accept_oom.yaml        # read the cpu_list nodepilot assigned
pid=$(pgrep -f 'innocent' | head -1)
grep Cpus_allowed_list /proc/$pid/status # must equal the assigned block, not 0-N
numastat -p "$pid" 2>/dev/null || cat /proc/$pid/numa_maps | head
```

Expected: `Cpus_allowed_list` matches the assigned block exactly; resident pages
sit on the bound node. Under an MPI launcher that re-binds ranks, the watchdog's
`re-pinned drifted procs` log line should appear and the affinity should be
corrected within one tick.

## 3. OOM-score bias — the scheduler outlives its jobs

```bash
# Orchestrator process: should carry the negative orchestrator_oom_score_adj.
opid=$(pgrep -f 'nodepilot run' | head -1); cat /proc/$opid/oom_score_adj
# A running job process: should carry a higher (more killable) score than the
# scheduler, so the kernel sacrifices the job, not nodepilot.
jpid=$(pgrep -f 'innocent' | head -1);     cat /proc/$jpid/oom_score_adj
```

Expected: the scheduler's `oom_score_adj` is negative (protected); the job's is
≥ 0 (preferred OOM victim). If nodepilot runs unprivileged and cannot lower its
own score, it logs the failure and continues — verify the value rather than
assuming it.

---

A short asciinema of checks 0–1 on a real two-socket host is the strongest
single artifact to attach to the README; these mechanisms are otherwise only
attested by prose.
