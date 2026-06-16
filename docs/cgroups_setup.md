# cgroups v2 setup: the OOM guard

nodepilot's real out-of-memory protection is not the declarative `ram_gb` sum in
your queue — those numbers always drift from reality by tens of percent. The
guard that actually holds is the kernel: every job runs inside its own
**cgroup v2 scope** with a hard `MemoryMax`, all of those scopes live under one
parent **slice**, and that slice has its *own* `MemoryMax` set safely below
physical RAM. When a job overshoots, the kernel OOM-kills the scope (one job),
not the host.

nodepilot reads the slice's `memory.current` as the single source of truth for
committed memory (see `nodepilot/cgroups.py::SliceMonitor`), and admission control
projects `memory.current + ram_gb` against `memory.max − ram_safety_gb` before
launching anything (`nodepilot/admission.py::_check_ram`).

This page walks through creating the slice, capping it below physical RAM,
verifying the accounting path, and making it survive logout. It targets a
**systemd user manager** with **cgroup v2** (the unified hierarchy, default on
modern Linux). If you have neither, jump to [Fallback](#fallback-no-user-bus-or-no-cgroup-v2).

---

## 0. Prerequisites

Confirm cgroup v2 is mounted as the unified hierarchy:

```console
$ stat -fc %T /sys/fs/cgroup
cgroup2fs
```

If this prints `cgroup2fs`, you are on cgroup v2. If it prints `tmpfs` you are on
the legacy v1/hybrid layout — see [Fallback](#fallback-no-user-bus-or-no-cgroup-v2).

Confirm a user manager and bus exist (nodepilot's `systemd_run_available()`
checks for exactly this):

```console
$ echo "$XDG_RUNTIME_DIR"
/run/user/1000
$ systemctl --user is-system-running
running          # "degraded" is also fine; "Failed to connect to bus" is not
```

If `systemctl --user` cannot connect to the bus, fix that first (often
[lingering](#4-survive-logout-enable-lingering) is the missing piece) or use the
fallback.

> nodepilot derives the slice cgroup path from your **numeric UID**, so note it:
>
> ```console
> $ id -u
> 1000
> ```
>
> The examples below use UID `1000`. Substitute your own everywhere you see it.

---

## 1. Create the `nodepilot.slice` unit

A slice is a systemd unit that owns a cgroup subtree; its `[Slice]` section sets
the limits inherited by every scope placed under it. Create it as a **user**
unit so it lives in your session's hierarchy and needs no root.

Write `~/.config/systemd/user/nodepilot.slice`:

```ini
# ~/.config/systemd/user/nodepilot.slice
[Unit]
Description=nodepilot job containment slice

[Slice]
# Hard ceiling for ALL nodepilot jobs combined. Set this BELOW physical RAM
# minus OS headroom (see "Cap math" below). The kernel OOM-kills inside this
# subtree when the sum of the scopes exceeds it -- so a runaway job dies, the
# host survives. Tune to your machine; 220G shown for a 256 GiB box.
MemoryMax=220G

# Never let jobs paper over a leak by swapping. A runaway should hit the wall
# and die, not crawl. This mirrors MemorySwapMax=0 that nodepilot sets on each
# scope in wrap_scope_command().
MemorySwapMax=0

# Be explicit even though the unified hierarchy enables it by default: this is
# what makes memory.current readable, which is nodepilot's source of truth.
MemoryAccounting=yes
```

`220G` here means 220 **GiB** (systemd's `G` is binary, `1G = 2^30` bytes), which
matches how `SliceMonitor.max_gb()` converts the raw byte value (`/ 1024**3`).

Reload the user manager and start the slice:

```console
$ systemctl --user daemon-reload
$ systemctl --user start nodepilot.slice
$ systemctl --user status nodepilot.slice
● nodepilot.slice - nodepilot job containment slice
     Loaded: loaded (.../systemd/user/nodepilot.slice; static)
     Active: active since ...
      Memory: 0B (max: 220.0G)
```

> A freshly created **empty** slice may not yet have a backing cgroup directory
> on disk — systemd can defer creating it until the first scope joins. That is
> normal: nodepilot's `SliceMonitor.is_active()` returns `False` until then, and
> admission control simply falls back to declarative accounting in the meantime.
> The directory appears the moment your first job launches. To force it to
> materialize immediately for inspection, drop one scope into it:
>
> ```console
> $ systemd-run --user --scope --slice=nodepilot.slice --unit=probe.scope sleep 30 &
> ```

The slice name must match `Config.memory_slice` (default `"nodepilot.slice"`). If
you name it something else, set it in your queue YAML:

```yaml
global:
  memory_slice: my-hpc.slice
  ram_safety_gb: 20      # kept free below the slice cap (admission headroom)
```

---

## 2. Cap math: why below physical RAM

The whole point of the slice cap is to keep a job overshoot *contained*. If
`MemoryMax` equalled or exceeded physical RAM, a runaway job could fill the slice
without the slice ever hitting its own limit — the kernel's *system-wide* OOM
killer would then fire and is free to pick **any** victim: PID 1, sshd, your
database, the nodepilot process itself. Containment would be zero.

So set the slice cap strictly below what the rest of the machine needs:

```
MemoryMax  =  PhysicalRAM  −  OS_and_other_headroom
```

Worked example for a **256 GiB** host that also runs the OS, a logging daemon, and
leaves slack for page cache and the orchestrator itself:

| Quantity | Value | Why |
| --- | --- | --- |
| Physical RAM | 256 GiB | `MemTotal` from `/proc/meminfo` |
| OS + services + page-cache headroom | 36 GiB | kernel, sshd, journald, FS cache |
| **Slice `MemoryMax`** | **220 GiB** | what jobs may collectively use |
| `ram_safety_gb` (nodepilot) | 20 GiB | admission stops launching *before* the hard wall |
| Effective launch ceiling | 200 GiB | `MemoryMax − ram_safety_gb` |

Two distinct safety margins are doing different jobs here, and it is worth being
precise about which is which:

- **`MemoryMax` below physical RAM** is the *kernel-enforced* margin. It protects
  the **host** from the slice. Crossing it kills a scope inside the slice.
- **`ram_safety_gb`** is nodepilot's *admission-time* margin (default `20`). It
  protects against ever *reaching* the hard wall: admission refuses to launch a
  job when `memory.current + ram_gb > MemoryMax − ram_safety_gb`, so in normal
  operation you throttle gracefully instead of relying on OOM kills. See the RAM
  guard in `nodepilot/admission.py::_check_ram`.

A reasonable starting point is `MemoryMax ≈ 0.85 × PhysicalRAM`, then adjust down
if the host feels tight under load. (That 85% figure is also the default
`ram_budget_gb` nodepilot derives for the *declarative* fallback path when no
live slice exists — see `Config.__post_init__`.)

> **Per-job cap, for context.** The slice cap is the aggregate. nodepilot *also*
> caps each individual job: `build_command()` wraps the job in
> `systemd-run --user --scope` with `MemoryMax=<ram_gb>G`, `MemorySwapMax=0`, and
> a `MemoryHigh` a few GiB below `Max` (throttle-before-kill). You do not set
> these by hand — they come from each job's `ram_gb`. The slice cap is the
> backstop for the *sum*; the scope cap is the backstop for *one* job.

---

## 3. Verify `memory.current` accounting

nodepilot finds the slice's cgroup directory by walking a fixed list of candidate
paths (`nodepilot/cgroups.py::_slice_dir_candidates`). Verifying that *one* of
them resolves — and that `memory.current` reads as a number — is the single most
important check on this page, because it is the difference between the live
kernel-truth RAM guard and the weaker declarative fallback.

For a user slice (UID `1000`), the **primary** path is:

```
/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/nodepilot.slice/memory.current
```

Depending on your systemd version, your user units may sit under an intermediate
**`app.slice`**, giving this **variant**:

```
/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/nodepilot.slice/memory.current
```

Both are checked, in this order, by `_slice_dir_candidates()` (followed by a
system-wide `/sys/fs/cgroup/nodepilot.slice` fallback for slices created
system-scope). You do not need to know *which* one your system uses — nodepilot
probes all of them — but it is reassuring to see the file with your own eyes.

Locate whichever exists (run this *after* the slice has at least one scope, or
after the `probe.scope` trick from step 1):

```console
$ UID_N=$(id -u)
$ BASE=/sys/fs/cgroup/user.slice/user-$UID_N.slice/user@$UID_N.service
$ ls -d "$BASE"/nodepilot.slice "$BASE"/app.slice/nodepilot.slice 2>/dev/null
/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/nodepilot.slice
```

Read the accounting files directly:

```console
$ SLICE_DIR=/sys/fs/cgroup/user.slice/user-$UID_N.slice/user@$UID_N.service/nodepilot.slice
$ cat "$SLICE_DIR/memory.current"   # bytes currently committed by all scopes
0
$ cat "$SLICE_DIR/memory.max"       # the cap from the unit file, in bytes
236223201280                        # == 220 GiB
```

`memory.current / 1024^3` is exactly what `SliceMonitor.used_gb()` returns, and
`memory.max / 1024^3` is `SliceMonitor.max_gb()` (which returns `None` if the
file reads the literal string `max`, i.e. uncapped).

**Watch it move under a real load.** Launch a memory-eating scope yourself and
watch the counter climb — this proves the accounting is wired end to end:

```console
$ # Allocate ~2 GiB inside the slice for 20 s (uses stress-ng if present).
$ systemd-run --user --scope --slice=nodepilot.slice --unit=memcheck.scope \
      stress-ng --vm 1 --vm-bytes 2G --timeout 20s &
$ for i in 1 2 3; do cat "$SLICE_DIR/memory.current"; sleep 3; done
118784
2147483648
2151677952
```

No `stress-ng`? Use a pure-shell allocator that needs nothing extra:

```console
$ systemd-run --user --scope --slice=nodepilot.slice --unit=memcheck.scope \
      bash -c 'block=$(head -c 1500000000 /dev/zero | tr "\0" "x"); sleep 15' &
$ cat "$SLICE_DIR/memory.current"
1500039680
```

Finally, confirm nodepilot agrees from Python:

```console
$ python3 -c "from nodepilot.cgroups import SliceMonitor; m=SliceMonitor('nodepilot.slice'); \
print('active:', m.is_active(), 'used_gb:', m.used_gb(), 'max_gb:', m.max_gb())"
active: True used_gb: 1.397 max_gb: 220.0
```

If `is_active()` is `True` and `max_gb()` shows your cap, the live RAM guard is
armed. If `is_active()` is `False` even after a scope has joined, the slice
directory is not where nodepilot looks — re-check the two candidate paths above
and the [Troubleshooting](#troubleshooting) table.

---

## 4. Survive logout: enable lingering

By default a user's systemd manager (and therefore its slices and scopes) is torn
down when the last session ends. For a long-running queue you want the slice — and
the jobs in it — to persist when you log out or your SSH connection drops. That is
what **lingering** does:

```console
$ loginctl enable-linger "$USER"
$ loginctl show-user "$USER" -p Linger
Linger=yes
```

With lingering on, `user@1000.service` stays up across logout, the
`nodepilot.slice` cgroup keeps existing, and a detached `nodepilot run` (e.g.
under `nohup`, `tmux`, or its own systemd user service) keeps scheduling.

To turn it back off later: `loginctl disable-linger "$USER"`.

> Lingering is also the usual fix when `systemctl --user` works in an interactive
> SSH session but not from cron or a detached process: without it, there is no
> user bus outside an active login session.

---

## 5. Change the cap at runtime

You do **not** need to edit the unit file and reload to retune the cap — you can
set it live, and the change is picked up immediately because
`SliceMonitor.max_gb()` re-reads `memory.max` on every query and admission control
re-queries each tick (`AdmissionController._slice_cap_gb`):

```console
$ systemctl --user set-property nodepilot.slice MemoryMax=200G
$ cat "$SLICE_DIR/memory.max"
214748364800                        # == 200 GiB, effective now
```

This is the knob to reach for when another tenant shows up on a shared box and you
need to shrink nodepilot's footprint without draining the queue. Lowering the cap
takes effect at once; running jobs are not killed unless their *combined*
`memory.current` already exceeds the new ceiling, in which case the kernel
reclaims/OOM-kills inside the slice as usual.

`set-property` is transient by default (it does not rewrite the unit file). To
make a new value the persistent default, edit `MemoryMax=` in
`~/.config/systemd/user/nodepilot.slice` and `systemctl --user daemon-reload`.

> Mirror at the per-job level: nodepilot itself does not expose a live per-scope
> retune — a job's cap is fixed from its `ram_gb` at launch. To change one job's
> ceiling, edit its `ram_gb` in the queue and let it relaunch, or
> `systemctl --user set-property nodepilot-<job-id>.scope MemoryMax=...` on the
> live scope (the scope unit name is `scope_unit_name(job_id)` →
> `nodepilot-<sanitized-id>.scope`).

---

## Fallback: no user bus, or no cgroup v2

Some environments have no user systemd manager (minimal containers, certain
locked-down or batch hosts) or are still on cgroup v1. nodepilot is designed to
degrade gracefully rather than refuse to run.

**Automatic degradation.** `build_command()` only wraps a job in a scope when
`config.memory_slice` is set *and* `cgroups.systemd_run_available()` returns
`True` (i.e. `systemd-run` is on `PATH` and a user bus is reachable). If the bus
is missing, jobs simply launch *without* a cgroup scope, and admission control
uses the declarative path: declared-RAM accounting plus a live `/proc/meminfo`
`MemAvailable` check (`_check_ram` → `_free_ram_gb`). You lose hard kernel
containment but keep a real free-memory backstop.

**Explicit opt-out.** To skip cgroup integration entirely — and silence any
probing — set the slice to empty in your queue YAML:

```yaml
global:
  memory_slice: ""        # disable cgroup containment; declarative accounting only
  ram_budget_gb: 200      # the declarative RAM ceiling now does the gating
  ram_safety_gb: 20       # keep this much MemAvailable free at launch time
```

With `memory_slice: ""`, `AdmissionController` does not even construct a
`SliceMonitor`, and the RAM guard relies on `ram_budget_gb` + live free memory.
Set `ram_budget_gb` conservatively (≈ 85% of physical RAM is the built-in
default) since there is no kernel wall behind it.

> Declarative-only mode is genuinely useful — it just trusts the `ram_gb` numbers
> in your queue more than the kernel-backed mode does. Keep those numbers honest.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `SliceMonitor.is_active()` is `False` after a job started | Slice cgroup not where nodepilot looks | Check both candidate paths in §3 (with and without `app.slice`); confirm `id -u` matches the `user-UID.slice` segment |
| `max_gb()` returns `None` | `memory.max` reads `max` (uncapped) | Set `MemoryMax=` in the unit (or `set-property`) and `daemon-reload` |
| `Failed to connect to bus` from `systemctl --user` | No user manager in this session | `loginctl enable-linger $USER`; or set `memory_slice: ""` and use declarative mode |
| `stat -fc %T /sys/fs/cgroup` prints `tmpfs` | Legacy cgroup v1 / hybrid hierarchy | Boot with `systemd.unified_cgroup_hierarchy=1`, or use declarative mode |
| Host OOM-kills unrelated processes when a job blows up | Slice `MemoryMax` ≥ physical RAM (no containment) | Lower `MemoryMax` below physical RAM minus OS headroom (§2) |
| Jobs silently swap instead of dying on overshoot | `MemorySwapMax` not 0 on the slice | Add `MemorySwapMax=0` to `[Slice]` and `daemon-reload` |
| Empty slice has no directory under `/sys/fs/cgroup` | systemd defers creating the cgroup until first scope | Expected; launch a job or use the `probe.scope` one-liner in §1 |

---

## Quick reference

```console
# create + start
$EDITOR ~/.config/systemd/user/nodepilot.slice     # [Slice] MemoryMax=NNNG, MemorySwapMax=0
systemctl --user daemon-reload
systemctl --user start nodepilot.slice

# persist across logout
loginctl enable-linger "$USER"

# verify accounting (UID-templated path; app.slice variant also valid)
U=$(id -u); D=/sys/fs/cgroup/user.slice/user-$U.slice/user@$U.service/nodepilot.slice
cat "$D/memory.current"   "$D/memory.max"

# retune live
systemctl --user set-property nodepilot.slice MemoryMax=NG

# what nodepilot sees
python3 -c "from nodepilot.cgroups import SliceMonitor as S; m=S('nodepilot.slice'); \
print(m.is_active(), m.used_gb(), m.max_gb())"
```
