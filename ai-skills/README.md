# ai-skills/

Skills for AI coding assistants working on **projects that depend on
rakka-accel** — not for editing rakka-accel itself. They follow the
standard `SKILL.md` + frontmatter convention used by Claude Code,
Claude Agent SDK, and other agentic tools.

These skills are deliberately separate from the repo's own dev
tooling (`.claude/skills/version-bump`, `xtask/`) so that
distributing them to consumers does not entangle this repo's
internal release workflow.

## What's here

| Skill | Use when… |
|---|---|
| [`rakka-accel-getting-started`](skills/rakka-accel-getting-started/SKILL.md) | Wiring rakka-accel into a new Rust project — what to add to `Cargo.toml`, picking features, sub-crate selection |
| [`rakka-accel-device`](skills/rakka-accel-device/SKILL.md) | Driving a GPU through `DeviceActor` — `GpuRef<T>`, typed allocations, host↔device memcpy, dispatching `Sgemm` |
| [`rakka-accel-kernels`](skills/rakka-accel-kernels/SKILL.md) | Picking or extending a kernel actor — cuBLAS / cuBLASLt / cuDNN / cuFFT / cuRAND / cuSOLVER / cuSPARSE / cuTENSOR / NVRTC / NCCL — and the `KernelEnvelope::run_kernel` pattern |
| [`rakka-accel-supervision`](skills/rakka-accel-supervision/SKILL.md) | Reasoning about failure recovery — two-tier `DeviceActor ↔ ContextActor`, sticky-error context loss, the `ContextPoisoned`/`OutOfMemory`/`Unrecoverable` panic-tag protocol, `GpuRef` generation tokens |
| [`rakka-accel-python`](skills/rakka-accel-python/SKILL.md) | Using the Python bindings — `System`/`Device`/`GpuBuffer`, numpy float32 roundtrip, GIL release, mock-mode tests |
| [`rakka-accel-troubleshooting`](skills/rakka-accel-troubleshooting/SKILL.md) | Diagnosing failures — feature-flag misses, `GpuRefStale`, mailbox stalls, OOM loops, no-GPU CI vs GPU-runtime gate |
| [`rakka-accel-backends`](skills/rakka-accel-backends/SKILL.md) | Choosing between portable (`AccelBackend` trait) and vendor-specific (`rakka-accel-cuda`) APIs; future ROCm/Metal/oneAPI/Vulkan story |

Each `SKILL.md` is a thin router: it points at canonical docs in
this repo (`docs/*.md`, `examples/*`) and at the relevant crate's
rustdoc. It deliberately does **not** restate API surfaces that
belong in rustdoc, because those drift faster than docs.

## Installing

### As a plugin (most agent runtimes)

Most agent harnesses (Claude Code, Cursor, etc.) accept a folder of
`SKILL.md` files via a plugin manifest. Point your tool at this
folder; the skills in `skills/` will be picked up automatically.

```text
# Claude Code (example)
/plugin install /path/to/rakka-accel/ai-skills
```

### By copying

If your tooling expects skills under a project-local directory, copy
them in:

```bash
cp -r /path/to/rakka-accel/ai-skills/skills/* .claude/skills/
# or wherever your assistant looks for SKILL.md files
```

### By symlink (track upstream)

```bash
ln -s /path/to/rakka-accel/ai-skills/skills/rakka-accel-device \
      .claude/skills/rakka-accel-device
```

## Authoring conventions

- **One job per skill.** A skill is a router into the right docs +
  examples for one task. If a skill is trying to teach two things,
  it should be two skills (or it should defer to docs).
- **Defer to source-of-truth docs.** Link to `docs/*.md`,
  `crates/*/README.md`, and `examples/*` rather than restating them.
  Skills go stale; docs travel with the code.
- **Vendor-neutral.** No references to a specific assistant,
  harness, or tool. Describe rakka-accel, not the runtime loading
  the skill.
- **Frontmatter.** Each skill begins with `---` frontmatter
  containing `name` and `description`. The description is a
  one-line activation hint — what the user is doing when this
  skill should kick in.

## Versioning

These skills version with the repo. If a release changes a public
API covered by a skill, update the skill in the same PR. The skills
are not separately published.

## Related

- [`.claude/skills/version-bump/SKILL.md`](../.claude/skills/version-bump/SKILL.md)
  — internal release-management skill for **maintainers** of this
  repo. Not part of the consumer skill bundle above.
- [Sibling rakka skills](https://github.com/rustakka/rakka/tree/main/ai-skills)
  — for projects using the rakka actor runtime directly.
