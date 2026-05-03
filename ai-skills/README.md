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
| `rakka-accel-getting-started` | Wiring rakka-accel into a new Rust project — `Cargo.toml`, picking features, sub-crate selection |
| `rakka-accel-device` | Driving a GPU through `DeviceActor` — `GpuRef<T>`, typed allocations, host↔device memcpy, dispatching `Sgemm` |
| `rakka-accel-kernels` | Picking or extending a kernel actor — cuBLAS / cuBLASLt / cuDNN / cuFFT / cuRAND / cuSOLVER / cuSPARSE / cuTENSOR / NVRTC / NCCL — and the `KernelEnvelope::run_kernel` pattern |
| `rakka-accel-supervision` | Reasoning about failure recovery — two-tier `DeviceActor ↔ ContextActor`, sticky-error context loss, the `ContextPoisoned`/`OutOfMemory`/`Unrecoverable` panic-tag protocol, `GpuRef` generation tokens |
| `rakka-accel-python` | Using the Python bindings — `System`/`Device`/`GpuBuffer`, numpy float32 roundtrip, GIL release, mock-mode tests |
| `rakka-accel-troubleshooting` | Diagnosing failures — feature-flag misses, `GpuRefStale`, mailbox stalls, OOM loops, no-GPU CI vs GPU-runtime gate |
| `rakka-accel-backends` | Choosing between portable (`AccelBackend` trait) and vendor-specific (`rakka-accel-cuda`) APIs; future ROCm/Metal/oneAPI/Vulkan story |

Each `SKILL.md` is a thin router: it points at canonical docs in
this repo (`docs/*.md`, `examples/*`) and at the relevant crate's
rustdoc. It deliberately does **not** restate API surfaces that
belong in rustdoc, because those drift faster than docs.

## Installing

Pick the path that matches your assistant. The skills themselves are
vendor-neutral `SKILL.md` files — only the install mechanism differs.

### Claude Code (recommended: marketplace)

If you use Claude Code, install via the plugin marketplace — this
keeps the skills updated as rakka-accel releases, with no manual
copy step:

```text
/plugin marketplace add rustakka/rakka-accel
/plugin install rakka-accel-ai-skills@rakka-accel
```

You can also install from a local checkout (useful when developing
against a rakka-accel fork):

```text
/plugin marketplace add /path/to/rakka-accel
/plugin install rakka-accel-ai-skills@rakka-accel
```

Skills auto-activate based on the `description` frontmatter — no need
to invoke them explicitly.

### Claude Agent SDK / project-local `.claude/skills/`

For SDK-based agents or project-local Claude Code setups that read
from `.claude/skills/`, copy or symlink the skills in:

```bash
# copy (snapshot)
cp -r /path/to/rakka-accel/ai-skills/skills/* .claude/skills/

# symlink (track upstream)
ln -s /path/to/rakka-accel/ai-skills/skills/rakka-accel-device \
      .claude/skills/rakka-accel-device
```

### Cursor

Cursor reads project rules from `.cursor/rules/`. Copy the skills in
as `.mdc` rules; Cursor will treat the frontmatter `description` as
the activation hint:

```bash
mkdir -p .cursor/rules
for s in /path/to/rakka-accel/ai-skills/skills/*/SKILL.md; do
  name=$(basename "$(dirname "$s")")
  cp "$s" ".cursor/rules/${name}.mdc"
done
```

### OpenAI Codex CLI

Codex CLI reads `AGENTS.md` (project-level) and `~/.codex/AGENTS.md`
(user-level). It does not have a SKILL.md loader, so reference the
skills from `AGENTS.md` and let the model pull them in on demand:

```markdown
<!-- in AGENTS.md -->
## rakka-accel skills
When working on rakka-accel, consult the matching skill in
`ai-skills/skills/<name>/SKILL.md`:
- first-time wiring / Cargo.toml          → rakka-accel-getting-started
- DeviceActor / GpuRef / memcpy / Sgemm   → rakka-accel-device
- picking or extending a kernel actor     → rakka-accel-kernels
- supervision / context loss / generations → rakka-accel-supervision
- Python bindings / numpy / GIL           → rakka-accel-python
- portable vs vendor-specific API choice  → rakka-accel-backends
- feature flags / OOM / CI vs GPU         → rakka-accel-troubleshooting
```

### Gemini CLI

Gemini CLI reads `GEMINI.md` and supports custom commands under
`.gemini/commands/`. Point Gemini at the skills the same way:

```markdown
<!-- in GEMINI.md -->
For rakka-accel work, load the relevant skill from
`ai-skills/skills/<name>/SKILL.md` before editing.
```

Optionally wrap each skill as a slash command in
`.gemini/commands/rakka-accel-<name>.toml` whose `prompt` includes
`@ai-skills/skills/<name>/SKILL.md`.

### Other harnesses (Aider, Continue, Zed, etc.)

Any tool that accepts a system prompt or rules file can load these
skills — `SKILL.md` is plain Markdown with YAML frontmatter. Either
include the file directly in the system prompt, or reference its path
from your tool's rules file (`.aider.conf.yml`, `.continue/`, etc.).

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
