# MNN Project Instructions

MNN is a lightweight deep learning **inference engine** (not a training framework), targeting mobile and server platforms. Supports CNN / Transformer / LLM / Diffusion models. Code must prioritize **performance and binary size**.

---

## Autonomous Agent Operational Directive

> Extracted from cam_app/AGENTS.md — general rules applicable to all projects.

### 1. Absolute Ground Rules & Constraints

* **INSPECT BEFORE EDITING**: You are strictly forbidden from editing, patching, or overwriting a file without viewing its contents first. You must always invoke the `bash` or file-viewing tool to read the target file completely. Never assume file structures, layouts, or line counts.

* **NO CLUTTER / NO `/tmp`**: Execute all operations and structural code modifications directly on the requested production file paths. Do not use, create, or reference `/tmp`, scratch directories, or temporary staging files unless explicitly mandated by the environment.

* **EXACT LAYOUT PRESERVATION**: You must match the existing file formatting flawlessly. Pay meticulous attention to:
  * Indentation type (spaces vs. tabs) and exact indentation counts.
  * Trailing spaces and whitespace hygiene (do not leave dangling whitespaces).
  * Bracket placement, trailing commas, and file-ending newlines.

### 2. Anti-Avoidance & Zero-Passivity Manifesto

* **NO WORKAROUNDS**: You are strictly forbidden from hiding, bypassing, or avoiding code problems.
* **NO MOCKING AWAY ERRORS**: Do not delete failing tests, do not write dummy try/catch blocks that silently swallow exceptions, and do not comment out problematic code blocks or mock out functions just to force a compilation step to pass. You must fix the root flaw.
* **FIRST-PRINCIPLES DIAGNOSTICS**: When an error or unmet criterion occurs, you are legally forbidden from modifying any source file until you have executed tools to trace the failure. You must actively investigate **WHY** the failure happens. Use `bash` to run verbose logging, inspect stack traces line-by-line, print intermediate variable states, and map out exactly where the runtime state diverges from expectations.

### 3. Small Changesets & Git Hygiene

* **ATOMIC COMMITS**: Break large engineering tasks into small, logical, and incremental modifications. Do not bundle multiple unrelated features or fixes into a single massive update.

* **STAGE AND COMMIT PROACTIVELY**: Once a small, isolated module or function is updated and successfully verified, staging and committing those changes immediately using Git before moving to the next code block is highly encouraged.

* **DESCRIPTIVE MESSAGES**: Write concise, meaningful commit messages that explicitly state what structural change was introduced.

* **FAIL-SAFE ROLLBACK**: If an attempted fix creates catastrophic regressions or structural confusion across more than 3 modules, you must execute `git checkout -- .` or `git reset` to revert to your last verified working changeset and formulate an entirely new architectural approach.

### 4. Mandatory Cognitive & Verification Loop

You must process every single engineering task through this strict, non-negotiable loop. A simple compilation or test passing message is only the starting baseline; you are forbidden from stopping until you have thoroughly analyzed the execution logs for hidden optimizations.

#### Phase 0: Status Declaration (MANDATORY)
Before EVERY tool call, you must output 1-2 short sentences declaring:
1. **Current status** — what was just completed or what you are about to do.
2. **Next step** — what you will do next and WHY.

#### Phase 1: Proactive Architecture Mapping
Read the target file and its surrounding modules. Map the dependencies and analyze the blast radius of your changes before typing code.

#### Phase 2: Root Cause Diagnosis (The Anti-Avoidance Layer)
If you are resolving a bug or fixing a quality degradation, do not guess. Use your tools to isolate the exact line, system state, or edge-case input triggering the failure. You must explicitly isolate and cite the exact file name and line number from the error log inside your inner monologue. State clearly:
* *What is the exact symptom?*
* *What is the proven root cause?*
* *What is the clean, non-hacky architectural fix?*

#### Phase 3: Clean Direct Modification
Apply your structural code or configuration updates directly to the production file path using native tools in small, manageable chunks based strictly on your Phase 2 diagnosis.

#### Phase 4: Environmental Verification
Instantly after saving modifications, invoke the `bash` tool to run the build, compilation, test, linting, or validation workflows. You must capture and parse the *entire* output payload of this run.

#### Phase 5: Critical Criteria Check & Evaluation
Parse the `stdout` and `stderr` logs meticulously. You must evaluate the run against two parallel standards:
* **Functional Standard**: Did the code compile and did the primary test suite return `exit 0`?
* **Quality & Performance Standard**: Are there any lingering warnings, deprecation notices, slow execution bottlenecks, type-checking flaws, or architectural shortcuts?

**DYNAMIC RETARGETING RULE**: If the functional standard is met (`exit 0`) but any secondary quality criteria are broken, **the task is not done.** You must immediately treat these unmet criteria as critical sub-task failures. Proactively isolate the root cause of the warning, bottleneck, or structural defect, engineer a robust solution to fix it, and apply it directly to the code. Loop back to Phase 4 and execute verification again. You must repeat this cycle until *both* functional and quality criteria are flawlessly satisfied.

**LOOP-BREAKING GUARDRAIL**: If your verification loop returns the exact same terminal error message or exit code two times in a row, you are hit with an internal logic lock. You must immediately stop your current approach, declare your previous assumption invalid in your monologue, change your debugging strategy completely, and try an alternative engineering pattern.

#### Phase 6: Mandatory Post-Success Compliance Check
After achieving a clean, un-warned `exit 0` execution run, double-check your code against the project's broader design intent. Ensure that your implementation did not create performance regressions, type-checking faults, or silent runtime errors in surrounding, imported modules.

#### Phase 7: Proactive Exhaustive Follow-Up
Conclude your execution run only when the test run output logs are completely clean, optimized, and free of architectural shortcuts. If no further micro-optimizations or cleanups can logically be made to make the codebase better, commit your final clean changeset before spinning down.

### 5. Session
**Created**: 2026-07-13

### 6. Active Extensions
- **pi-replace-tool**: Enhanced replace with content dump on no-match
- **pi-multi-subs**: Interactive subscription manager (/subs)
- **pi-multi-pass**: Interactive route manager (/route)
- **pi-session-id**: Session tracking and Mistral role fixes

### 7. Tool & Notification Rules
- Use `ctx.ui.notify(message, level)` for all inline output (level: "info" | "warning" | "error")
- Use Node.js `fs/promises` for all file operations
- Provisioned providers selectable via `ctx.ui.select()`
- Cloned provider names auto-generated as `-N` suffix
- Session ID injected into system prompts for Mistral compatibility

### 8. Route & Failover
- Route auto-switches on HTTP 429 (rate limit) via `after_provider_response`
- Fallback via `agent_end` with `isRateLimitError` pattern matching
- Provider cooldown: 30s window per provider (prevents rapid repeat)
- Route wraps around to first hop after exhausting all hops
- **Wildcard model** (`""`): preserves current model ID across provider switches
- Notification on switch: `Switched: oldProvider/oldModel → newProvider/newModel`

### 9. Mistral Compatibility
- `requiresAssistantAfterToolResult: true` set per-model in `~/.pi/agent/models.json`
- `before_provider_request` handler inserts `assistant` between `tool → user` for Mistral models
- Detects Mistral models by ID keyword (`mistral-*` in model name)
- Works for all Mistral variants (Nvidia, OpenRouter, HuggingFace, Together, etc.)

### 10. Native Tool Execution & Structural Tagging Rules
* **STRUCTURAL STATE TAGS**: To maintain cognitive alignment across long horizons, you must structure your outputs into explicit markdown XML blocks:
  * Place all planning, error reflections, and architectural mapping inside `<thought>` blocks.
  * Place your native function and tool call execution choices inside `<action>` blocks.

* **DO NOT WAIT FOR THE USER**: You are an entirely unattended pipeline. Do not write text prompts asking the user "Should I proceed?", "Is this correct?", or "What should I do next?". Chain your tool blocks continuously and autonomously execute until the entire scope of work and its follow-ups are closed out.

* **NO RAW CODE IN CHAT BLOCK**: Do not dump raw source code, unified diffs, or markdown code snippets (e.g., ```python, ```text) into standard conversational response blocks. All source modifications must happen natively inside tool payloads.

* **EXECUTION MANDATE**: Every single engineering action must happen exclusively through native LLM tool calls. Text-based simulations break the parsing framework.

* **DIALOGUE SUPPRESSION**: Completely eliminate filler text, pleasantries, explanations, and standard chat. Focus your generation entirely on your inner monologue, planning states, and immediate native tool execution.

## Restricted Access

> The following directories contain internal proprietary code. **Do NOT read, modify, or reference** any files within:
> - `schema/private/`
> - `source/internal/`

## Architecture Overview

MNN uses a **graph optimization + heterogeneous backend scheduling** architecture.

Two inference APIs are available:
- **Session API** (low-level): `Interpreter → createSession → runSession`, operates on Tensor directly
- **Module API** (high-level, recommended): `Module::load → onForward(VARP)`, Express-based dynamic graph. Used by LLM / Diffusion and most modern workloads

**Key abstractions** (see corresponding headers under `source/core/`):
- **Interpreter** / **Session**: model loading and inference session management
- **Backend** / **Execution**: hardware backend abstraction and per-op implementation (CPU/Metal/CUDA/OpenCL/Vulkan/...)
- **Tensor**: data container; internally uses NC4HW4 format (channels packed by 4 for SIMD)
- **Op / Schema**: FlatBuffers-defined operator descriptors (`schema/default/*.fbs`)

**Op registration pattern**: Schema definition → shape inference (`source/shape/`) → Geometry decomposition (optional) → Backend Execution implementation

## LLM Subsystem

MNN supports end-to-end LLM export and inference:

- **Python export** (`transformers/llm/export/`): HuggingFace model → MNN format. Core modules: `llmexport.py` entry point, `utils/model_mapper.py` (model field mapping), `utils/model.py` (unified LlmModel class), `utils/transformers.py` (Attention/Decoder/RoPE export)
- **C++ inference** (`transformers/llm/engine/`): `llm.cpp` (text inference), `omni.cpp` (multimodal: vision/audio), includes KVCache management and sampling strategies

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `include/MNN/` | Public C++ headers |
| `source/core/` | Inference core (Interpreter, Session, Pipeline, Backend) |
| `source/backend/` | Hardware backend implementations (cpu, arm82, metal, cuda, opencl, vulkan, ...) |
| `source/shape/` | Shape inference |
| `source/geometry/` | Geometry computation (op decomposition) |
| `express/` | Express API (high-level dynamic graph, VARP) |
| `schema/default/` | FlatBuffers schema (op definitions) |
| `tools/converter/` | Model converter (ONNX/TF/Caffe → MNN) |
| `transformers/llm/` | LLM export (Python) + inference engine (C++) |
| `transformers/diffusion/` | Diffusion model support |
| `pymnn/` | Python bindings |
| `test/` | Test cases |
| `skills/` | AI Agent Skills |

## Coding Style

- **C++**: Google Style variant, see `.clang-format`. 4-space indent, 120-char line width, attached braces. Class names `PascalCase`, functions `camelCase`, member variables `mCamelCase`. RTTI and exceptions disabled (`-fno-rtti -fno-exceptions`). Default standard: C++11.
- **Python**: Standard Python conventions
- **Formatting**: `clang-format -i -style=file <file>`

## Build & Test

```bash
# Build C++ (with LLM)
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)

# Common CMake options: MNN_BUILD_TEST, MNN_BUILD_CONVERTER, MNN_METAL, MNN_OPENCL,
# MNN_VULKAN, MNN_CUDA, MNN_ARM82, MNN_BUILD_QUANTOOLS, MNN_SUPPORT_TRANSFORMER_FUSE
# Full list: see option() declarations at the top of CMakeLists.txt

# Unit tests
cd build && ./run_test.out

# LLM export
cd transformers/llm/export
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path ./MODEL

# LLM test
cd build
./llm_demo /path/to/MODEL/config.json prompt.txt

# LLM benchmark
./llm_bench -m /path/to/MODEL/config.json
```

Test suite includes: unit tests (`run_test.out`), model tests, conversion tests (ONNX/TF/TFLite/Torch), quantization tests, LLM tests, PyMNN tests. See `test.sh`, `test_stages.json`, and `test/` directory for details.

## Development Workflow

- **Branch**: Use `feature/<short-description>`, for example `feature/llm-streaming`.
- **Commit**: Use a concise, one-line English message: `[Module:Type] Summary`, for example `[LLM:Feature] Add streaming support`.
  - Modules: `LLM`, `CPU`, `Metal`, `CUDA`, `OpenCL`, `Vulkan`, `Core`, `Infra`, `Doc`, etc.
  - Types: `Feature`, `Bugfix`, `Perf`, `Refact`, `Style`, `Doc`, `Test`, `Chore`, `Release`.

## Skills

For the following tasks, **read the Skill entry file first** and execute step by step. Each step must pass its tests before proceeding.

**After non-trivial skill-driven tasks, run Retrospective only when there are reusable lessons.**

Public skills are listed below. Environment-dependent skills may exist under `skills/*/SKILL.md`.

| Skill | Entry File | Trigger |
|-------|-----------|---------|
| Support new LLM | `skills/support-new-llm/SKILL.md` | Add / adapt a new LLM model |
| Add new op | `skills/add-new-op/SKILL.md` | Add a new operator |
| ARM CPU optimization | `skills/arm-cpu-optimize/SKILL.md` | Optimize op performance on ARM CPU |
| OpenCL optimization | `skills/opencl-optimize/SKILL.md` | Optimize op performance on OpenCL |
| Vulkan optimization | `skills/vulkan-optimize/SKILL.md` | Optimize op performance on Vulkan |
| Metal optimization | `skills/metal-optimize/SKILL.md` | Optimize op performance on Metal |
| Bugfix / debugging | `skills/general-debug/SKILL.md` | Diagnose correctness bugs / regressions in MNN — organized by bug category. |
| Run tests / CI | `skills/test-ci/SKILL.md` | Run the regression / CI suite (host or on-device), or add / select / retune a test stage |
| Retrospective | `skills/retrospective/SKILL.md` | After non-trivial tasks with reusable lessons |