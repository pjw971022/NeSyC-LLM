# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks in Open Domains

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openreview.net/pdf?id=VoayJihXra)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-green)](https://openreview.net/forum?id=VoayJihXra)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/pjw971022/nesyc-LLM)


<p align="center">
  <img src="static/videos/ani_v2.gif" alt="NeSyC Framework" width="80%">
</p>


## Authors
[Jinwoo Park](https://pjw971022.github.io/)<sup>1*</sup>, [Wonje Choi](https://scholar.google.com/citations?user=L4d1CjEAAAAJ&hl=ko)<sup>1*</sup>, [Sanghyun Ahn](https://scholar.google.co.kr/citations?user=xGh7hdIAAAAJ&hl=ko)<sup>1</sup>, [Daehee Lee](https://www.linkedin.com/in/daehee-lee-10b396246/?locale=en_US)<sup>1,2</sup>, [Honguk Woo](https://scholar.google.co.kr/citations?user=Gaxjc7UAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>Sungkyunkwan University (SKKU), <sup>2</sup>Carnegie Mellon University (CMU)
<sup>*</sup>Equal contribution

## Abstract

We explore neuro-symbolic approaches to generalize actionable knowledge, enabling embodied agents to tackle complex tasks more effectively in open-domain environments. A key challenge for embodied agents is the generalization of knowledge across diverse environments and situations, as limited experiences often confine them to their prior knowledge.

To address this issue, we introduce a novel framework, **NeSyC**, a neuro-symbolic continual learner that emulates the hypothetico-deductive model by continuously formulating and validating knowledge from limited experiences through the combined use of Large Language Models (LLMs) and symbolic tools. Specifically, NeSyC incorporates a **contrastive generality improvement scheme**. This scheme iteratively produces hypotheses using LLMs and conducts contrastive validation with symbolic tools, reinforcing the justification for admissible actions while minimizing the inference of inadmissible ones.

We also introduce a **memory-based monitoring scheme** that efficiently detects action errors and triggers the knowledge evolution process across domains. Experiments conducted on embodied control benchmarks—including ALFWorld, VirtualHome, Minecraft, RLBench, and a real-world robotic scenario—demonstrate that NeSyC is highly effective in solving complex embodied tasks across a range of open-domain settings.

## Key Features

- Neuro-symbolic continual learning framework
- Contrastive generality improvement scheme
- Memory-based monitoring for error detection
- Effective knowledge evolution across domains

---

## Reproduction Guide (ALFWorld Experiment)

This section provides a complete, step-by-step guide to reproduce the ALFWorld experiment results. It also documents known issues in the original codebase and the fixes required.

### Prerequisites

- Python 3.11+ (tested with 3.12)
- pip
- A Google Gemini API key (obtain from [Google AI Studio](https://aistudio.google.com/))
  - Note: The original paper used `gemini-1.0-pro`, which has been **deprecated**. Use `gemini-2.0-flash` instead.
  - OpenAI GPT-4o is also supported if you have an API key.

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install alfworld==0.3.5
pip install 'langchain==0.2.16' 'langchain-community==0.2.12' 'langchain-core==0.2.38' \
            'langchain-openai==0.1.21' 'langchain-google-genai==1.0.8' \
            'langchain-text-splitters==0.2.2' 'langsmith==0.1.99'
pip install openai google-generativeai clingo replicate tabulate pyyaml sentence-transformers
```

### Step 2: Download ALFWorld Data

```bash
alfworld-download
```

This downloads game data to `~/.cache/alfworld/`.

### Step 3: Apply Required Code Fixes

The original repository has several issues that prevent out-of-the-box execution. The following fixes are required:

#### Fix 3.1: Path References (Critical)

All hardcoded paths reference `./Nesyc/Alfworld/` but the actual directory is `./Alfworld/`. Apply the following changes:

**`Alfworld/utils.py`** — Set `ALFWORLD_DATA_PATH` and fix config path:
```python
# Before
ALFWORLD_DATA_PATH = ""
CONFIG_FILE = './Nesyc/Alfworld/base_config.yaml'

# After
ALFWORLD_DATA_PATH = os.path.expanduser("~/.cache/alfworld")
CONFIG_FILE = './Alfworld/base_config.yaml'
```

Replace all `./Nesyc/Alfworld/` with `./Alfworld/` in these files:
- `Alfworld/main.py` (prompt paths, save paths, ILP rule paths)
- `Alfworld/env_utils.py` (paraphrased instruction paths, error log paths)
- `Alfworld/our_pipeline.py` (demo data paths)
- `Alfworld/base_config.yaml` (invalid_game_file path)
- `Alfworld/utils.py` (demo data path in RetrievalEngine)

#### Fix 3.2: Missing `__init__.py` Files (Critical)

The code uses `from Alfworld.xxx import ...` imports, which require package `__init__.py` files:

```bash
touch Alfworld/__init__.py
touch Alfworld/data/__init__.py
touch Alfworld/data/base/__init__.py
```

#### Fix 3.3: Missing `oracle_custom.py` Module (Critical)

The file `Alfworld/alfworld/agents/controller/oracle_custom.py` is referenced but not included in the repository. Since the ALFWorld TextWorld environment (`NusawAlfredTWEnv`) does not use this controller, remove the import:

**`Alfworld/alfworld/agents/controller/__init__.py`** — Remove line:
```python
from alfworld.agents.controller.oracle_custom import CustomOracleAgent
```

#### Fix 3.4: Missing CLI Arguments (Critical)

The shell scripts use `--asp` and `--ilp` flags that are not defined in `argparse`:

**`Alfworld/main.py`** — Add before `args = parser.parse_args()`:
```python
parser.add_argument('--asp', action='store_true', help='use ASP solver')
parser.add_argument('--ilp', action='store_true', help='use ILP rules')
```

#### Fix 3.5: Indentation Bug in `env_utils.py` (Critical)

`raise ValueError` is inside the `if method == 'ours'` block instead of an `else` clause:

**`Alfworld/env_utils.py`** — Fix the `run_episode` function (~line 374):
```python
# Before (broken — always raises error for 'ours')
        if method == 'ours':
            ...
            raise ValueError(f"Invalid method: {method}")

# After (correct — only raises for unknown methods)
        if method == 'ours':
            ...
        else:
            raise ValueError(f"Invalid method: {method}")
```

#### Fix 3.6: Create Error Log Directory

```bash
mkdir -p Alfworld/data/error_log
```

### Step 4: Set API Key

Set your API key as an environment variable:

```bash
# For Google Gemini
export GOOGLE_API_KEY='your-google-api-key-here'

# For OpenAI (if using GPT-4o)
export OPENAI_API_KEY='your-openai-api-key-here'
```

Alternatively, edit `Alfworld/llm_utils.py` directly and set the key values.

### Step 5: Run the Experiment

The experiment has two stages that must be run sequentially from the repository root:

```bash
export PYTHONPATH=$(pwd):$(pwd)/Alfworld
```

#### Stage 1: General — Generate ILP Rules

```bash
python -m Alfworld.main \
    --procedure general \
    --method ours \
    --engine gemini-2.0-flash \
    --seed 42
```

This generates ASP rules from demonstration trajectories using the LLM. Rules are saved to `./Alfworld/data/ilp_rule_gemini-2.0-flash.txt`.

**Important: You must manually clean the generated rules file** (see [Known Issue 1](#known-issue-1-malformed-asp-rules-from-llm-generation) below).

#### Stage 2: Adapt — Evaluate on ALFWorld

```bash
python -m Alfworld.main \
    --method ours \
    --engine gemini-2.0-flash \
    --dynamics stationary \
    --eval_episode_num 20 \
    --seed 42 \
    --asp \
    --ilp
```

Results are saved to `./Alfworld/ablation2_alfworld_results.csv`.

### Example Results (3 episodes, stationary dynamics)

```
+--------+------------------+----------+----------+---------------+
| Method |      Engine      | Total SR | Total GC | Total PLANACC |
+--------+------------------+----------+----------+---------------+
|  ours  | gemini-2.0-flash |   1.0    |   1.0    |      1.0      |
+--------+------------------+----------+----------+---------------+
```

---

## Known Issues and Analysis

### Known Issue 1: Malformed ASP Rules from LLM Generation

**Status: Confirmed — Present in original code**

**Symptom:** The `general` stage produces ASP rules with syntax errors (incomplete lines, missing `.` terminators, truncated predicates).

**Root Causes:**
1. **`max_tokens=64` in Pipeline** (`our_pipeline.py:86`) — This default is far too low for ASP rule generation. A single rule can consume 20-25 tokens, so multi-rule responses are routinely truncated.
2. **No syntax validation** — `extract_rule()` strips markdown formatting but does not validate ASP syntax (balanced parentheses, `.` terminators).
3. **Aggressive line-based filtering** — `rule_filtering()` splits by newline and filters by `:-` presence, which breaks multiline rules and discards all positive/effect rules.
4. **No clingo validation before save** — Malformed rules are written to disk without testing if they parse.

**Example of malformed output:**
```
:- action(pick_up(O,L),T), is_opened(L        ← truncated, no closing )  or .
:- action(put_down(                            ← severely truncated
:- action(pick_up(O, L), T), not at(O, L, T   ← missing closing ).
```

**Workaround:** After running `--procedure general`, manually edit the generated rules file (`Alfworld/data/ilp_rule_<engine>.txt`):
- Remove lines that don't end with `.`
- Remove duplicate rules
- Ensure all parentheses are balanced

**Example of a valid rules file:**
```prolog
:- action(pick_up(O,L),T), not at(O,L,T).
:- action(pick_up(O,L),T), not robot_at(L,T).
:- action(pick_up(O, L), T), holding(O, T).
:- action(put_down(O, L), T), not holding(O, T).
:- action(put_down(O, L), T), not robot_at(L, T).
:- action(heat(O, L), T), not is_heater(L).
:- action(heat(O, L), T), not holding(O, T).
:- action(heat(O, L), T), not robot_at(L, T).
:- action(cool(O, L), T), not holding(O, T).
:- action(cool(O, L), T), not robot_at(L, T).
:- action(cool(O, L), T), not is_cooler(L).
:- action(cool(O, L), T), not coolable(O).
:- action(clean(O, L), T), not cleanable(O).
:- action(clean(O, L), T), not is_cleaner(L).
:- action(clean(O, L), T), not robot_at(L, T).
:- action(clean(O, L), T), not holding(O, T).
:- action(open(L), T), not robot_at(L, T), openable(L).
```

### Known Issue 2: Adaptation Prompt Missing Original Rules

**Status: Confirmed — Present in original code**

**Symptom:** The `adapt_rule` stage prompt references "Original Program" but never actually includes the generalized rules from Stage 1. This causes the LLM to generate rules without context, leading to SR/GC/Plan Accuracy of 0.00 in some episodes.

**Root Cause:** In `our_pipeline.py`, `gen_adapt_rule_response()` only appends the environment trajectory to the prompt:
```python
prompt = f"{self.prompt[kind]}\n\nEnvironment Trajectory:\n{observations}"
```
The `self.adapted_rules` (loaded from the general stage) are never injected into the prompt, despite the template instructing the LLM: *"Carefully read the 'Observation' and 'Original Program'"*.

**Impact:** The adaptation stage cannot refine existing rules since the LLM never sees them. It generates rules from scratch, often producing ASP-incompatible output.

### Known Issue 3: No-op (Do Nothing) Actions

**Status: Confirmed — Present in original code**

**Symptom:** The robot outputs `do nothing` repeatedly, resulting in `Nothing happens.` observations for entire episodes.

**Root Cause chain:**
1. ASP solver receives malformed rules (Issue 1) or lacks necessary predicates
2. Solver returns empty answer sets (`[]`)
3. `parse_predicted_plan([])` returns empty plan `[]`
4. `filter_actions_by_step([], step_num)` finds no action for the current step
5. Returns `'do nothing'` as fallback (see `env_utils.py:177`)

**Particularly affected scenarios:**
- Episodes with goal predicates using custom types (e.g., `is_spray_bottle`, `is_cup`) that are not defined in the base ASP program
- Episodes requiring rules that were truncated during generation

### Known Issue 4: Deprecated Gemini Models

**Status: Resolved by using newer model**

The original code defaults to `gemini-1.0-pro`, which has been discontinued. Use `gemini-2.0-flash` (or `gemini-2.0-flash-lite`) instead:
```bash
--engine gemini-2.0-flash
```

### Known Issue 5: Missing `oracle_custom.py`

**Status: Resolved by removing import**

The file `Alfworld/alfworld/agents/controller/oracle_custom.py` is referenced but missing from the repository. This module is not needed for the TextWorld-based evaluation (`NusawAlfredTWEnv`), so the import can be safely removed.

---

## Summary of All Required Fixes

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | Multiple files | `./Nesyc/Alfworld/` hardcoded paths | Replace with `./Alfworld/` |
| 2 | `utils.py` | `ALFWORLD_DATA_PATH` empty | Set to `~/.cache/alfworld` |
| 3 | `Alfworld/`, `data/`, `data/base/` | Missing `__init__.py` | Create empty files |
| 4 | `alfworld/agents/controller/__init__.py` | Missing `oracle_custom.py` | Remove import line |
| 5 | `main.py` | `--asp` `--ilp` flags undefined | Add to argparse |
| 6 | `env_utils.py` | `raise ValueError` wrong indentation | Move to `else` clause |
| 7 | — | `data/error_log/` directory missing | `mkdir -p` |
| 8 | `llm_utils.py` | API keys empty | Set your keys |
| 9 | — | `gemini-1.0-pro` deprecated | Use `gemini-2.0-flash` |
| 10 | Generated rules file | Malformed ASP syntax | Manual cleanup required |

---

## Project Structure

```
NeSyC-LLM/
├── README.md
├── environment.yml              # Conda environment specification
├── Alfworld/
│   ├── __init__.py              # (must be created)
│   ├── main.py                  # Entry point
│   ├── our_pipeline.py          # NeSyC pipeline (ASP + LLM)
│   ├── env_utils.py             # ALFWorld environment utilities
│   ├── utils.py                 # Helper functions
│   ├── llm_utils.py             # LLM API wrappers (Gemini, OpenAI, Meta)
│   ├── base_config.yaml         # ALFWorld configuration
│   ├── alfworld/                # Modified ALFWorld package (custom NusawAlfredTWEnv)
│   ├── data/
│   │   ├── base/base_program.py # Base ASP programs (ORACLE + BASE)
│   │   ├── demo/                # Demonstration trajectories for each action type
│   │   ├── instr/               # Paraphrased instructions
│   │   └── error_log/           # (must be created) Runtime error logs
│   ├── ours_prompts/
│   │   ├── general/             # Prompts for Stage 1 (rule generalization)
│   │   │   ├── fact.txt
│   │   │   ├── ILP_bk.txt
│   │   │   └── ILP_rule.txt
│   │   └── adapt/               # Prompts for Stage 2 (rule adaptation)
│   │       ├── fact.txt
│   │       ├── rule.txt
│   │       └── goal_state_ours.txt
│   └── script/                  # Experiment shell scripts
│       ├── static.sh
│       ├── low_dynamics.sh
│       └── high_dynamics.sh
└── static/                      # Paper assets
```

---

## Citation

```bibtex
@inproceedings{choinesyc,
  title={NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains},
  author={Choi, Wonje and Park, Jinwoo and Ahn, Sanghyun and Lee, Daehee and Woo, Honguk},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
@article{choi2025nesyc,
  title={NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains},
  author={Choi, Wonje and Park, Jinwoo and Ahn, Sanghyun and Lee, Daehee and Woo, Honguk},
  journal={arXiv preprint arXiv:2503.00870},
  year={2025}
}
```
