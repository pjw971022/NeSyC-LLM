# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks in Open Domains

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openreview.net/pdf?id=VoayJihXra)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-green)](https://openreview.net/forum?id=VoayJihXra)
[![arXiv](https://img.shields.io/badge/arXiv-2503.00870-blue)](https://arxiv.org/abs/2503.00870)

<p align="center"><img src="static/videos/ani_v2.gif" alt="NeSyC Framework" width="100%"></p>

[Jinwoo Park](https://pjw971022.github.io/)\*, [Wonje Choi](https://scholar.google.com/citations?user=L4d1CjEAAAAJ&hl=ko)\*, [Sanghyun Ahn](https://scholar.google.co.kr/citations?user=xGh7hdIAAAAJ&hl=ko), [Daehee Lee](https://www.linkedin.com/in/daehee-lee-10b396246/?locale=en_US), [Honguk Woo](https://scholar.google.co.kr/citations?user=Gaxjc7UAAAAJ&hl=en) — SKKU, CMU (\*equal contribution)

NeSyC learns action rules (ASP integrity constraints) from a few demonstrations, plans with them, and re-learns them
whenever an action fails. This repository is the ALFWorld implementation.

> **Status.** A re-implementation of Algorithms 1–2 that runs end to end; the code behind the paper's Table 1 no
> longer exists. It was not run with `gpt-4o-2024-08-06` and does not reproduce the paper's numbers (NeSyC 82.9 %,
> ReAct 35.8 %, Reflexion 39.0 %). Every number below replays without an API key: `bash Alfworld/script/replay.sh`.

## Setup

```bash
pip install -r requirements.txt   # Python 3.11, clingo 5.7.1
alfworld-download                 # game files -> ~/.cache/alfworld
export OPENAI_API_KEY=...         # environment only; .env is git-ignored
export OPENAI_BASE_URL=...        # any OpenAI-compatible endpoint, e.g. Gemini, OpenRouter, Groq
```

Answers are cached (`LLM_CACHE`, default `llm_cache.json`), so an interrupted run resumes. Use a non-reasoning model
or raise `--max_tokens` (256). Gemini's free tier needs a project with no billing attached; a prepaid project without
credit answers `402`.

## Run

```bash
ENGINE=<model id> bash Alfworld/script/static.sh        # learn rules, then the 123 valid_seen games
ENGINE=<model id> bash Alfworld/script/hard_stream.sh   # hidden-rule kitchens (also hard.sh, hard_suite.sh)
pytest                                                  # 27 LLM-free checks
```

| Option | Effect |
|---|---|
| `--method ours \| react \| reflexion \| react_memory` | NeSyC (default) or a baseline |
| `--eval_episode_num N` | the first N games of a fixed, seeded order |
| `--hard FILE` | the tasks of FILE instead of the benchmark games (`--split train` for validation 4–5) |
| `--continual` | keep learned rules from one episode to the next (set by the hard-task scripts) |
| `--revise`, `--hints`, `--plus`, `--v2` | [extensions](#extensions-not-in-the-paper); off by default |

## Results

The LLM was **Claude, through Claude Code sub-agents** answering the recorded prompts (no API key was available), so
temperature and the token cap were not enforced. The 72 runs below ship as transcripts (`Alfworld/data/llm_cache_*`)
and `replay.sh` checks every success rate. Baselines exist only for Fable and Haiku: with Opus and Sonnet the
sub-agents answering ReAct prompts were stopped by the models' safeguards.

| | |
|---|---|
| NeSyC ahead of ReAct / Reflexion with the same LLM | **once**: Haiku, hard suite, `--v2` 3/6 against 0/6 |
| LLM calls | NeSyC needs **7–30× fewer** |
| Paper's version on hidden-rule kitchens | **0**, with Opus, Sonnet and Haiku |
| 120 held-out kitchens (validation 4–5) | paper **0**, `--plus` **38**, `--v2` **69** |

**Against the baselines** (episodes solved):

| | Fable: 20 games | Fable: hard stream (11) | Haiku: 20 games | Haiku: hard suite (6) |
|---|---|---|---|---|
| NeSyC | 20 | 4 | 19 | 0 |
| NeSyC + `--v2` | | | | **3** |
| ReAct | 20 | 0 | 20 | 0 |
| ReAct + notes | | 10 | | |
| Reflexion, 3 trials | 20 | 11 (0 on the first) | 20 | 0 |

With Fable, NeSyC also solves 122 of the 123 games, and all three methods solve the single hard task. LLM calls per
game on the stream: 2.2 for NeSyC, 23–62 for the baselines.

**Held-out hidden-rule streams** (NeSyC only, kitchens solved of 10, one run per cell):

| | Val. 3 | Val. 4 | Val. 5 · A | Val. 5 · B | Val. 5 · C | Val. 4–5 (40) |
|---|---|---|---|---|---|---|
| Opus: NeSyC / `--plus` / `--v2` | 0 / 7 / – | 0 / 9 / 7 | 0 / 5 / 9 | 0 / 5 / 0 | 0 / 0 / 9 | 0 / 19 / **25** |
| Sonnet: NeSyC / `--plus` / `--v2` | 0 / 0 / – | 0 / 0 / 8 | 0 / 0 / 0 | 0 / 0 / 9 | 0 / 0 / 9 | 0 / 0 / **26** |
| Haiku: NeSyC / `--plus` / `--v2` | 0 / 0 / – | 0 / 9 / 0 | 0 / 0 / 0 | 0 / 0 / 8 | 0 / 10 / 10 | 0 / 19 / **18** |

- `--v2` is ahead of `--plus` in 6 of 12 cells, behind in 3, tied in 3. Cells are mostly all or nothing: a version
  that learns the hidden rules in the first kitchens solves nearly every later one; when it fails, the LLM blamed its
  first failures on the wrong condition.
- In the traced runs the paper's version learned a wrong rule from an ambiguous failure, never retracted it and was
  left without a plan.
- The extensions were developed with Opus on the dev stream and validation 1–2 (`--revise` 2/11, `--v2` 5/11, all
  others 0) and judged on later streams only.
- Caveats: 6–30 games per cell, one run each. The baselines are favoured: ReAct's thoughts cost no step, Reflexion
  counts any of 3 trials, and they see the object locations NeSyC plans with.

## Hard tasks

*Rinse a mug, chill it and leave it in the coffee machine. Also microwave a potato and store it, still hot, in the
fridge. When you are done, the fridge and the microwave must both be shut.* About 20 actions (benchmark games need
3–9), 30 allowed. Preconditions that no demonstration shows make actions fail silently until they are learned.

| Task | Kitchens | Hidden rules |
|---|---|---|
| `hard_task.json` | 1 | doors open to heat or chill |
| `hard_suite.json` | 6 | doors open; only washed objects can be heated or chilled |
| `hard_stream.json` (dev), `hard_stream_test{,2,3}.json` | 11, 11 + 11, 10 | A |
| `hard_stream_test4.json`, `hard_stream_test5{a,b,c}.json` (training split) | 10, 3 × 10 | A; A, B, C |

| Hidden rule (streams; the potato must also be washed) | A | B | C |
|---|---|---|---|
| heating needs the door open | ✓ | ✓ | |
| chilling needs the door open | ✓ | ✓ | ✓ |
| only a washed object can be heated / chilled | – / ✓ | ✓ / – | ✓ / ✓ |
| a washed object can no longer be heated / chilled | ✓ / – | – / ✓ | – / – |
| one door open at a time | ✓ | | ✓ |

Task files were fixed before any model ran on them. Kitchens have all six objects and an unrelated original task,
are drawn in a seeded order, and are kept only if the planner solves them given the right rules (for validation 5,
both the `--plus` and the `--v2` planner).

## What the code does

| Paper | Code |
|---|---|
| Algorithm 1, rule reformulation | `Pipeline.reformulate` (`Alfworld/our_pipeline.py`): the LLM proposes constraints, clingo scores them with HI (Eq. 9) and returns feedback, `ITER_MAX = 3` rounds |
| Algorithm 2, rule application | `run_episode` (`Alfworld/env_utils.py`): plan with clingo, act, remember; a failed action re-enters Algorithm 1 |
| Baselines (Appendix C) | `Alfworld/baselines.py`: ReAct, Reflexion (a plan after each failed trial), ReAct + notes (a note after every episode); top-3 demonstrations instead of task-specific few-shot examples |

Open choices: α = 0.5, λ = 0.3; one constraint per hypothesis, kept greedily for HI; a rule that forbids a
demonstrated action needs a failure from the current environment; rules reset per episode unless `--continual`.

## Extensions (not in the paper)

- `--revise`: when no plan exists, learned rules that no failure of the episode supports become weak constraints; a
  learned rule that fired on an action which then worked is deleted.
- `--hints`: three general guidelines in the rule prompt (`ours_prompts/general/rule_hints.txt`).
- `--plus`: `--hints --revise`; type predicates in the rule prompt; run-time experience weighted as current evidence;
  ALFWorld's effects the base program leaves open (chilling removes heat, closing closes, doors start closed).
- `--v2`: `--plus`; going somewhere puts the robot there (the base program let it stay put, so failures were stored
  with the wrong location); a failed action is not planned again before the believed state changes; with still no
  plan, every learned rule becomes a weak constraint.

`--plus` and `--v2` are opt-in so that earlier runs still replay.

## Changes since `bad2329`

- Crashes fixed: `UnboundLocalError` on the first failed action, an `ipdb` breakpoint, a missing `error_log/`.
- Proposed rules are checked with clingo and scored before they are kept; prompts, demonstrations and base program
  share one vocabulary; demonstrations are re-grounded at receptacle level (the expert rules accept 17/17).
- Planner: objects with an unknown location were assumed to be at the goal, nested receptacles made the program
  unsatisfiable, the goal prompt kept the first episode's predicates; all fixed.
- LLM: any OpenAI-compatible endpoint (`openai>=1.55.3`), key from the environment, on-disk cache.
- Added: baselines, hard tasks, extensions, tests, `script/replay.sh`.

## Citation

```bibtex
@inproceedings{choinesyc,
  title={NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains},
  author={Choi, Wonje and Park, Jinwoo and Ahn, Sanghyun and Lee, Daehee and Woo, Honguk},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
