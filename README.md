# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks in Open Domains

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openreview.net/pdf?id=VoayJihXra)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-green)](https://openreview.net/forum?id=VoayJihXra)
[![arXiv](https://img.shields.io/badge/arXiv-2503.00870-blue)](https://arxiv.org/abs/2503.00870)

<p align="center"><img src="static/videos/ani_v2.gif" alt="NeSyC Framework" width="100%"></p>

[Jinwoo Park](https://pjw971022.github.io/)\*, [Wonje Choi](https://scholar.google.com/citations?user=L4d1CjEAAAAJ&hl=ko)\*, [Sanghyun Ahn](https://scholar.google.co.kr/citations?user=xGh7hdIAAAAJ&hl=ko), [Daehee Lee](https://www.linkedin.com/in/daehee-lee-10b396246/?locale=en_US), [Honguk Woo](https://scholar.google.co.kr/citations?user=Gaxjc7UAAAAJ&hl=en) — SKKU, CMU (\*equal contribution)

NeSyC learns action rules (ASP integrity constraints) from a few demonstrations, plans with them, and re-learns them
whenever an action fails. This repository is the ALFWorld implementation.

> **Status.** The code and data behind the paper's Table 1 no longer exist. This is a re-implementation of
> Algorithms 1–2 that runs end to end. It has **not** been run with `gpt-4o-2024-08-06` and does not reproduce the
> paper's numbers (NeSyC 82.9 %, ReAct 35.8 %, Reflexion 39.0 %). What was run is under [Results](#results), and
> every number there replays without an API key: `bash Alfworld/script/replay.sh`.

## Setup

```bash
pip install -r requirements.txt   # Python 3.11, clingo 5.7.1
alfworld-download                 # game files -> ~/.cache/alfworld
export OPENAI_API_KEY=...         # read from the environment only; never commit a key (.env is git-ignored)
export OPENAI_BASE_URL=...        # only for a non-OpenAI endpoint
```

The LLM is one chat call to any OpenAI-compatible cloud endpoint (`Alfworld/llm_utils.py`), so a free tier works:

| Provider | `OPENAI_BASE_URL` |
|---|---|
| Google AI Studio (Gemini) | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| OpenRouter (`...:free` models) | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |

Free tiers are rate-limited. Every answer is cached (`LLM_CACHE`, default `llm_cache.json`; `.gz` works too), so an
interrupted run resumes where it stopped. Use a non-reasoning chat model, or raise `--max_tokens` (default 256).

## Run

```bash
ENGINE=<model id> bash Alfworld/script/static.sh         # learn rules, then 123 valid_seen games
ENGINE=<model id> bash Alfworld/script/low_dynamics.sh   # / high_dynamics.sh
ENGINE=<model id> bash Alfworld/script/hard.sh           # one hard task
ENGINE=<model id> bash Alfworld/script/hard_suite.sh     # six hard tasks in a row
ENGINE=<model id> bash Alfworld/script/hard_stream.sh    # eleven kitchens with shared hidden rules
pytest                                                   # 27 LLM-free checks
```

Any script takes these options:

| Option | Effect |
|---|---|
| `--method ours \| react \| reflexion \| react_memory` | NeSyC (default) or a baseline |
| `--eval_episode_num N` | the first N games of a fixed, seeded order |
| `--hard FILE` | the tasks of FILE instead of the benchmark games |
| `--continual` | NeSyC keeps its learned rules and experience from one episode to the next (set by the hard-task scripts) |
| `--revise`, `--hints`, `--plus`, `--v2` | extensions, see [below](#extensions-not-in-the-paper) |

Stage 1 writes the rules to `Alfworld/data/ilp_rule_<model id>[_hints|_plus|_v2].txt` (`/` and `:` become `_`); results
go to `Alfworld/ablation2_alfworld_results.csv`. Seed 77, clingo `--opt-mode=optN -t 1`, 15 steps per benchmark
episode and 30 per hard task, at most 50 000 conflicts per planning call.

## Results

No API key was available, so the LLM was **Claude, called through Claude Code sub-agents** that answered the
recorded prompts. That is not a raw API call: temperature and the 256-token cap were not enforced, and one agent
answered several prompts in a row (for the baselines, five games in a row, so it could carry knowledge from one game
to the next). All prompts and answers ship in `Alfworld/data/llm_cache_*.json.gz`, and
`bash Alfworld/script/replay.sh` replays all 72 runs below from them (one to two hours on 20 cores) and fails if any
success rate differs. The baselines were not run with Opus or Sonnet, nor on validation 3–5: the sub-agents
answering the ReAct prompts were stopped by those models' safeguards.

Each table compares methods only within one model. Success is the number of episodes solved.

**Claude Fable 5.2**

| | 123 games | 20 games | Hard task | Hard stream (11) | LLM calls: 20 games / stream |
|---|---|---|---|---|---|
| NeSyC | 122 | 20 | solved in 19 actions | 4 | 1 / 2.2 per game (+24 once) |
| ReAct | | 20 | solved in 19 actions | 0 | 6.8 / 35 per game |
| ReAct + notes (`react_memory`) | | | | 10 | – / 23 per game |
| Reflexion, 3 trials | | 20 | solved in 19 actions | 11 (0 on the first trial) | 6.8 / 62 per game |

**Claude Haiku 5.5**

| | 20 games | Hard suite (6) | LLM calls, 20 games |
|---|---|---|---|
| NeSyC | 19 | 0 | 4.5 per game (+ stage 1, once) |
| NeSyC + `--plus` | | 0 | |
| NeSyC + `--v2` | | **3** | |
| ReAct | 20 | 0 | 7.2 per game |
| Reflexion, 3 trials | 20 | 0 | 7.2 per game |

**Claude Opus 5.5** on the dev stream and the first validation streams (NeSyC only):

| | Dev stream (11) | Validation 1 (11) | Validation 2 (11) |
|---|---|---|---|
| NeSyC | 0 | 0 | 0 |
| NeSyC + `--revise` | 2 | 0 | |
| NeSyC + `--hints` | 0 | | 0 |
| NeSyC + `--hints --revise` | 0 | | 0 |
| NeSyC + `--plus` | 0 | | |
| NeSyC + `--v2` (tuned on this stream) | 5 | | |

**Held-out validation streams, three models** (NeSyC only; each stream was fixed before the version it judges was
run on it, and each cell is one run):

| | Opus, val. 3 (10) | Opus, val. 4 (10) | Sonnet 5.5, val. 3 | Sonnet 5.5, val. 4 | Haiku, val. 3 | Haiku, val. 4 |
|---|---|---|---|---|---|---|
| NeSyC | 0 | 0 | 0 | 0 | 0 | 0 |
| NeSyC + `--plus` | 7 | 9 | 0 | 0 | 0 | 9 |
| NeSyC + `--v2` | | 7 | | 8 | | 0 |

**Validation 5: three hidden-rule sets, ten new kitchens each** (fixed before any run on them; one run per cell):

| | Rules A (10) | Rules B (10) | Rules C (10) | Total (30) |
|---|---|---|---|---|
| Opus, NeSyC | 0 | 0 | 0 | 0 |
| Opus, `--plus` | 5 | 5 | 0 | 10 |
| Opus, `--v2` | 9 | 0 | 9 | 18 |
| Sonnet, NeSyC | 0 | 0 | 0 | 0 |
| Sonnet, `--plus` | 0 | 0 | 0 | 0 |
| Sonnet, `--v2` | 0 | 9 | 9 | 18 |
| Haiku, NeSyC | 0 | 0 | 0 | 0 |
| Haiku, `--plus` | 0 | 0 | 10 | 10 |
| Haiku, `--v2` | 0 | 8 | 10 | 18 |

What these runs show:

- **With the same model, NeSyC is ahead of the baselines in one place**: Haiku's hard suite, where `--v2` solves 3
  of 6 and ReAct, Reflexion, the paper's NeSyC and `--plus` solve none (the suite was not used to develop `--v2`).
  On the benchmark games all methods solve (nearly) everything. On Fable's hard stream the baselines that carry
  memory are well ahead (10 and 11 against 4); `--v2` could not be run with Fable (usage limit).
- **NeSyC uses far fewer LLM calls**: one goal parse per game instead of one call per action, about 7 times fewer
  on the benchmark games and 10 to 30 times fewer on the stream.
- **The paper's version solves no hidden-rule kitchen with Opus, Sonnet or Haiku**, on every stream it was run on.
- **The extensions are what makes NeSyC work there, and `--v2` does so more often.** Over the 120 held-out kitchens
  of validation 4 and 5 (three models, four hidden-rule sets) the paper's version solves 0, `--plus` 38 and `--v2`
  69. Per model and stream `--v2` is ahead in 6 of 12 cells, behind in 3 (Opus on validation 4, 9 → 7, and on
  rules B, 5 → 0; Haiku on validation 4, 9 → 0) and tied in 3. Within a cell the outcome is close to all or
  nothing: when a version fails, the LLM blamed its first failures on the wrong condition, and the learned rules
  then leave no plan or keep the agent retrying the same action. One run per cell, so a single cell says little.
- The paper's gap (NeSyC far ahead of ReAct and Reflexion) does not appear in any run. Samples are small: 6–30
  games per cell, one run each.

Details behind the numbers:

- **123 games.** Stage 1 produced a complete rule set, so no action failed and Algorithm 2's re-learning was not
  used; it is exercised by the hard tasks and the tests. The one failure is a goal parse: "Refrigerate the cleaned
  lettuce." became *cooled* instead of *in the fridge*. Commit `bad2329`, given a hand-written (perfect) goal parse
  and its shipped rule file, scored 113/123; the other 10 episodes crashed or had no plan.
- **Hard stream with Fable.** NeSyC learned all four hidden rules and, from one ambiguous failure, one wrong rule
  (nothing can be heated while another object is in the microwave). The paper's method never retracts a rule, so
  every kitchen whose microwave is not empty was left without a plan. ReAct without memory solves nothing.
- **Haiku.** Stage 1 learned too few rules (nothing about where cooling or cleaning can happen); in the hard suite
  NeSyC ends its first episode with a rule set under which no plan exists and carries it into the next five. No
  ReAct or Reflexion episode managed to heat the potato. A first NeSyC run scored 4/20 because the interpreter kept
  a rule that forbade demonstrated actions; it now refuses such rules (that run no longer replays).
- **Opus.** Each extension was judged on a validation stream it had not been tuned on (`--revise` on 1, `--hints` on
  2, `--plus` on 3, `--v2` on 4, where the paper's version and `--plus` were run again for comparison); see
  [Extensions](#extensions-not-in-the-paper). Sonnet and Haiku were run on validation 3 and 4 only.
- **Settings that favour the baselines**: ReAct's thoughts do not use up environment steps, Reflexion counts a game
  as solved if any of its 3 trials succeeds, and the baselines get the same object locations and attributes that
  NeSyC plans with. Demonstrations are retrieved by TF-IDF (the original sentence-embedding retriever is gone).
- **Not run with an LLM**: low and high dynamics (10-game checks with a scripted stand-in only).

## Hard tasks

All task files were fixed before any model ran on them. Success is checked on the simulator's own facts, and only
success is scored.

- `hard_task.json`: one kitchen. *Rinse a mug, chill it and leave it in the coffee machine. Also microwave a potato
  and store it, still hot, in the fridge. When you are done, the fridge and the microwave must both be shut.* About
  20 actions (benchmark games need 3–9), and one hidden precondition no demonstration shows: heating and cooling
  fail unless the appliance door is open.
- `hard_suite.json`: the same instruction in six kitchens, plus a second hidden precondition (the object must have
  been washed first), which the demonstrations contradict.
- `hard_stream.json` (the dev stream): eleven kitchens, the potato must end up hot and washed, and four hidden rules
  hold everywhere: doors must be open to heat or chill; only a washed object can be chilled; only one door may be
  open at a time; a washed object can no longer be heated, so washing first is a dead end.
- `hard_stream_test*.json`: validation streams with the same task. Kitchens are taken in a seeded order over the 44
  eligible ones of `valid_seen` (all six objects present, original task unrelated) and kept if the planner solves
  them when given all the right rules: 22 of the first 30 (validation 1 and 2), 10 of the last 14 (validation 3).
- `hard_stream_test4.json` (validation 4, run with `--split train`): the first 15 eligible games of the training
  split in a seeded order (seed 2027), of which the planner solves 10 when given the right rules; the `--plus` and
  `--v2` planners solve the same 10. It was fixed before any `--v2` run on it and has since been used for diagnosis.
- `hard_stream_test5{a,b,c}.json` (validation 5, `--split train`): the next 45 eligible training games in a seeded
  order (seed 2028), 15 per hidden-rule set, keeping the first 10 that both the `--plus` and the `--v2` planner
  solve when given the right rules (11, 14 and 13 of 15 qualified). Same instruction; the rules differ:
  - A, as the dev stream: doors open to heat or chill, only a washed object can be chilled, a washed object can no
    longer be heated, one door at a time;
  - B: doors open to heat or chill, only a washed object can be heated, a washed object can no longer be chilled,
    any number of doors;
  - C: only chilling needs the door open, both need a washed object, one door at a time.

## What the code does

| Paper | Code |
|---|---|
| Algorithm 1, rule reformulation | `Pipeline.reformulate` in `Alfworld/our_pipeline.py`: the LLM proposes constraints per batch of positive/negative examples (`gen_hypotheses`), clingo scores them with HI (Eq. 9, `interpret`) and returns feedback, for `ITER_MAX = 3` rounds |
| Algorithm 2, rule application | `run_episode` in `Alfworld/env_utils.py`: plan with clingo, act, store the transition in working memory (`remember`); a failed action re-enters Algorithm 1 (`refine`) |
| Demonstrations | `Alfworld/data/demo/total_external_data.json`, the 12 action types in `TARGET_PREDICATES` (31 examples) |
| Baselines (Appendix C) | `Alfworld/baselines.py`: ReAct; Reflexion = ReAct + a plan written after each failed trial; ReAct + notes = a short note after every episode, shown in all later ones. No task-specific few-shot examples: the top-3 demonstrations instead |

Choices the paper leaves open: α = 0.5 and λ = 0.3 in Eq. 9; a hypothesis is one constraint, and the interpreter
keeps a subset of the proposed ones chosen greedily for HI; the background knowledge used for scoring is the static
facts of each example; a rule that forbids a demonstrated action is kept only if it also excludes a failure seen in
the current environment; learned rules are reset at the start of every episode unless `--continual`; a
demonstration's `fact_set` is its semantic parse (the LLM parser runs only if it is missing); plans may not contain
idle steps.

## Extensions (not in the paper)

All off by default, so every NeSyC number without a flag is the paper's method.

- `--revise`: learned rules stay hard constraints, but when the planner finds no plan, the learned rules that no
  failure of the current episode supports become weak constraints (broken as rarely as possible), and a learned
  rule that fired on an action which then worked is deleted.
- `--hints`: three general guidelines in the rule prompt (`ours_prompts/general/rule_hints.txt`): cover where the
  robot must be, what it must hold and what state things must be in; a condition that also holds in a positive
  example is not the cause; explain a failure by what differs from the successes.
- `--plus`: `--hints --revise` plus three fixes, each found by tracing a failed dev run:
  - the rule prompt may use the type predicates of the examples (`is_coffeemachine`, `is_mug`, ...), so a failure
    such as putting a potato on the coffee machine can be explained by its real cause;
  - everything experienced at run time counts as evidence from the current environment (the paper's M, weighted
    above the demonstrations); before, only the latest failure did, so a correct rule that contradicts the
    demonstrations was dropped again at the next failure;
  - the planner gets the effects that ALFWorld's PDDL has and the base program leaves open: chilling removes heat,
    heating removes cold, closing closes, receptacles start closed. Without them it believed a closed fridge was
    still open and a chilled potato still hot. (Opt-in only because it changes which plans are chosen, so earlier
    runs would no longer replay.)
- `--v2`: `--plus` and three changes, each found by tracing the dev stream with Opus:
  - going somewhere puts the robot there. The base program let it stay put and the observation parser drops
    ALFWorld's "You arrive at", so a plan could open the fridge from the countertop. The failure was then stored with
    the wrong location, and the LLM explained it by something else it saw ("a mug cannot be chilled"). Given the
    right rules by a script, the `--plus` planner solves 10 of the 11 dev kitchens and the `--v2` planner all 11;
  - an action that failed is not planned again until the believed state has changed (the world is deterministic and
    a failed action changes nothing); dev runs had retried the same failing action up to ten times in a row;
  - when there is still no plan after doubting, every learned rule becomes a weak constraint, including those a
    failure seemed to confirm; on the dev stream each such freeze came from a rule that fitted its failure by
    coincidence. A broken rule is deleted if the action works, as with `--revise`.

What the extension runs show:

- **Validation 4.** Where a version does well, the first one or two kitchens fail while the hidden rules are learned
  and nearly every later one succeeds (Opus `--plus` from the 2nd kitchen, Sonnet `--v2` from the 3rd). Sonnet with
  `--plus` failed in each kitchen within a few actions and then had no plan for the remaining 23–26 steps: the
  freeze `--v2` removes. Haiku with `--v2` never learned that the microwave must be open; it kept proposing the
  opposite ("nothing can be heated with the door open"), which also fits one failure (a washed potato heated with
  the door open), and retried heating six or seven times per kitchen in states that differed only elsewhere.
- **Validation 5.** Paper's version: no kitchen with any model or rule set. Two failures of `--v2` traced: on rules
  B, Opus learned the heating rules by the second kitchen but kept blaming failed chilling on the door ("an
  appliance that opens cannot chill") instead of the washing, so it washed the mug first and then failed to chill
  it three or four times in every kitchen. Haiku on rules A never learned when heating works either: 49 failed
  attempts to heat over the ten kitchens, ending with the rule that a potato cannot be heated.
- **Dev stream with `--v2` (Opus)**: 5 of 11, the last four in a row; a first version without the retry rule scored
  4 (it no longer replays). Not a fair test, since `--v2` was developed on this stream.
- **Validation 3 (Opus `--plus`).** From the fourth kitchen on every episode succeeds: the rules learned in the first three carry
  over, and two wrong rules the LLM proposed (nothing can be closed with an object inside; nothing can be put on a
  receptacle that is not openable unless that receptacle stands somewhere) were broken when they left no plan, the
  action worked, and they were deleted.
- **Dev stream with `--plus` (Opus), 0.** Opus proposed rules that make the task impossible (a mug cannot be chilled;
  a sliceable object cannot be heated; nothing can be chilled with the door open). Each fits the failure it was
  proposed for, so it counts as confirmed in that episode and is not doubted. Some of those failures were stored
  with the wrong robot location (the bug `--v2` fixes). The type predicates make some of these rules possible
  (`is_mug(O)`), so they help and hurt.
- With the right rules supplied by a script, NeSyC, `--revise` and `--plus` all solve 10 of 11 dev kitchens; the
  eleventh becomes unsolvable once the potato is washed before the agent knows that washed food cannot be heated.
  The `--v2` planner solves all 11.
- Bugs and dropped variants: the first `--plus` run on validation 3 scored 1/10 because a rule with an aggregate
  (`#count{...}`) became an invalid weak constraint and clingo rejected every fallback program; the fixed run reuses
  that run's answers up to that point. Two variants were dropped after a scripted check on the dev stream, before
  any LLM run: a first `--revise` (a rule learned from one failure could be broken for the cost of 3 actions)
  retried a futile action 13 times, and keeping a new rule only if it explained an otherwise unexplained failure
  made the agent relearn correct rules one failure at a time and run out of steps.

## Changes since `bad2329`

- **Crashes and hangs**: `UnboundLocalError` on the first failed action, `ipdb` breakpoint in the plan parser, a
  sentence-embedding model loaded every episode, missing `error_log/` directory, `script/*.sh` run from the wrong
  directory.
- **Rules that do not parse or block every plan**: every proposed rule is checked with clingo and scored against the
  examples before it is kept; `max_tokens` 64 → 256; prompts, demonstrations and base program use one vocabulary
  (`is_open`, not `is_opened`).
- **Demonstrations**: `robot_at` was grounded at room level while actions use receptacles, so the expert rules
  rejected 13 of 17 positive examples. Fact sets are re-grounded (the expert rules now accept 17/17 and reject
  14/14); the eight per-action files are merged into one, which also restores the `put_down_openable` examples.
- **Planner**: an object with no known location could be assumed to be at the goal (empty plan, "do nothing");
  objects listed in two nested receptacles made the program unsatisfiable; the goal prompt kept the first episode's
  predicates; non-stationary initial states were not valid ASP. The robot not moving when it goes somewhere is fixed
  only under `--v2`, so that the earlier runs still replay.
- **LLM**: any OpenAI-compatible endpoint, key from the environment, on-disk cache. LangChain, Replicate,
  sentence-transformers and `environment.yml` are gone (`requirements.txt`).
- **Added**: the baselines, the hard tasks, the extensions, `pytest` checks and `script/replay.sh`.

## Citation

```bibtex
@inproceedings{choinesyc,
  title={NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains},
  author={Choi, Wonje and Park, Jinwoo and Ahn, Sanghyun and Lee, Daehee and Woo, Honguk},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
