# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks in Open Domains

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openreview.net/pdf?id=VoayJihXra)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-green)](https://openreview.net/forum?id=VoayJihXra)
[![arXiv](https://img.shields.io/badge/arXiv-2503.00870-blue)](https://arxiv.org/abs/2503.00870)

<p align="center"><img src="static/videos/ani_v2.gif" alt="NeSyC Framework" width="100%"></p>

[Jinwoo Park](https://pjw971022.github.io/)\*, [Wonje Choi](https://scholar.google.com/citations?user=L4d1CjEAAAAJ&hl=ko)\*, [Sanghyun Ahn](https://scholar.google.co.kr/citations?user=xGh7hdIAAAAJ&hl=ko), [Daehee Lee](https://www.linkedin.com/in/daehee-lee-10b396246/?locale=en_US), [Honguk Woo](https://scholar.google.co.kr/citations?user=Gaxjc7UAAAAJ&hl=en) — SKKU, CMU (\*equal contribution)

NeSyC learns action rules (ASP integrity constraints) from a few demonstrations, plans with them, and re-learns them
whenever an action fails. This repository is the ALFWorld implementation.

> **Status.** The code and data behind the paper's Table 1 no longer exist. This is a re-implementation of
> Algorithms 1–2 that runs end to end; it has **not** been run with `gpt-4o-2024-08-06`, so it does not claim the
> paper's 82.9 %. What we did run is listed under [Results](#results).

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

Free tiers are rate-limited. Every answer is cached in `llm_cache.json`, so an interrupted run resumes where it
stopped and a finished run replays for free. Use a non-reasoning chat model, or raise `--max_tokens` (default 256).

## Run

```bash
ENGINE=<model id> bash Alfworld/script/static.sh         # learn rules, then 123 valid_seen games (static)
ENGINE=<model id> bash Alfworld/script/low_dynamics.sh
ENGINE=<model id> bash Alfworld/script/high_dynamics.sh
ENGINE=<model id> bash Alfworld/script/hard.sh           # one hard task, see below
ENGINE=<model id> bash Alfworld/script/hard_suite.sh     # six harder tasks in a row, see below
ENGINE=<model id> bash Alfworld/script/hard_stream.sh    # eleven tasks with shared hidden rules, see below
ENGINE=<model id> bash Alfworld/script/static.sh --method react       # baselines: react | reflexion | react_memory
pytest                                                   # LLM-free checks
```

Stage 1 writes the rules to `Alfworld/data/ilp_rule_<model id>.txt` (`/` and `:` become `_`); results go to
`Alfworld/ablation2_alfworld_results.csv`. Seed 77, clingo `--opt-mode=optN -t 1`, 15 steps per episode, at most 50 000 conflicts per planning call.

## What the code does

| Paper | Code |
|---|---|
| Algorithm 1, rule reformulation | `Pipeline.reformulate` in `Alfworld/our_pipeline.py`: the LLM proposes constraints per batch of positive/negative examples (`gen_hypotheses`), clingo scores them with HI (Eq. 9, `interpret`) and returns feedback, for `ITER_MAX = 3` rounds |
| Algorithm 2, rule application | `run_episode` in `Alfworld/env_utils.py`: plan with clingo, act, store the transition in working memory (`remember`); a failed action re-enters Algorithm 1 (`refine`) |
| Demonstrations | `Alfworld/data/demo/total_external_data.json`, the 12 action types in `TARGET_PREDICATES` (31 examples) |
| Baselines (Appendix C) | `Alfworld/baselines.py`: ReAct, and Reflexion = ReAct + a plan written after each failed trial. No task-specific few-shot examples: the prompt gets the top-3 demonstrations and the same object facts NeSyC plans with |

Choices the paper leaves open: α = 0.5 and λ = 0.3 in Eq. 9; a hypothesis is one constraint and the interpreter
keeps a subset of the proposed ones chosen greedily for HI; the background knowledge used for scoring is the static facts of
each example; learned rules are reset at the start of every episode; a demonstration's `fact_set` is used as its
semantic parse (the LLM parser runs only if it is missing); a rule that forbids a demonstrated action is kept only
if it also excludes a failure seen in the current environment; plans may not contain idle steps. Baselines: demonstrations are retrieved by TF-IDF (the
original sentence-embedding retriever is gone), thoughts do not use up environment steps, and Reflexion gets 3
trials per game with success in any of them counted. The last two favour the baselines.

## Results

| Setting | LLM | Success |
|---|---|---|
| Static, 123 `valid_seen` games | Claude Fable 5.2 | **122 / 123 (99.2 %)**: Pick & Place 31/31, Examine 10/10, Clean 24/25, Heat 14/14, Cool 24/24, Pick Two 19/19 |
| Hard task | Claude Fable 5.2 | solved with 19 of the 30 allowed actions, after 2 failed actions and 2 rule refinements |
| Hard task, no hypotheses (ablation, hand-written goal) | none | fails |
| Low / high dynamics | — | not run with an LLM; 10-game smoke runs with a scripted stand-in only (9/10, 10/10) |

Against the baselines, same LLM, same 20 games (`--eval_episode_num 20`) and the hard task:

| | Success (20 games) | Actions / game | LLM calls / game | Prompt text / game | Hard task |
|---|---|---|---|---|---|
| NeSyC | 20 / 20 | 5.6 | 1 (+24 once, to learn the rules) | 2.2 kB | solved: 19 actions, 2 failed, 5 LLM calls |
| ReAct | 20 / 20 | 5.8 | 6.8 | 26.9 kB | solved: 19 actions, 1 failed, 22 LLM calls |
| Reflexion (3 trials) | 20 / 20 | 5.8 | 6.8 | 26.9 kB | same as ReAct: nothing failed, so it never reflected |

With this LLM the baselines are as successful as NeSyC and need as many actions on the hard task. What NeSyC
saves is LLM usage: one call per game instead of one per step.

The same comparison with a small model, Claude Haiku 5.5, plus the hard suite:

| | Success (20 games) | Actions / game | LLM calls / game | Hard suite (6 tasks) |
|---|---|---|---|---|
| NeSyC | 19 / 20 | 7.9 | about 3 (+27 once) | 0 / 6 |
| ReAct | 20 / 20 | 7.0 | 7.2 | 0 / 6 |
| Reflexion (3 trials) | 20 / 20 | 7.0 | 7.2 | 0 / 6 |

The suite is hard enough that nothing solves it with this model, so it does not separate the methods either. No
ReAct or Reflexion episode ever managed to heat the potato (the two hidden preconditions together were never
found). NeSyC learned too few rules in stage 1 (nothing about where cooling or cleaning can happen), spent its
first episode failing on those, and ended with a rule set under which the planner finds no plan; in continual
mode that rule set is carried into the remaining five episodes. A first NeSyC run with Haiku scored 4 / 20: the
interpreter had kept a rule that forbade demonstrated actions, which is why it now refuses such rules.
The hard stream (11 kitchens, shared hidden rules, learned knowledge carried over), Claude Fable 5.2:

| | Solved | LLM calls | Notes |
|---|---|---|---|
| NeSyC (`--continual`) | 4 / 11 | 24 | 2 episodes lost while learning, 5 with no plan at all |
| ReAct | 0 / 11 | 389 | no memory: runs into the same hidden rules every time |
| ReAct + notes carried over (`react_memory`) | 10 / 11 | 249 | fails the first episode, then solves every one |
| Reflexion (3 trials per game) | 11 / 11 | 687 | 0 / 11 on the first trial, 10 on the second, 1 on the third |

Carrying knowledge over is what matters here: ReAct without memory solves nothing. But NeSyC is not the best way
to carry it. It learned all four hidden rules correctly and, from one ambiguous failure, one wrong rule (nothing
can be heated while another object is in the microwave). Nothing ever retracts a rule, so every kitchen whose
microwave is not empty was left without a plan. The baselines' free-text memory made no such mistake.

The paper's gap in success rate (ReAct 35.8 %, Reflexion 39.0 %, NeSyC 82.9 % with gpt-4o) does not appear in
any of these runs. NeSyC's measured advantage is the number of LLM calls.

Read these numbers with the following in mind:

- **Not comparable to Table 1.** Different LLM, re-implemented algorithms, repaired demonstrations and planner.
- **How Claude was called.** No API key was available, so Claude Code sub-agents answered the recorded prompts in
  batches. That is not a raw API call: temperature and the 256-token cap were not enforced (the longest answer is
  28 words) and an agent saw the other prompts of its batch (for the baselines: five games in a row, so it could
  carry knowledge from one game to the next). All prompts and answers are in
  `Alfworld/data/llm_cache_claude-fable-5-2.json.gz`, `llm_cache_claude-haiku-5-5.json.gz` and, for the
  `react_memory` run, `llm_cache_claude-fable-5-2_react_memory.json.gz` (about 2,200 calls in all); the
  learned rules are in `Alfworld/data/ilp_rule_claude-*.txt`.
- **Replay without a key**, from the transcript (identical numbers, no API call):
  `LLM_CACHE=Alfworld/data/llm_cache_claude-fable-5-2.json.gz ENGINE=claude-fable-5-2 bash Alfworld/script/static.sh`
  (baselines: add `--method react --eval_episode_num 20`)
- **Refinement was not needed in the static run.** Stage 1 produced a complete rule set, so no action failed;
  Algorithm 2's re-entry into Algorithm 1 is exercised by the hard task and the tests only.
- **The one failure** is a goal parse: "Refrigerate the cleaned lettuce." became *cooled* instead of *in the fridge*.
- **Before these fixes**, commit `bad2329` with a hand-written (perfect) goal parse and the shipped rule file
  scored 113/123; the other 10 episodes crashed or had no plan.

## Hard tasks

`Alfworld/data/hard_task.json` is one deliberately difficult episode in an ALFWorld kitchen:

> Rinse a mug, chill it and leave it in the coffee machine. Also microwave a potato and store it, still hot, in the
> fridge. When you are done, the fridge and the microwave must both be shut.

It needs about 20 actions (the benchmark tasks need 3–9), both objects start inside or next to closed appliances,
and the environment adds a precondition no demonstration shows: heating and cooling fail unless the appliance door
is open. The agent only finds that out by failing and has to learn the rule on the spot. Success is checked on the
simulator's own facts.

With Claude Fable 5.2 the agent failed at `heat potato 1 with microwave 1`, learned
`:- action(heat(O, L), T), openable(L), not is_open(L, T).`, later failed to close the microwave from across the
room, learned that precondition too, and finished on action 19. The episode is scored on success only (GC and
plan accuracy are set to it).

`Alfworld/data/hard_suite.json` is the harder version, fixed before any model was run on it: the same instruction
in six kitchens, with a second hidden precondition on top of the first (an object must also have been washed
before it can be heated or chilled). A failed action only says "Nothing happens", so the agent has to work out
two missing conditions at once, and the demonstrations contradict one of them. With `--continual` NeSyC keeps
its learned rules and experience from one episode to the next, which is where a rule learner should gain over
an agent that starts from scratch each time. Every task is solvable in 19-22 actions: the pipeline does it when
the LLM is replaced by a script that proposes the right rules.

`Alfworld/data/hard_stream.json` is the version built to test knowledge reuse, also fixed before any model ran
on it: eleven kitchens (a twelfth was dropped because it is unsolvable even with the right rules), the potato
must end up hot and washed, and four hidden rules hold everywhere: doors must be open to heat or chill; only a
washed object can be chilled; only one door may be open at a time; a washed object can no longer be heated (so
washing first, the natural order, is a dead end for that potato).

## Changes since `bad2329`

- **Crashes and hangs**: `UnboundLocalError` on the first failed action, `ipdb` breakpoint in the plan parser, a
  sentence-embedding model loaded every episode, missing `error_log/` directory, `script/*.sh` run from the wrong
  directory.
- **Rules that do not parse or block every plan**: every proposed rule is checked with clingo and scored against the
  examples before it is kept; `max_tokens` 64 → 256; prompts, demonstrations and base program now use one
  vocabulary (`is_open`, not `is_opened`).
- **Demonstrations**: `robot_at` was grounded at room level while actions use receptacles, so the expert rules
  rejected 13 of 17 positive examples. Fact sets are re-grounded (the expert rules now accept 17/17 and reject
  14/14); the eight per-action files are merged into one, which also restores the `put_down_openable` examples.
- **Planner**: an object with no known location could be assumed to be at the goal (empty plan, "do nothing");
  objects listed in two nested receptacles made the program unsatisfiable; the goal prompt kept the first
  episode's predicates; non-stationary initial states were not valid ASP.
- **LLM**: any OpenAI-compatible endpoint, key from the environment, on-disk cache. LangChain, Replicate,
  sentence-transformers and `environment.yml` are gone (`requirements.txt`).

## Citation

```bibtex
@inproceedings{choinesyc,
  title={NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains},
  author={Choi, Wonje and Park, Jinwoo and Ahn, Sanghyun and Lee, Daehee and Woo, Honguk},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
