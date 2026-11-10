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
ENGINE=<model id> bash Alfworld/script/hard.sh           # one very hard task, see below
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

Choices the paper leaves open: α = 0.5 and λ = 0.3 in Eq. 9; a hypothesis is one constraint and the interpreter
keeps a subset of the proposed ones chosen greedily for HI; the background knowledge used for scoring is the static facts of
each example; learned rules are reset at the start of every episode; a demonstration's `fact_set` is used as its
semantic parse (the LLM parser runs only if it is missing).

## Results

| Setting | LLM | Success |
|---|---|---|
| Static, 123 `valid_seen` games | Claude Fable 5.2 | **122 / 123 (99.2 %)**: Pick & Place 31/31, Examine 10/10, Clean 24/25, Heat 14/14, Cool 24/24, Pick Two 19/19 |
| Hard task | Claude Fable 5.2 | solved with 22 of the 30 allowed actions, after 2 failed actions and 2 rule refinements |
| Hard task, no hypotheses (ablation, hand-written goal) | none | fails |
| Low / high dynamics | — | not run with an LLM; 10-game smoke runs with a scripted stand-in only (9/10, 10/10) |

Read these numbers with the following in mind:

- **Not comparable to Table 1.** Different LLM, re-implemented algorithms, repaired demonstrations and planner.
- **How Claude was called.** No API key was available, so Claude Code sub-agents answered the recorded prompts in
  batches. That is not a raw API call: temperature and the 256-token cap were not enforced (the longest answer is
  28 words) and an agent saw the other prompts of its batch. All 149 prompts and answers are in
  `Alfworld/data/llm_cache_claude-fable-5-2.json`; the rules it produced are in
  `Alfworld/data/ilp_rule_claude-fable-5-2.txt`.
- **Replay without a key**, from the transcript (identical numbers, no API call):
  `LLM_CACHE=Alfworld/data/llm_cache_claude-fable-5-2.json ENGINE=claude-fable-5-2 bash Alfworld/script/static.sh`
- **Refinement was not needed in the static run.** Stage 1 produced a complete rule set, so no action failed;
  Algorithm 2's re-entry into Algorithm 1 is exercised by the hard task and the tests only.
- **The one failure** is a goal parse: "Refrigerate the cleaned lettuce." became *cooled* instead of *in the fridge*.
- **Before these fixes**, commit `bad2329` with a hand-written (perfect) goal parse and the shipped rule file
  scored 113/123; the other 10 episodes crashed or had no plan.

## Hard task

`Alfworld/data/hard_task.json` is one deliberately difficult episode in an ALFWorld kitchen:

> Rinse a mug, chill it and leave it in the coffee machine. Also microwave a potato and store it, still hot, in the
> fridge. When you are done, the fridge and the microwave must both be shut.

It needs about 20 actions (the benchmark tasks need 3–9), both objects start inside or next to closed appliances,
and the environment adds a precondition no demonstration shows: heating and cooling fail unless the appliance door
is open. The agent only finds that out by failing and has to learn the rule on the spot. Success is checked on the
simulator's own facts.

With Claude Fable 5.2 the agent failed at `heat potato 1 with microwave 1`, learned
`:- action(heat(O, L), T), openable(L), not is_open(L, T).`, later failed to close the microwave from across the
room, learned that precondition too, and finished on action 22. The episode is scored on success only (GC and
plan accuracy are set to it). Raise the difficulty with `max_steps` and `needs_open` in the JSON.

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
