#!/bin/bash
# ENGINE=<chat model id> bash Alfworld/script/high_dynamics.sh   (needs OPENAI_API_KEY; OPENAI_BASE_URL for non-OpenAI endpoints)
cd "$(dirname "$0")/../.." && export PYTHONPATH=$PWD:$PWD/Alfworld
python -m Alfworld.main --ilp --asp --procedure both --dynamics high_non_stationary --eval_episode_num 123 --seed 77 --engine "${ENGINE:?set ENGINE to a chat model id}" "$@"
