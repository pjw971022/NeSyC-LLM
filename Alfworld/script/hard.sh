#!/bin/bash
# ENGINE=<chat model id> bash Alfworld/script/hard.sh   (the single hard task, Alfworld/data/hard_task.json)
cd "$(dirname "$0")/../.." && export PYTHONPATH=$PWD:$PWD/Alfworld
python -m Alfworld.main --ilp --asp --procedure both --hard --seed 77 --engine "${ENGINE:?set ENGINE to a chat model id}" "$@"
