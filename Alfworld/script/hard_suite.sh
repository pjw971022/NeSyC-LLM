#!/bin/bash
# ENGINE=<chat model id> bash Alfworld/script/hard_suite.sh   (six hard tasks in a row; NeSyC keeps what it learns)
cd "$(dirname "$0")/../.." && export PYTHONPATH=$PWD:$PWD/Alfworld
python -m Alfworld.main --ilp --asp --procedure both --hard ./Alfworld/data/hard_suite.json --continual --seed 77 --engine "${ENGINE:?set ENGINE to a chat model id}" "$@"
