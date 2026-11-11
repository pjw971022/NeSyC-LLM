#!/bin/bash
# ENGINE=<chat model id> bash Alfworld/script/hard_stream.sh   (eleven hard tasks in a row with shared hidden rules)
cd "$(dirname "$0")/../.." && export PYTHONPATH=$PWD:$PWD/Alfworld
python -m Alfworld.main --ilp --asp --procedure both --hard ./Alfworld/data/hard_stream.json --continual --seed 77 --engine "${ENGINE:?set ENGINE to a chat model id}" "$@"
