# NeSyC

## Installation

### 1. Set up the Conda environment
Create a new Conda environment using the provided 
`environment.yml` file:
#### Note: Requires python 3.9 or higher
```bash
conda env create -f environment.yml
```

### 2. Install ALFWorld and AI2THOR
Follow the installation instructions for ALFWorld and AI2THOR. 
```
git clone https://github.com/alfworld/alfworld.git alfworld
cd alfworld

pip install -e .[full]
```
#### Note: Requires Clingo 5.7.1

### 3. Run the simulator with headless
For headless VMs and Cloud-Instances, you can use the following steps:

```bash
python docker/docker_run.py --headless

# inside docker
tmux new -s startx  # start a new tmux session

# start nvidia-xconfig
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# start X server on DISPLAY 0
# single X server should be sufficient for multiple instances of THOR
sudo python ~/alfworld/docker/startx.py 0  # if this throws errors e.g "(EE) Server terminated with error (1)" or "(EE) already running ..." try a display > 0

# detach from tmux shell
# Ctrl+b then d

# source env
source ~/alfworld_env/bin/activate

# set DISPLAY variable to match X server
export DISPLAY=:0

# check THOR
python ~/alfworld/docker/check_thor.py

###############
## (300, 300, 3)
## Everything works!!!
```

Note: The above code is sourced from https://github.com/alfworld/alfworld

### 3. Configure the project
Use the `base_config.yaml` file located in the provided folder as your `CONFIG_FILE`.

Please note that you must use the environment in the ./Nesyc/Alfworld/alfworld/agents/environment/nusaw_tw_env.py file at this time.

### 4. Set up OpenAI API Key
Open the `llm_utils.py` file and replace the placeholder in the `OPENAI_API_KEY` variable with your actual OpenAI API key:
```python
OPENAI_API_KEY = "your-api-key-here"
```

## Generate the Rule
To generate the rule from demonstraition for target embodiment in alfworld, use the command. 
```
python ../main.py --procedure general --engine gpt-4o-2024-08-06 --ilp
```

## Running the Project
To run the project in different environments (static, low dynamics, or high dynamics), use the script files provided in the `script` folder.

Example command to run the project:
```bash
python main.py --ilp --asp --dynamics low_non_stationary --eval_episode_num 123 --seed 77 --engine gpt-4-0314
```

You can modify the parameters as needed:
- `--dynamics`: Choose from `static`, `low_non_stationary`, or `high_non_stationary`
- `--eval_episode_num`: Set the number of evaluation episodes
- `--seed`: Set the random seed for reproducibility
- `--engine`: Specify the GPT model to use

Refer to the script files in the `script` folder for more specific run configurations.
