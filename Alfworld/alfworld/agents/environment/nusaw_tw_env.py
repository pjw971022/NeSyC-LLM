import os
import json
from tqdm import tqdm
import textworld
import textworld.agents
import textworld.gym

from alfworld.agents.utils.misc import Demangler, get_templated_task_desc, add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv,AlfredDemangler,AlfredInfos,AlfredExpert

TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}

class NusawAlfredInfos(AlfredInfos):
    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        return state

def calculate_matching_degree(predict, expert_plan):
    matching_count = 0
    for p, l in zip(predict, expert_plan):
        if p == l:
            matching_count += 1
        else:
            break
    return matching_count / len(expert_plan)

import random
class CustomRewardWrapper(textworld.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal = ''
        # self.random_db_update_step = random.randint(1, 3)
        # self.random_instr_update_step = random.randint(1, 3)

    def step(self, action_info: dict):
        ##### Preprocessing #####

        if action_info['dynamics_type'] == 'high_non_stationary':
            if 'go' in action_info['command']:
                if random.random() < 0.3:
                    action_info['command'] = 'moving error'
        
        ##### Postprocessing #####
        game_state, score, done = super().step(action_info['command'])
        if 'No ' in action_info['command']:
            done = True
        if 'stop' in action_info['command']:
            done = True

        if 'safeglove' in action_info['command']:
            game_state['feedback'] += '\nsafeglove is broken. so you can not use it anymore.'

        step_num = action_info['step_num']
        if action_info['dynamics_type'] == 'high_non_stationary':
            # TODO: debug new task
            if step_num == action_info['instr_update_num']:
                new_instr = action_info['new_instr']
                game_state['feedback'] += f'\nYour new task is to: {new_instr}'

            if 'moving error' in action_info['command']:
                game_state['feedback'] += '\nError: The robot has encountered an error while moving.'
                
            if step_num == action_info['db_update_num']:
                locs = action_info['low_perturb_locations'].replace('0', f'{step_num}')
                game_state['feedback'] += f'\nDatabase update:\n{locs}'

        elif action_info['dynamics_type'] == 'low_non_stationary':
            if step_num == action_info['db_update_num']:
                locs = action_info['low_perturb_locations'].replace('0', f'{step_num}')
                game_state['feedback'] += f'\nDatabase update:\n{locs}'

        return game_state, score, done

class NusawAlfredTWEnv(AlfredTWEnv):
    
    def __init__(self, config, train_eval="train"):
        print("Initializing AlfredTWEnv...")
        self.config = config
        self.train_eval = train_eval

        self.goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']
        self.get_game_logic()
        self.gen_game_files(regen_game_files=self.config['env']['regen_game_files'],
                            invalid_game_file=self.config['env']['invalid_game_file'])

        self.random_seed = 42
        
    def gen_game_files(self, invalid_game_file, regen_game_files=False, verbose=False):
        def log(info):
            if verbose:
                print(info)

        self.game_files = []

        if self.train_eval == "train":
            data_path = os.path.expandvars(self.config['dataset']['data_path'])
        elif self.train_eval == "eval_in_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
        elif self.train_eval == "eval_out_of_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_ood_data_path'])
        print("Checking for solvable games...")

        # get task types
        assert len(self.config['env']['task_types']) > 0
        task_types = []
        for tt_id in self.config['env']['task_types']:
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        env = None
        count = 0
        list1 = list(os.walk(data_path, topdown=False))#[:500]
        for root, dirs, files in tqdm(list1):
            if 'traj_data.json' in files:
                count += 1
                pddl_path = os.path.join(root, 'initial_state.pddl')
                json_path = os.path.join(root, 'traj_data.json')
                game_file_path = os.path.join(root, "game.tw-pddl")
                name = game_file_path.split('/')[-3]
                
                # Ablation 2
                # if not (name.startswith('pick_heat') or name.startswith('pick_clean') or name.startswith('pick_two_obj_and_place')): # or name.startswith('look_at') 
                #     continue

                with open(invalid_game_file, 'r') as file:
                    invalid_game_file_task_id = file.read().splitlines()
                if game_file_path.split('/')[-2] in invalid_game_file_task_id:
                    continue
                # Skip if no PDDL file
                if not os.path.exists(pddl_path):
                    log("Skipping %s, PDDL file is missing" % root)
                    continue

                if 'movable' in root or 'Sliced' in root:
                    log("Movable & slice trajs not supported %s" % (root))
                    continue

                # Get goal description
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)

                # Check for any task_type constraints
                if not traj_data['task_type'] in task_types:
                    log("Skipping task type")
                    continue

                # Add task description to grammar
                grammar = add_task_to_grammar(self.game_logic['grammar'], traj_data, goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)

                # Check if a game file exists
                if not regen_game_files and os.path.exists(game_file_path):
                    with open(game_file_path, 'r') as f:
                        gamedata = json.load(f)

                    # Check if previously checked if solvable
                    if 'solvable' in gamedata:
                        if not gamedata['solvable']:
                            log("Skipping known %s, unsolvable game!" % root)
                            continue
                        else:
                            # write task desc to tw.game-pddl file
                            gamedata['grammar'] = grammar
                            if self.goal_desc_human_anns_prob > 0:
                                json.dump(gamedata, open(game_file_path, 'w'))
                            self.game_files.append(game_file_path)
                            continue

                # To avoid making .tw game file, we are going to load the gamedata directly.
                gamedata = dict(pddl_domain=self.game_logic['pddl_domain'],
                                grammar=grammar,
                                pddl_problem=open(pddl_path).read(),
                                solvable=False)
                json.dump(gamedata, open(game_file_path, "w"))

                # Check if game is solvable (expensive) and save it in the gamedata
                if not env:
                    alfred_demangler = AlfredDemangler(shuffle=False)
                    request_infos = textworld.EnvInfos(admissible_commands=True, extras=["gamefile"])
                    expert = AlfredExpert(env, expert_type=self.config["env"]["expert_type"])
                    env = textworld.start(game_file_path, request_infos, wrappers=[alfred_demangler, AlfredInfos, expert])

                log("Generating walkthrough for {}.".format(game_file_path))
                trajectory = self.is_solvable(env, game_file_path)

                gamedata['walkthrough'] = trajectory
                gamedata['solvable'] = gamedata['walkthrough'] is not None
                json.dump(gamedata, open(game_file_path, "w"))

                # Skip unsolvable games
                if not gamedata['solvable']:
                    print("Skipping", game_file_path)
                    continue

                # Add to game file list
                self.game_files.append(game_file_path)

                # Print solvable
                expert_steps = len(gamedata['walkthrough'])
                log("%s (%d steps), %d/%d solvable games" % (game_file_path, expert_steps, len(self.game_files), count))

        print(f"Overall we have {len(self.game_files)} games in split={self.train_eval}")
        self.num_games = len(self.game_files)

        if self.train_eval == "train":
            num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_train_games]
            self.num_games = len(self.game_files)
            print("Training with %d games" % (len(self.game_files)))
        else:
            num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_eval_games]
            self.num_games = len(self.game_files)
            print("Evaluating with %d games" % (len(self.game_files)))
    
    def init_env(self, batch_size):
        domain_randomization = self.config["env"]["domain_randomization"]
        if self.train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, NusawAlfredInfos, CustomRewardWrapper] # , CustomRewardWrapper

        # Register a new Gym environment.
        request_infos = textworld.EnvInfos(won=True,
                                           admissible_commands=True,
                                            facts=True,
                                            command_templates=True,
                                            intermediate_reward=True,
                                           extras=["gamefile","init_constraints"])

        max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]

        env_id = textworld.gym.register_games(self.game_files, request_infos,
                                              batch_size=batch_size,
                                              asynchronous=True,
                                              max_episode_steps=max_nb_steps_per_episode,
                                              wrappers=wrappers)
        # Launch Gym environment.
        env = textworld.gym.make(env_id)

        return env
