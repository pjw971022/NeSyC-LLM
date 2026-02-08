import os
import alfworld
import alfworld.agents.environment
from Alfworld.our_pipeline import Pipeline, load_generalized_rules
from Alfworld.llm_utils import google_llm
from utils import *
import json
import datetime
from typing import Dict, Any, Callable, List

def load_paraphrased_instruction(name: str, dynamics_type: str):
    try:
        with open(f"./Alfworld/data/instr/{dynamics_type}_paraphrased_instr.json", "r") as f: # {dynamics_type}
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error: Unable to read the file.")
        return

    for item in data:
        if item["task_name"] == name:
            return item["new_instruction"]

    print(f"Task '{name}' not found in the data.")
    return None

def save_error_log(dynamics_type, name, goal, goal_state, adapted_rules, predicted_plans, total_traj):
    log_entry = {
        "task_name": name,
        "goal": goal,
        "goal_state": goal_state,
        "adapted_rules": adapted_rules,
        "predicted_plans": predicted_plans,
        "total_traj": total_traj,
        "timestamp": datetime.datetime.now().isoformat()
    }
    file_path = f"./Alfworld/data/error_log/{dynamics_type}_error_analysis.json"
    if not os.path.exists(file_path):
        existing_data = []
    else:
        with open(file_path, "r") as f:
            existing_data = json.load(f)

    existing_data.append(log_entry)

    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Error log saved to {file_path}")

def run_alfworld(
    method: str,
    dynamics_type: str,
    pipeline: Pipeline,
    split: str = "train",
    perturb:float = 0.3,
    eval_episode_num: int = 300,
    seed: int = 42,
    to_print: bool = False,
) -> float:
    """
    Run the ALFWorld pipeline.

    Args:
        dynamics_type (str): Type of dynamics to use.
        pipeline (Pipeline): The pipeline object.
        split (str): Data split to use (default: "train").
        to_print (bool): Whether to print detailed output (default: False).

    Returns:
        float: Average reward across all episodes.
    """
    # Load configuration and set up environment
    config = load_config(CONFIG_FILE)
    os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA_PATH
    config['env']['dynamics_type'] = dynamics_type
    config['general']['random_seed'] = seed
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    # Initialize counters and rewards
    cnts: List[int] = [0] * 6
    rs: List[int] = [0] * 6
    gcs: List[int] = [0] * 6
    plan_accs: List[int] = [0] * 6
    raw_r, raw_gc, raw_plan = [], [], []
    for episode in range(eval_episode_num):
        obs, info = env.reset()
        obs = '\n'.join(obs[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        task_type = name.split('-')[0]
        # if 'trial_T20190909_183724_205399' not in name:
        #     continue
        print("\n" + "=" * 50)
        print(f"Episode {episode + 1}: {name.split('/')[-1]}")
        print("=" * 50)

        missing_paraphrased_goal = load_paraphrased_instruction(name, 'missing')
        paraphrased_goal = load_paraphrased_instruction(name, 'original')
        if pipeline.mode != 'complete':
            generalized_rules = load_generalized_rules(pipeline.rule_save_path)
            pipeline.adapted_rules = generalized_rules
        if not paraphrased_goal:
            print(f"Instruction for task '{name}' not found. Skipping...")
            continue

        if dynamics_type == 'high_non_stationary':
            obs = obs.split('\nYour task is to:')[0] + f"\nYour task is to: {missing_paraphrased_goal}"
        else:
            obs = obs.split('\nYour task is to:')[0] + f"\nYour task is to: {paraphrased_goal}"
        
        # Process target objects and receptacles
        target_obj, target_recep, target_obj_receps = process_targets(name, info)

        # Process facts
        facts_list = process_facts(dynamics_type, 
                                   perturb, 
                                   info['facts'][0], 
                                   target_obj, 
                                   target_recep, 
                                   target_obj_receps, 
                                   task_type,
                                   mode=pipeline.mode)
        
        high_perturb_locations, low_perturb_locations, locations, attributes, objects = facts_list


        # Initialize pipeline state
        if method == 'ours' and pipeline.asp and pipeline.ilp:
            pipeline.init_state = high_perturb_locations + attributes
            predicates_str = parse_predicates(attributes)
            pipeline.prompt['goal'] = pipeline.prompt['goal'].replace('XXXXX', predicates_str)
        else:
            obs += '\nObject Information:\n' + high_perturb_locations + attributes
            obs = obs.replace('_',' ')

        r, gc, plan_acc, step_num = run_episode(env, 
                                                method,
                                                pipeline,
                                                obs, 
                                                name, 
                                                dynamics_type, 
                                                target_obj, 
                                                target_recep, 
                                                low_perturb_locations, 
                                                attributes,
                                                paraphrased_goal,)

        # Update statistics
        update_statistics(raw_r, raw_gc, raw_plan, rs, gcs, plan_accs, cnts, r, gc, plan_acc, name)

        # Clean up and print episode results
        pipeline.clean()
        print_episode_results(episode, r, rs, gcs, plan_accs,cnts)
    return raw_r, raw_gc, raw_plan, rs, gcs, plan_accs, cnts


def process_targets(name: str, info: dict) -> Tuple[str, str, List[str]]:
    """Process target objects and receptacles from the task name and info."""
    name_parts = name.split('-')
    target_obj = name_parts[1].lower()
    target_recep = name_parts[3].lower()
    
    target_obj_receps = []
    for data in info['facts'][0]:
        if data.name == 'inreceptacle' and len(data.arguments) == 2:
            d0 = data.arguments[0].name.replace(' ', '_')
            d1 = data.arguments[1].name.replace(' ', '_')
            recep = d1.split('_')[0]
            if target_obj in d0 and recep not in target_obj_receps:
                target_obj_receps.append(recep)
    
    return target_obj, target_recep, target_obj_receps

def filter_actions_by_step(actions, step_number):
    filtered_action = [action for t, action in actions if t == step_number]
    if len(filtered_action) == 0:
        filtered_action = ['do nothing']
    assert len(filtered_action) == 1
    return filtered_action[0]


def check_command_gc(command: str, target: str, obs:str, action: str) -> bool:
    if 'Nothing' in obs:
        return False
    
    if 'heat' in command:
        if 'microwave' not in command:
            return False
    elif 'cool' in command:
        if 'fridge' not in command:
            return False
    elif 'clean' in command:
        if 'sink' not in command:
            return False
    
    if 'go' in command:
        return False
    elif 'open' in command:
        return False

    return target in command and action in command

def check_command_plan(command: str, target: str, obs, action: str) -> bool:
    if 'Nothing' in obs:
        return False

    if 'go' in command:
        ret_val = target in obs and action in command
        return ret_val
    elif 'open' in command:
        return 'open' in obs and target in obs and action in command
    elif 'heat' in command:
        if 'microwave' not in command:
            return False
    elif 'cool' in command:
        if 'fridge' not in command:
            return False
    elif 'clean' in command:
        if 'sink' not in command:
            return False
        
    return target in command and action in command

def update_score_gc(subgoal_tracker: Dict[str, Any], command: str) -> None:
    if command not in subgoal_tracker['gc_prev_commands']:
        subgoal_tracker['gc'] += 1
        subgoal_tracker['gc_prev_commands'].add(command)
        print(f"gc+1 || command: {command}")
        
def update_score_plan(subgoal_tracker: Dict[str, Any], command: str) -> None:
    if command not in subgoal_tracker['plan_prev_commands']:
        if 'open' in command:
            subgoal_tracker['open_cnt'] += 1
        subgoal_tracker['plan_acc'] += 1
        subgoal_tracker['plan_prev_commands'].add(command)
        print(f"plan_acc+1 || command: {command}")

TaskCondition = Callable[[Dict[str, Any], Dict[str, Any]], bool]

def create_check_condition(target: str, action: str) -> TaskCondition:
    def check_condition(action_info: Dict[str, Any], subgoal_tracker: Dict[str, Any]) -> bool:
        command = action_info['command']
        if check_command_gc(command, action_info[target], action_info['obs'], action):
            update_score_gc(subgoal_tracker, command)

        if check_command_plan(command, action_info[target], action_info['obs'], action):
            update_score_plan(subgoal_tracker, command)

    return check_condition

task_conditions: Dict[str, List[TaskCondition]] = {
    'pick_and_place_simple': [
        create_check_condition('target_recep', 'open'),
        create_check_condition('target_obj', 'go'),
        create_check_condition('target_recep', 'go'),
        create_check_condition('target_obj', 'take'),
        create_check_condition('target_recep', 'put'),
    ],
    'look_at_obj_in_light': [
        create_check_condition('target_recep', 'open'),
        create_check_condition('target_obj', 'go'),
        create_check_condition('target_recep', 'go'),
        create_check_condition('target_obj', 'take'),
        create_check_condition('target_recep', 'use')
    ],
    'pick_clean_then_place_in_recep': [
        create_check_condition('target_recep', 'open'),
        create_check_condition('target_obj', 'go'),
        create_check_condition('target_recep', 'go'),
        create_check_condition('target_tool', 'go'),
        
        create_check_condition('target_obj', 'take'),
        create_check_condition('target_recep', 'put'),
        create_check_condition('target_obj', 'clean')
    ],
    'pick_heat_then_place_in_recep': [
        create_check_condition('target_recep', 'open'),
        create_check_condition('target_obj', 'go'),
        create_check_condition('target_recep', 'go'),
        create_check_condition('target_tool', 'go'),

        create_check_condition('target_obj', 'take'),
        create_check_condition('target_recep', 'put'),
        create_check_condition('target_obj', 'heat')
    ],
    'pick_cool_then_place_in_recep': [
        create_check_condition('target_recep', 'open'),
        create_check_condition('target_obj', 'go'),
        create_check_condition('target_recep', 'go'),
        create_check_condition('target_tool', 'go'),

        create_check_condition('target_obj', 'take'),
        create_check_condition('target_recep', 'put'),
        create_check_condition('target_obj', 'cool')
    ],
    'pick_two_obj_and_place': [
        create_check_condition('target_recep', 'open'),
        create_check_condition('target_obj', 'go'),
        create_check_condition('target_recep', 'go'),
        create_check_condition('target_obj', 'take'),
        create_check_condition('target_recep', 'put'),
    ]
}

def check_subgoal_condition(action_info: Dict[str, Any], subgoal_tracker: Dict[str, Any]) -> None:
    task = action_info['name'].split('-')[0]
    if task in task_conditions:
        for condition in task_conditions[task]:
            condition(action_info, subgoal_tracker)

SUB_GOAL_NUM = {
    'pick_and_place_simple': 2,
    'pick_clean_then_place_in_recep': 3,
    'pick_heat_then_place_in_recep': 3,
    'pick_cool_then_place_in_recep': 3,
    'look_at_obj_in_light': 2,
    'pick_two_obj_and_place': 4
}
PLAN_ACC_NUM = {
    'pick_and_place_simple': 4,
    'pick_clean_then_place_in_recep': 6,
    'pick_heat_then_place_in_recep': 6,
    'pick_cool_then_place_in_recep': 6,
    'look_at_obj_in_light': 4,
    'pick_two_obj_and_place': 7
}
def load_target_tool(name: str) -> str:
    if 'pick_heat_then_place_in_recep' in name:
        return 'microwave'
    elif 'pick_cool_then_place_in_recep' in name:
        return 'fridge'
    elif 'pick_clean_then_place_in_recep' in name:
        return 'sinkbasin'

def run_episode(
    env, method, pipeline: Pipeline, obs: str, name: str,
    dynamics_type: str, target_obj: str, target_recep: str, 
    low_perturb_locations: List[str], attributes: str, new_instr: str
) -> Tuple[int, int]:
    """Run a single episode of the ALFWorld environment."""
    retrieval_engine =  RetrievalEngine()
    r = 0
    prev_obs = obs
    step_num = 0
    cur_traj1 = ''
    cur_traj2 = ''
    total_traj = f'Obs 0: {obs}\n'
    error_message = ''
    predicted_plan = ''
    predicted_plans = []
    goal = ''
    demo = ''
    db_update_num = random.randint(2, 6)
    instr_update_num = random.randint(1, 5)
    subgoal_tracker = {'open_cnt':0, 'gc': 0, 'gc_prev_commands': set(), 'plan_acc': 0, 'plan_prev_commands': set()}
    pipeline.set_task(name)

    while step_num < MAX_STEP_NUM:
        if method != 'ours':
            demo = retrieval_engine.search_demo(goal + obs)

        example = process_observation(obs, demo, error_message, cur_traj1, cur_traj2)
        if 'goal' in example.keys():
            goal = example['goal']
        
        if method == 'ours':
            answer_sets = pipeline.eval_single_plan(example, opt=True)
            if pipeline.asp:
                predicted_plan = parse_predicted_plan(answer_sets)
                predicted_plans.append(predicted_plan)
                action = filter_actions_by_step(predicted_plan, step_num)
            else:
                action = answer_sets[0]
        else:
            raise ValueError(f"Invalid method: {method}")
        
        action_info = {
            'command': action,
            'dynamics_type': dynamics_type,
            'low_perturb_locations': low_perturb_locations,
            'new_instr': new_instr,
            'step_num': step_num,
            'db_update_num': db_update_num,
            'instr_update_num': instr_update_num,
        }
        obs, r, done, info = env.step([action_info])
        obs, r, done = process_ob(obs[0]), info['won'][0], done[0]
        total_traj += f'\nAct {step_num}: {action}\nObs {step_num+1}: {obs}'
        
        if '\nDatabase update:' in obs:
            new_state = obs.split('\nDatabase update:')[1]
            if method == 'ours' and pipeline.asp and pipeline.ilp:
                pipeline.init_state = new_state + '\n' + attributes       
                obs = obs.split('\nDatabase update:')[0]
            else:
                obs = obs.replace('_',' ')
            print(f"\n{'Database update':=^40}\n{new_state}\n{'':=^40}")

        obs, error_message = extract_error_message(obs)
        if error_message:
            print(f"\n{'Error':=^40}\n{error_message}\n{'':=^40}")
        
        if method == 'ours':
            cur_traj1 = f'\nAct {step_num}: {action}\nObs {step_num+1}: {obs}'
            cur_traj2 = f'Obs {step_num}: {prev_obs}\nAct {step_num}: {action}\nObs {step_num+1}: {obs}'
            print(f"\n{'Observation':=^40}\n{cur_traj1}\n{'':=^40}")

        result_info = {
            'command': action,
            'name': name,
            'target_obj': target_obj,
            'target_tool': load_target_tool(name),
            'obs': obs,
            'target_recep': target_recep,
        }
        check_subgoal_condition(result_info, subgoal_tracker)

        if done:
            break

        prev_obs = obs
        step_num += 1

    if r == 0:
        if method == 'ours':
            save_error_log(dynamics_type, 
                        name, 
                        goal, 
                        pipeline.goal_state, 
                        pipeline.adapted_rules,
                        predicted_plans,
                        total_traj)
    task = name.split('-')[0]
    gc =  subgoal_tracker['gc'] / (SUB_GOAL_NUM[task])
    plan_acc = subgoal_tracker['plan_acc'] / (PLAN_ACC_NUM[task] + subgoal_tracker['open_cnt'])
    if gc > 1:
        gc = 1
    if plan_acc > 1:
        plan_acc = 1
        
        
    return r, gc, plan_acc, step_num

def process_observation(obs: str, demo:str, error_message:str, cur_traj1: str, cur_traj2: str) -> dict:
    """Process the observation and return an example dictionary."""
    example = {}
    example['demo'] = demo
    obj_info = None
    if 'Object Information:' in obs:
        obj_info = obs.split('Object Information:')[1]
        obs = obs.split('Object Information:')[0]
    
    if '\nYour task is to: ' in obs:
        goal = obs.split('\nYour task is to: ')[1]
        example['goal'] = goal.replace('\n','')
        obs = obs.split('\nYour task is to: ')[0]
        print(f"\n{'Instruction':=^40}\n{goal}\n{'':=^40}")

    if '\nYour new task is to: ' in obs:
        goal = obs.split('\nYour new task is to: ')[1]
        example['goal'] = goal.replace('\n','')
        obs = obs.split('\nYour new task is to: ')[0]
        print(f"\n{'New Instruction':=^40}\n{goal}\n{'':=^40}")
    
    if obj_info is not None:
        example['obs'] = obs + '\nObject Information:\n' + obj_info
    else:
        example['obs'] = obs

    if cur_traj1:
        if 'Nothing' in obs and '' == error_message and 'do nothing' not in cur_traj1:
            example['adapt_rule'] = (cur_traj1, cur_traj2) # Rule refinement
            example['adapt_fact'] = cur_traj1
        else:
            example['adapt_fact'] = cur_traj1

    return example

def extract_error_message(obs: str) -> str:
    """Extract error message from the observation if present."""
    if '\nError:' in obs:
        error_message = obs.split('\nError:')[1]
        obs = obs.split('\nError:')[0]
        return obs, error_message
    return obs, ''

def update_statistics(raw_r: List[int],
                      raw_gc: List[int],
                      raw_plan_acc: List[int],
                      rs: List[int],
                      gcs: List[int], 
                      plan_accs: List[int], 
                      cnts: List[int], 
                      r: int, 
                      gc: int, 
                      plan_acc: int, 
                      name: str) -> None:
    """Update the statistics based on the episode result."""

    for i, (k, v) in enumerate(PREFIXES.items()):
        if name.startswith(k):
            if gc >1:
                gc = 1
            if plan_acc > 1:
                plan_acc = 1
            if r == 1:
                gc = 1
                plan_acc = 1
            raw_r.append(int(r))
            raw_gc.append(gc)
            raw_plan_acc.append(plan_acc)
            rs[i] += r
            gcs[i] += gc
            plan_accs[i] += plan_acc
            cnts[i] += 1
            break

def print_episode_results(episode: int, r: int, rs: List[int], gcs : List[int], plan_accs: List[int], cnts: List[int]) -> None:
    """Print the results of the episode."""
    print('\n' + '-' * 70)

    print(f"Episode {episode + 1} Results:")
    print(f"Reward: {r}")
    print(f"Cumulative Rewards: {rs}")
    print(f"Goal Conditions: {gcs}")
    print(f"Plan Accuracies: {plan_accs}")
    print(f"Counts: {cnts}")
    print(f"Average SR: {sum(rs) / sum(cnts):.4f}")
    print(f"Average GC: {sum(gcs) / sum(cnts):.4f}")
    print(f"Average Plan Acc: {sum(plan_accs) / sum(cnts):.4f}")
    print('-' * 70 + '\n')

if __name__ == "__main__":
    pipeline = Pipeline()  # Assuming Pipeline is imported and initialized correctly
    _ = run_alfworld(pipeline)
