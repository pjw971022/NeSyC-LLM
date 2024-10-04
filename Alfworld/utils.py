import re
import os
import yaml
import alfworld
import alfworld.agents.environment
import random
from typing import List, Tuple, Dict, Any
ALFWORLD_DATA_PATH = ""
CONFIG_FILE = './Nesyc/Alfworld/base_config.yaml'
EVAL_SET_NUM = 1
MAX_STEP_NUM = 15
PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def load_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(file_path) as reader:
            return yaml.safe_load(reader)
    except FileNotFoundError:
        print(f"Config file not found: {file_path}")
        return {}

def load_task_ids(file_path: str) -> List[str]:
    """Load task IDs from file."""
    try:
        with open(file_path, 'r') as file:
            return file.read().splitlines()
    except FileNotFoundError:
        print(f"Task ID file not found: {file_path}")
        return []

def check_target_in_arguments(data: Any, target_obj: str, target_recep: str, target_obj_receps: str, task_type: str) -> bool:
    """Check if target objects are in data arguments."""

    if task_type.startswith('pick_clean_then_place'):
        targets = [target_obj, target_recep, 'sinkbasin'] + target_obj_receps
    elif task_type.startswith('pick_heat_then_place'):
        targets = [target_obj, target_recep, 'microwave'] + target_obj_receps
    elif task_type.startswith('pick_cool_then_place'):
        targets = [target_obj, target_recep, 'fridge'] + target_obj_receps
    else:
        targets = [target_obj, target_recep] + target_obj_receps

    # targets = (target_obj, target_recep, target_obj_recep, 'microwave', 'fridge', 'sinkbasin')
    return any(target in arg.name for target in targets for arg in data.arguments[:2])


def parse_location_info(info: str) -> Tuple[str, str, str]:
    match = re.match(r"at\((\w+),\s*(\w+),\s*(\d+)\)\.", info.strip())
    if match:
        return match.groups()
    return None, None, None


def perturb_locations(locations: List[str], dynamics_type:str, perturb_prob: float = 0.3, mode:str = 'base') -> List[str]:
    pattern = r"at\((\w+),\s*(\w+),\s*(\d+)\)"
    extracted = []
    valid_locations = []
    
    for loc in locations:
        match = re.match(pattern, loc)
        if match:
            extracted.append(match.groups())
            valid_locations.append(loc)
        else:
            print(f"Warning: Ignoring invalid location format: {loc}")
    
    if not extracted:
        print("No valid locations found. Returning original list.")
        return locations

    objects, locs, times = zip(*extracted)

    # Get unique locations
    unique_locs = list(set(locs))
    high_perturbed_locations = []
    low_perturbed_locations = []
    if dynamics_type == 'stationary':
        high_perturbed_locations = locations
        low_perturbed_locations = locations
    else:
        for obj, loc, time in zip(objects, locs, times):
            if random.random() < perturb_prob and len(unique_locs) > 1:
                # Choose a random location different from the current one
                new_loc = random.choice([l for l in unique_locs if l != loc])
                high_perturbed_locations.append(f"at({obj}, {new_loc}, {time})")
            else:
                high_perturbed_locations.append(f"at({obj}, {loc}, {time})")
        
        for obj, loc, time in zip(objects, locs, times):
            
            if random.random() < 0.005 and len(unique_locs) > 1:
                # Choose a random location different from the current one
                new_loc = random.choice([l for l in unique_locs if l != loc])
                low_perturbed_locations.append(f"at({obj}, {new_loc}, {time}).")
            else:
                low_perturbed_locations.append(f"at({obj}, {loc}, {time}).")

    if mode == 'noisy':
        random_loc = random.choice(unique_locs)
        high_perturbed_locations.append(f'at(safeglove, {random_loc}, 0).')
        high_perturbed_locations.append(f'object(safeglove).')

    return high_perturbed_locations, low_perturbed_locations


def filter_contradictory_locations(location_info: List[str]) -> List[str]:
    object_locations: Dict[str, str] = {}
    
    for info in reversed(location_info):
        obj, location, time = parse_location_info(info)
        if obj and obj not in object_locations:
            object_locations[obj] = info
    
    return list(object_locations.values())

def process_facts(dynamics_type:str, 
                  perturb:float, 
                  facts: List[Any], 
                  target_obj: str, 
                  target_recep: str, 
                  target_obj_receps: str, 
                  task_type:str,
                  mode:str) -> Tuple[str, str]:
    """Process facts to generate locations and attributes."""
    attributes = []
    locations = []
    object_list = []
    # tmp_lst = [ data.arguments[0].name for data in facts if data.name == 'objecttype' or data.name == 'receptacletype']
    for data in facts:
        d0 = data.arguments[0].name.replace(' ', '_')
        if "statue" in d0:
            continue
        
        if data.name == 'objecttype':
            object_list.append(d0.split('_')[0])

        elif data.name == 'receptacletype':
            attributes.append(f'location({d0}).')
            object_type = d0.split('_')[0]
            attributes.append(f'is_{object_type}({d0}).')
            if "microwave" in d0:
                attributes.append(f'is_heater({d0}).')
            elif "fridge" in d0:
                attributes.append(f'is_cooler({d0}).')
            elif "sinkbasin" in d0:
                attributes.append(f'is_cleaner({d0}).')
            elif "knife" in d0:
                attributes.append(f'is_slicer({d0})')

        if not check_target_in_arguments(data, target_obj, target_recep, target_obj_receps, task_type):
            continue
        
        if data.name == 'inreceptacle' and len(data.arguments) == 2:
            d1 = data.arguments[1].name.replace(' ', '_')
            locations.append(f'at({d0}, {d1}, 0).')
        elif data.name in ['openable', 'heatable', 'coolable', 'sliceable', 'cleanable', 'pickable']:
            attributes.append(f'{data.name}({d0}).')
        elif data.name == 'objecttype':
            attributes.append(f'object({d0}).')
            object_type = d0.split('_')[0]
            attributes.append(f'is_{object_type}({d0}).')
        
    locations = filter_contradictory_locations(locations)
    locations = sorted(map(str, locations))
    attributes = sorted(map(str, attributes))
    attributes = sorted(map(str, attributes))
    object_list = sorted(map(str, object_list))

    high_perturbed_locations, low_perturbed_locations = perturb_locations(locations, dynamics_type ,perturb, mode)

    return '\n'.join(high_perturbed_locations), '\n'.join(low_perturbed_locations), '\n'.join(locations),'\n'.join(attributes),'\n'.join(object_list)

BASE_PREDICATES = """location(L).
object(O).
goal(T).
step(T).
at(O, L, T).
robot_at(L, T).
holding(O, T).
is_heated(O, T).
is_cooled(O, T).
is_cleaned(O, T).
is_looked(O, T).
is_opened(O, T).
is_heater(L).
is_cooler(L).
is_cleaner(L).
openable(L).
cleanable(O).
coolable(O).
heatable(O).
sliceable(O)."""
def parse_predicates(fact_set):

    predicate_pattern = re.compile(r'^(\w+)\(')
    predicates = set()
    
    for line in fact_set.split('\n'):
        line = line.strip()
        if line:
            match = predicate_pattern.match(line)
            if match:
                predicates.add(match.group(1)+'(X)')
    predicates = sorted(list(predicates))
    predicates_str = BASE_PREDICATES + '\n' + '\n'.join(predicates)
    return predicates_str

def parse_answer_set(answer_set):  #@ 
    actions = []
    for atom in answer_set:
        if atom.startswith('action('):
            parts = atom[7:-1].split(',')
            if len(parts) == 3:
                motion = parts[0].split('(')[0]
                object = parts[0].split('(')[1].replace('_', ' ')
                recep = parts[1].replace(')', '').replace('_', ' ')

                if motion == 'pick_up':
                    action_args = f"take {object} from {recep}"
                elif motion == 'put_down':
                    action_args = f"put {object} in/on {recep}"
                elif motion == 'heat':
                    action_args = f"heat {object} with {recep}"
                elif motion == 'cool':
                    action_args = f"cool {object} with {recep}"
                elif motion == 'slice':
                    action_args = f"slice {object} with {recep}"
                elif motion == 'clean':
                    action_args = f"clean {object} with {recep}"
                time = int(parts[2])        
            else:   
                motion = parts[0].split('(')[0]
                object = parts[0].split('(')[1].replace(')', '').replace('_', ' ')
                # param = parts[1].replace(')', '')
                if motion == 'go_to':
                    action_args = f"go to {object}"
                elif motion == 'open' or motion == 'close' or motion == 'use' or motion == 'examine':
                    action_args = f"{motion} {object}"
                time = int(parts[1])
            actions.append((time, f"{action_args}")) #  {param}
    
    # Sort actions by time
    # actions.sort(key=lambda x: x[0])
    # if actions:
    #     import ipdb; ipdb.set_trace()
    # Remove time from the final output
    # actions = [action for _, action in actions]
    return actions

def parse_predicted_plan(answer_sets):
    if len(answer_sets) != 0:    
        selected_answer_set = random.sample(answer_sets, EVAL_SET_NUM)[0]
    try:
        predicted_plan = parse_answer_set(selected_answer_set) if answer_sets else []
        #     import ipdb; ipdb.set_trace()
            
        print(f"\n{'Predicted Plan':=^40}\n{predicted_plan}\n{'':=^40}")
    except Exception as e:
        print("Error in parsing predicted plan.")
        import ipdb; ipdb.set_trace()

    return predicted_plan

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

###############################################################

def parse_act_obs(input_text):
    lines = input_text.strip().split('\n')
    act = None
    obs = None
    semantic_parse = []
    
    for line in lines:
        if line.startswith('Act'):
            act = parse_act(line)
        elif line.startswith('Obs'):
            obs = parse_obs(line)

    step, ob  = obs
    if 'Nothing happens' in ob:
        semantic_parse.append(f"action(go_to(middle_of_room), {step-1}).")
    else:
        if act: 
            semantic_parse.extend(act_to_semantic(act))
        if obs:
            semantic_parse.extend(obs_to_semantic(obs))
    
    return semantic_parse

def parse_act(act_line):
    match = re.match(r'Act (\d+): (.+)', act_line)
    if match:
        step = int(match.group(1))
        action = match.group(2)
        return (step, action)
    return None

def parse_obs(obs_line):
    match = re.match(r'Obs (\d+): (.+)', obs_line)
    if match:
        step = int(match.group(1))
        observation = match.group(2)
        return (step, observation)
    return None

def act_to_semantic(act):
    step, action = act
    semantic = []

    # Pick up action
    if 'take' in action or 'pick up' in action:
        match = re.search(r'(take|pick up) (\w+ \d+) from (\w+ \d+)', action)
        if match:
            obj, loc = match.group(2), match.group(3)
            semantic.append(f"action(pick_up({obj.replace(' ', '_')}, {loc.replace(' ', '_')}), {step}).")

    # Go to action
    elif 'go to' in action:
        match = re.search(r'go to (\w+ \d+)', action)
        if match:
            loc = match.group(1)
            semantic.append(f"action(go_to({loc.replace(' ', '_')}), {step}).")
            semantic.append(f"location({loc.replace(' ', '_')}).")

    # Place action
    elif 'place' in action or 'put' in action:
        match = re.search(r'(place|put) (\w+ \d+) (?:in|on|in/on) (\w+ \d+)', action)
        if match:
            obj, loc = match.group(2), match.group(3)
            semantic.append(f"action(put_down({obj.replace(' ', '_')}, {loc.replace(' ', '_')}), {step}).")

    # Cool action
    elif 'cool' in action:
        match = re.search(r'cool (\w+ \d+) (?:in|at|with) (\w+ \d+)', action)
        if match:
            obj, loc = match.group(1), match.group(2)
            semantic.append(f"action(cool({obj.replace(' ', '_')}, {loc.replace(' ', '_')}), {step}).")

    # clean action
    elif 'clean' in action:
        match = re.search(r'clean (\w+ \d+) (?:in|at|with) (\w+ \d+)', action)
        if match:
            obj, loc = match.group(1), match.group(2)
            semantic.append(f"action(clean({obj.replace(' ', '_')}, {loc.replace(' ', '_')}), {step}).")

    # Heat action
    elif 'heat' in action:
        match = re.search(r'heat (\w+ \d+) (?:in|at|with) (\w+ \d+)', action)
        if match:
            obj, loc = match.group(1), match.group(2)
            semantic.append(f"action(heat({obj.replace(' ', '_')}, {loc.replace(' ', '_')}), {step}).")

    # Use action
    elif 'use' in action:
        match = re.search(r'use (\w+ \d+)', action)
        if match:
            obj = match.group(1)
            semantic.append(f"action(use({obj.replace(' ', '_')}), {step}).")

    # Open action
    elif 'open' in action:
        match = re.search(r'open (\w+ \d+)', action)
        if match:
            obj = match.group(1)
            semantic.append(f"action(open({obj.replace(' ', '_')}), {step}).")

    # Close action
    elif 'close' in action:
        match = re.search(r'close (\w+ \d+)', action)
        if match:
            obj = match.group(1)
            semantic.append(f"action(close({obj.replace(' ', '_')}), {step}).")

    return semantic


def obs_to_semantic(obs):
    step, observation = obs
    semantic = []

    # Robot location
    if 'arrive at' in observation or 'you are at' in observation:
        match = re.search(r'(arrive at|you are at) (\w+ \d+)', observation)
        if match:
            loc = match.group(2)
            semantic.append(f"robot_at({loc.replace(' ', '_')}, {step}).")
    
    # Object location and existence
    receptacle_match = re.search(r'On the (\w+ \d+),', observation)
    if receptacle_match:
        receptacle = receptacle_match.group(1)
        objects = re.findall(r'a (\w+ \d+)', observation)
        for obj in objects:
            semantic.extend([
                # f"object({obj.replace(' ', '_')}).",
                f"location({receptacle.replace(' ', '_')}).",
                f"at({obj.replace(' ', '_')}, {receptacle.replace(' ', '_')}, {step})."
            ])
    else:
        matches = re.findall(r'(\w+ \d+) (?:is in|is on|in/on|,? you see) (\w+ \d+)', observation)
        for match in matches:
            obj, loc = match[0], match[1]
            semantic.extend([
                # f"object({obj.replace(' ', '_')}).",
                f"location({loc.replace(' ', '_')}).",
                f"at({obj.replace(' ', '_')}, {loc.replace(' ', '_')}, {step})."
            ])
    
    # Holding object
    if 'pick up' in observation or 'put' in observation:
        match = re.search(r'(?:pick up|put) the (\w+ \d+) (?:from|in|on|in/on) the (\w+ \d+)', observation)
        if match:
            obj, loc = match.group(1), match.group(2)
            if 'pick up' in observation:
                semantic.append(f"holding({obj.replace(' ', '_')}, {step}).")
            else:
                semantic.append(f"at({obj.replace(' ', '_')}, {loc.replace(' ', '_')}, {step}).")
    
    # Object states and actions
    action_patterns = {
        'heat': ('is_heated', 'heat'),
        'cool': ('is_cooled', 'cool'),
        'clean': ('is_cleaned', 'clean'),
    }
    
    for action, (state_predicate, action_predicate) in action_patterns.items():
        if action in observation:
            match = re.search(f'{action} the (\w+ \d+) using the (\w+ \d+)', observation)
            if match:
                obj, tool = match.group(1), match.group(2)
                semantic.append(f"{state_predicate}({obj.replace(' ', '_')}, {step}).")

    # Other object states
    if 'is open' in observation:
        match = re.search(r'(\w+ \d+) is open', observation)
        if match:
            obj = match.group(1)
            semantic.append(f"is_opened({obj.replace(' ', '_')}, {step}).")

    return semantic


from typing import List, Dict, Callable, Tuple, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core._api.deprecation")

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
from ast import literal_eval
import random


class RetrievalEngine:
    def __init__(self,
                 demo_data_path: str = './Nesyc/Alfworld/data/demo/total_external_data.json',):
        self.naturalFormat=True
        self.demo_set = pd.read_json(demo_data_path) # pd.read_pickle(knn_data_path)
        self.sentence_embedder = SentenceTransformer("all-MiniLM-L12-v2")

    def knn_retrieval(self, query, k=3):
        # Find K train examples with closest sentence embeddings to test example
        traj_emb = self.sentence_embedder.encode(query) # 
        topK = []
        for _, trainItem in self.demo_set.iterrows():
            train_step_emb = self.sentence_embedder.encode(trainItem["episode"])
            dists = -1 * cos_sim(traj_emb, train_step_emb)
            for j, dist in enumerate(dists[0]):                
                if len(topK) < k:
                    topK.append((j, trainItem["positive"], trainItem["episode"], dist))
                    topK = sorted(topK, key=lambda x: x[-1])
                else:
                    if dist < topK[-1][-1]:
                        if (j, trainItem["positive"], trainItem["episode"], dist) not in topK:
                            topK.append((j, trainItem["positive"], trainItem["episode"], dist))
                            topK = sorted(topK, key=lambda x: x[-1])
                            topK = topK[:k]

        return [entry for entry in topK]

    def search_demo(self, query, k=3,):
        prompt = ''
        knn_retrieved_examples = self.knn_retrieval(query, k)

        # Add in-context examples from knn retrieval
        for retrieved_tuple in knn_retrieved_examples:
            _ ,positive , retrieved_episode, dist = retrieved_tuple

            if positive:
                prompt += f"\n{retrieved_episode} (success)"
            else:
                prompt += f"\n{retrieved_episode} (fail)"
            
        return prompt



import re

def extract_code_elements(text):
    pattern = r'\((.*?)\)'
    match = re.search(pattern, text)
    
    if match:
        elements = match.group(1).split(',')
        elements = [elem.strip().replace('_',' ') for elem in elements]
        return elements
    else:
        return []
