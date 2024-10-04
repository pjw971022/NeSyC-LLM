import os
import pickle
import clingo
from clingo.control import Control
from clingo.symbol import parse_term
import pandas as pd
import json
from Alfworld.llm_utils import google_llm, openai_llm, meta_llm
from Alfworld.data.base.base_program import * 
MAX_VERIFICATION_TRIAL = 3
TARGET_PREDICATES = {
                            ###################################################
                            'pick_up_openable':'action(pick_up(O, L), T)',
                            'put_down_openable':'action(put_down(O, L), T)',
                            'pick_up_precondition1':'action(pick_up(O, L), T)',
                            'pick_up_precondition2':'action(pick_up(O, L), T)',
                            'pick_up_precondition3':'action(pick_up(O, L), T)',
                            
                            'put_down_precondition2':'action(put_down(O, L), T)',
                            'heat_precondition3':'action(heat(O, L), T)',
                                    
                            'cool_precondition1':'action(cool(O, L), T)',
                            'cool_precondition2':'action(cool(O, L), T)',
                            'clean_precondition2':'action(clean(O, L), T)',
                            'open_precondition1':'action(open(L), T)',
                            'use_precondition1':'action(use(O), T)',
                        }


from utils import parse_act_obs
def extract_rule(text):
    # Split the text into lines
    splited = text
    extract_rule = splited.replace('```', ' ').replace('**','')
    extract_rule = extract_rule.replace('\\','!').strip()
    return extract_rule + '\n'

def load_generalized_rules(load_path):
    with open(load_path, 'r') as f:
        generalized_rules = f.read()
    return generalized_rules

def save_generalized_rules(generalized_rules, save_path):
    with open(save_path, 'w') as f:
        f.write(generalized_rules)

class Context:
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def rule_filtering(text):
    lines = text.replace('`','').split('\n')
    lines = [line for line in lines if ':-' in line]
    filtered_lines = [line for line in lines if 'take' not in line]
    
    return '\n'.join(filtered_lines)

class Pipeline:
    def __init__(self, args = {}):
        self.asp_program = ''
        self.init_state = ''
        self.dynamic_facts = ''
        self.adapted_rules = ''
        self.adaptation_chat_history = ''
        self.goal_state = ''
        self.method = ''
        self.v = ''
        self.grounding = False
        self.fewshot = False
        self.clingo_seed = 1
        self.engine = 'gemini-1.5-flash'
        self.rule_save_path = ''
        self.temperature = 0.
        self.max_tokens = 64
        self.prompt = {} # a mapping from prompt kind (str) to the prompt (str)
        self.count = 0 # a counter for the number of GPT-3 query
        self.history = ''
        self.step_num = 0
        ###########
        # Cache
        ###########
        self.path_cache = {} # store the mapping from kind (str) to cache file (str)
        self.cache = {} # store the GPT3 responses for visited stories
        self.path_mistakes = 'mistakes.xlsx' # file to store the wrong pridictions
        self.mistakes = [] # store the wrong predictions
        for k,v in args.items():
            setattr(self, k, v)
        self.total_cost = 0
    
    def set_task(self, name):
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                self.v = v

    def clean(self):
        self.init_state = ''
        self.dynamic_facts = ''
        self.adapted_rules = ''
        self.goal_state = ''
        self.adaptation_chat_history = ''
        self.step_num = 0
        self.history = ''

    def load_prompt(self, kind_to_path):
        for kind in kind_to_path:
            with open(kind_to_path[kind], 'r', encoding='utf-8') as f:
                self.prompt[kind] = f.read()
                
    def load_cache(self):
        for kind in self.path_cache:
            if os.path.isfile(self.path_cache[kind]):
                with open(self.path_cache[kind], 'rb') as f:
                    self.cache[kind] = pickle.load(f)
            else:
                self.cache[kind] = {}
    
    def save_cache(self):
        for kind in self.path_cache:
            with open(self.path_cache[kind], 'wb') as f:
                pickle.dump(self.cache[kind], f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.count % 100 == 0:
                with open(self.path_cache[kind]+str(self.count), 'wb') as f:
                    pickle.dump(self.cache[kind], f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _generate_llm_response(self, prompt, instruction='',stop=[]):

        try:
            if 'gemini' in self.engine:
                response = google_llm(
                    prompt=prompt + instruction,
                    model_name=self.engine,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop)
            elif 'gpt' in self.engine:
                response, cost = openai_llm(
                    prompt=prompt + instruction,
                    model_name=self.engine,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop)
                self.total_cost += cost
                print(f"##### Total Cost: {self.total_cost} #####")
            elif 'llama' in self.engine:
                response = meta_llm(
                    prompt=prompt + instruction,
                    model_name=self.engine,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop)

            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def _generate_thought(self, prompt):
        if 'cot' in self.method:
            thought = self._generate_llm_response(prompt, '\nThought:')
            if thought:
                print("\nThought: ", thought)
                return thought
        return ''
    
    def gen_general_fact_response(self, example, kind='general_fact'):
        traj = example['traj']
        target_predicate = example['target_predicate']
        prompt = f"{self.prompt[kind]}\nExternal Trajectory: {traj}\nTarget Predicate: {target_predicate}"
        thought = self._generate_thought(prompt)
        prompt += f"\nThought:{thought}" if thought else ""

        response = self._generate_llm_response(prompt, '\nFact set: ')
        return response


    def gen_ilp_bk_response(self, examples, kind='general_bk'):
        pn_e = examples['pn_e']
        prompt = f"{self.prompt[kind]}\nPositive/Negative Examples:\n{pn_e}"
        general_bk = self._generate_llm_response(prompt, 'Make Background knowledge.\n')
        return general_bk

    def gen_ilp_rule_response(self, examples, kind='general_rule'):
        pn_e = examples['pn_e']
        bk = examples['bk']
        target_predicate = examples['target_predicate']
        action_type = examples['action_type']
        general_rule = ''

        ########## Make Positive Rules ###########
        if 'effect' in action_type:
            prompt = f"{self.prompt[kind]}\nPositive/Negative Examples:\n{pn_e}\n\nBackground Knowledge:\n{bk}\n\nTarget Predicate:\n{target_predicate}\n\n"
            prompt = prompt.replace('XXXXX', 'Positive Rules')
            thought = self._generate_thought(prompt)
            prompt += f"\nThought:{thought}" if thought else ""
            response = self._generate_llm_response(prompt, 'Make Positive Rules.\n')
            general_rule = extract_rule(response)

        ########## Make Constraint Rules ###########
        elif 'precondition' in action_type or 'openable' in action_type:
            prompt = f"{self.prompt[kind]}\nPositive/Negative Examples:\n{pn_e}\n\nBackground Knowledge:\n{bk}\n\nTarget Predicate:\n{target_predicate}\n\n"
            prompt = prompt.replace('XXXXX', 'Constraint Rules')
            thought = self._generate_thought(prompt)
            prompt += f"\nThought:{thought}" if thought else ""
            # import ipdb; ipdb.set_trace()
            response = self._generate_llm_response(prompt,'Make Constraint Rules.\n')
            general_rule = extract_rule(response)
        return general_rule


    def gen_ilp_response(self, examples, kind='general_rule'):
        pn_e = examples['pn_e']
        target_predicate = examples['target_predicate']
        action_type = examples['action_type']
        general_rule = ''

        ########## Make Positive Rules ###########
        if 'effect' in action_type:
            prompt = f"{self.prompt['general_rule']}\nPositive/Negative Examples:\n{pn_e}\n\nTarget Predicate:\n{target_predicate}\n\n"
            prompt = prompt.replace('XXXXX', 'Positive Rules')
            thought = self._generate_thought(prompt)
            prompt += f"\nThought:{thought}" if thought else ""
            response = self._generate_llm_response(prompt, 'Make Positive Rules.\n')
            general_rule = extract_rule(response)

        ########## Make Constraint Rules ###########
        elif 'precondition' in action_type or 'openable' in action_type:
            prompt = f"{self.prompt['general_rule']}\nPositive/Negative Examples:\n{pn_e}\n\nTarget Predicate:\n{target_predicate}\n\n"
            prompt = prompt.replace('XXXXX', 'Constraint Rules')
            thought = self._generate_thought(prompt)
            prompt += f"\nThought:{thought}" if thought else ""
            response = self._generate_llm_response(prompt,'Make Constraint Rules.\n')
            general_rule = extract_rule(response)
        return general_rule

    def generalize_external_traj(self, save=True):
        rule_sets = ''
        for action_type, target_predicate in TARGET_PREDICATES.items():
            action = action_type.split('_')[0]
            path_to_dataset = f'./Nesyc/Alfworld/data/demo/{action}_external_data.json'
            with open(path_to_dataset, 'r') as f:
                dataset = json.load(f)
            p_examples = []
            n_examples = []
            pn_examples_str = ''
            external_trajs = [(data['episode'], data['positive']) for data in dataset if data['action_type'] == action_type]
            for traj_info in external_trajs:
                traj, positive = traj_info
                example = {'action_type':action_type, 'traj': traj, 'target_predicate': target_predicate}
                pn_e = self.gen_general_fact_response(example)
                if positive == 'true':
                    p_examples.append(pn_e.split('\n')[0])
                else:
                    n_examples.append(pn_e.split('\n')[0])
            # p_examples = [data['fact_set'] for data in dataset if data['action_type'] == action_type and data['positive'] == "true"]
            for p_example in p_examples:
                pn_examples_str += f'Positive: {p_example}\n'
            # n_examples = [data['fact_set'] for data in dataset if data['action_type'] == action_type and not data['positive']== "false"]
            for n_example in n_examples:
                pn_examples_str += f'Negative: {n_example}\n'

            examples = {'action_type':action_type, 'pn_e': pn_examples_str, 'target_predicate': target_predicate}
            bk = self.gen_ilp_bk_response(examples)
            print(f"#### BK: {action_type} ####\n{bk}" )
            examples = {'action_type':action_type, 'pn_e': pn_examples_str, 'target_predicate': target_predicate, 'bk': bk}
            rule_set = self.gen_ilp_rule_response(examples)
            print(f"#### ILP rule set: {action_type} ####\n{rule_set}" )
            rule_sets += rule_set + '\n'
       
        rule_sets = rule_filtering(rule_sets)
        if save:
            save_generalized_rules(rule_sets, self.rule_save_path)
        return rule_sets
    

    def gen_goal_state_response(self, instruction, kind):
        prompt = f"{self.prompt[kind]}{instruction.strip()}"
        response = self._generate_llm_response(prompt, '\nSemantic Parse:')
        response = f"goal(T) :- {response}"
        # Heuristic matching
        response = response.replace('is_coffemachine', 'is_coffeemachine')
        response = response.replace('is_soap(', 'is_soapbar(')
        return response
    
    def gen_adapt_fact_response(self, observations, kind):
        prompt = f"{self.prompt[kind]}\n\nEnvironment Trajectory:\n{observations}"
        thought = self._generate_thought(prompt)
        prompt += f"\nThought: {thought}" if thought else ""

        response = self._generate_llm_response(prompt, '\nSemantic Parse:\n') 
        added_facts = response.split('Semantic Parse:')[-1]
        print(f"\n{'New Facts':=^40}\n{added_facts}\n{'':=^40}")

        return [], added_facts
    
    def get_HI_score(self, tp, tn, fp, fn, alpha=0.5):
        # Calculate HI score
        TPR = tp/(tp+fn) if tp+fn > 0 else 0
        FPR = fp/(fp+tn) if fp+tn > 0 else 0
        score = alpha * TPR + (1-alpha) * FPR
        return score

    def gen_adapt_rule_response(self, observations, kind='adapt'):
        # Implement advanced rule refinement process
        # This is a simple version of the Rule Refinement process.
        # To enhance rule quality using HI scores:
        # 1. Calculate tp/tn/fp/fn based on the current rule and experience.
        # 2. Calculate HI score based on the tp/tn/fp/fn values using get_HI_score.
        # 3. Modify prompt to incorporate HI scores
        # 4. Adjust rule generation based on HI feedback

        if kind == 'adapt':
            prompt = f"{self.prompt[kind]}\n\nEnvironment Trajectory:\n{observations}"
        
        if self.adaptation_chat_history == '':
            self.adaptation_chat_history = prompt
        else:
            self.adaptation_chat_history += '\nFeedback: The Program you created has 0 Stable models. Identify what went wrong, improve it, and create it again.\n'
        
        new_program = self._generate_llm_response(self.adaptation_chat_history).replace('```asp','').replace('```','').replace('prolog','')
        print(f"\n{'New Rules':=^40}\n{new_program}\n{'':=^40}")
        new_program = rule_filtering(new_program)
        return f'{new_program}'

    def setup_and_ground(self, program: str) -> clingo.Control:
        context=Context()
        ctl = Control(['0', '--warn=none', '--opt-mode=optN', f'--seed={self.clingo_seed}', '-t', '16'])
        ctl.configuration.solve.models = 10
        ctl.add('base', [], program)
        ctl.ground([('base', [])], context=context)
        return ctl

    def gen_answer_set(self, program, opt=False):
        """
        Args:
            program (str): a string of ASP program
            opt (bool): if true, only optimal answer sets are returned
                        leave it to False when there is no weak constraint
        """
        models = []
        try:
            clingo_control = self.setup_and_ground(program)
        except Exception as e:
            return []
        if opt:
            solve_result = clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True,shown=True)) if model.optimality_proven else None)
        else:
            solve_result = clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True,shown=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models #, complete

    def eval_single_plan(self, example, opt=False, to_print=False):
        cnt = 0
        answer_sets = []
        for kind in example:
            if kind == 'goal':
                responses = self.gen_goal_state_response(example[kind], kind) + '\n'
                self.goal_state = responses
                print(f"\n{'goal':=^40}\n{self.goal_state}\n{'':=^40}")

                program = self.asp_program + self.init_state + '\n' + self.goal_state + '\n\n' + self.adapted_rules +'\n'     
                answer_sets = self.gen_answer_set(program, opt=opt)

            elif kind == 'adapt_fact':
                new_facts = '\n'.join(parse_act_obs(example[kind]))
                self.dynamic_facts +='\n'+ new_facts
                if to_print:
                    print(f"\n{'new facts':=^40}\n{new_facts}\n{'':=^40}")
                cur_state = self.init_state + '\n' + self.dynamic_facts
                # New facts have high priority
                program = self.asp_program + '\n' + cur_state + '\n' + self.goal_state + '\n\n' + self.adapted_rules +'\n'     
                answer_sets = self.gen_answer_set(program, opt=opt)
                
            elif kind == 'adapt_rule':
                traj1, traj2 = example[kind]
                new_facts = '\n'.join(parse_act_obs(traj1))

                self.dynamic_facts += '\n'+ new_facts
                if to_print:
                    print(f"\n{'new facts':=^40}\n{new_facts}\n{'':=^40}")
                cur_state = self.init_state + '\n' + self.dynamic_facts
                while len(answer_sets) == 0 and cnt < MAX_VERIFICATION_TRIAL :
                    self.adapted_rules = self.gen_adapt_rule_response(traj2, kind)
                    program = self.asp_program + cur_state + '\n' + self.goal_state + '\n\n' + self.adapted_rules +'\n'      
                    answer_sets = self.gen_answer_set(program, opt=opt)
                    cnt += 1

                self.adaptation_chat_history = ''
        return answer_sets

