import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import os
import argparse
from Alfworld.our_pipeline import Pipeline
from env_utils import run_alfworld
from utils import MAX_STEP_NUM
from data.base.base_program import *

def load_or_create_df(file_path):
    return pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()

def setup_pipeline(args):
    if args.method == 'ours':
        args_dict = vars(args)
        pipeline = Pipeline(args_dict)
        pipeline.cache = {'init': {}, 'goal': {}, 'general_rule':{}, 'general_fact':{}, 'adapt_rule':{}, 'adapt_fact':{}}
        pipeline.asp_program = BASE_ASP_PROGRAM.replace('{max_step}', str(MAX_STEP_NUM))
        pipeline.clingo_seed = args.seed
        path_prompt = {
            'adapt_fact': f'./Nesyc/Alfworld/ours_prompts/adapt/fact.txt',
            'adapt_rule': f'./Nesyc/Alfworld/ours_prompts/adapt/rule.txt',
            'goal': f'./Nesyc/Alfworld/ours_prompts/adapt/goal_state_ours.txt'
        }

        pipeline.load_prompt(path_prompt)
    
    pipeline.method = args.method
    pipeline.engine = args.engine
    return pipeline

def run_experiment(args, pipeline):
    raw_r, raw_gc, raw_plan, rewards, gcs, planaccs, cnts = run_alfworld(
        args.method, args.dynamics, pipeline, 
        split=args.split, eval_episode_num=args.eval_episode_num, seed=args.seed,
    )
    total_reward = sum(rewards) / sum(cnts)
    total_gc = sum(gcs) / sum(cnts)
    total_planacc = sum(planaccs) / sum(cnts)

    return {
        'Method': args.method,
        'COT': args.cot,
        'Engine': args.engine,
        'Grounding(only comp)': getattr(pipeline, 'grounding', False),
        'Fewshot': getattr(pipeline, 'fewshot', False),
        'Dynamics': args.dynamics,
        'Generalization': pipeline.method,
        'COUNT':cnts,
        'Total SR': total_reward,
        'Total GC': total_gc,
        'Total PLANACC': total_planacc,
        'Task SR': rewards,
        'Task GC': gcs,
        'Task PLANACC': planaccs,
        'Perturb': args.perturb,
        'Split': args.split,
        'Seed': args.seed,
        'Raw Reward': raw_r,
        'Raw GC': raw_gc,
        'Raw Plan': raw_plan,
    }

def save_results(results, args):
    new_df = pd.DataFrame(results)
    new_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = load_or_create_df(args.save_path)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(args.save_path, index=False)
    print(tabulate(df, headers='keys', tablefmt='pretty'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics', type=str, default='stationary', help='stationary | low_non_stationary | high_non_stationary')
    parser.add_argument('--procedure', type=str, default='adapt', help='general | adapt | both')
    parser.add_argument('--method', type=str, default='ours', help='ours | react | reflexion | llm_planner | clmasp')
    parser.add_argument('--engine', type=str, default='gemini-1.0-pro', help='gpt-4-0314 | gemini-1.0-pro | gemini-1.5-pro | gemini-1.5-flash')
    parser.add_argument('--split', type=str, default='eval_in_distribution', help='train | eval_in_distribution | eval_out_distribution')
    parser.add_argument('--perturb', type=float, default=0.5, help='missing rate of location')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_path', type=str, default='./Nesyc/Alfworld/ablation2_alfworld_results.csv', help='path to save the results')
    parser.add_argument('--eval_episode_num', type=int, default=20, help='number of episodes to evaluate')
    parser.add_argument('--grounding', action='store_true', help='whether to use grounding')
    parser.add_argument('--cot', action='store_true', help='Chain of thought')
    parser.add_argument('--mode', type=str, default='base', help='base | noisy | incomplete')
    parser.add_argument('--refine', action='store_true', help='whether to refine the rules')
    args = parser.parse_args()

    print("Argument Specification:")
    arg_spec = [
        ["Argument", "Value", "Description"],
        *[[arg, getattr(args, arg), parser._option_string_actions[f'--{arg}'].help] for arg in vars(args)]
    ]
    print(tabulate(arg_spec, headers="firstrow", tablefmt="grid"))
    print("\n")

    np.random.seed(args.seed)
    save_path = f'./Nesyc/Alfworld/data/ilp_rule_{args.engine}.txt'

    if args.procedure == 'general':
        args_dict = vars(args)
        pipeline = Pipeline(args=args_dict)
        pipeline.method = args.method
        pipeline.engine = args.engine
        path_prompt = {
            'general_fact': f'./Nesyc/Alfworld/ours_prompts/general/fact.txt',
            'general_bk': f'./Nesyc/Alfworld/ours_prompts/general/ILP_bk.txt',
            'general_rule': f'./Nesyc/Alfworld/ours_prompts/general/ILP_rule.txt',
        }
        pipeline.rule_save_path = save_path

        pipeline.load_prompt(path_prompt)
        pipeline.generalize_external_traj()
    else:
        pipeline = setup_pipeline(args)
        pipeline.rule_save_path = save_path
        results = [run_experiment(args, pipeline)]
        save_results(results, args)

if __name__ == "__main__":
    main()