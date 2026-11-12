import argparse
import json
import os
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
from tabulate import tabulate

from Alfworld.data.base.base_program import BASE_ASP_PROGRAM, PHYSICS, V2
from Alfworld.our_pipeline import Pipeline, load_demos
from env_utils import run_alfworld

PROMPTS = {
    'general_fact': './Alfworld/ours_prompts/general/fact.txt',
    'general_bk': './Alfworld/ours_prompts/general/ILP_bk.txt',
    'general_rule': './Alfworld/ours_prompts/general/ILP_rule.txt',
    'goal': './Alfworld/ours_prompts/adapt/goal_state_ours.txt',
    'rule_hints': './Alfworld/ours_prompts/general/rule_hints.txt',
}


def load_or_create_df(file_path):
    return pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()


def setup_pipeline(args):
    pipeline = Pipeline(vars(args))
    if args.hard:
        with open(args.hard) as f:
            tasks = json.load(f)
        pipeline.hard_tasks = tasks if isinstance(tasks, list) else [tasks]
        pipeline.max_steps = pipeline.hard_tasks[0]['max_steps']
        args.eval_episode_num = len(pipeline.hard_tasks)
    pipeline.asp_program = (BASE_ASP_PROGRAM.replace('{max_step}', str(pipeline.max_steps))
                            + (PHYSICS if args.plus else '') + (V2 if args.v2 else ''))
    pipeline.clingo_seed = args.seed
    engine = re.sub(r'[^\w.-]', '_', args.engine)  # model ids may contain '/' or ':'
    suffix = '_v2' if args.v2 else '_plus' if args.plus else '_hints' if args.hints else ''
    pipeline.rule_save_path = args.rule_path or f'./Alfworld/data/ilp_rule_{engine}{suffix}.txt'
    pipeline.load_prompt(PROMPTS)
    pipeline.demos = load_demos(pipeline.parse_demo)
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
        'Dynamics': 'hard_task' if args.hard else args.dynamics,
        'Generalization': pipeline.method,
        'COUNT': cnts,
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
    parser.add_argument('--procedure', type=str, default='adapt', help='general (learn rules) | adapt (evaluate) | both')
    parser.add_argument('--method', type=str, default='ours', choices=['ours', 'react', 'reflexion', 'react_memory'], help='ours | react | reflexion | react_memory')
    parser.add_argument('--engine', type=str, required=True, help='chat model id served by OPENAI_BASE_URL')
    parser.add_argument('--rule_path', type=str, default=None, help='rule file (default: ./Alfworld/data/ilp_rule_<engine>.txt)')
    parser.add_argument('--max_tokens', type=int, default=256, help='max new tokens per LLM call')
    parser.add_argument('--split', type=str, default='eval_in_distribution', help='train | eval_in_distribution | eval_out_of_distribution')
    parser.add_argument('--perturb', type=float, default=0.5, help='missing rate of location')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_path', type=str, default='./Alfworld/ablation2_alfworld_results.csv', help='path to save the results')
    parser.add_argument('--eval_episode_num', type=int, default=20, help='number of episodes to evaluate')
    parser.add_argument('--grounding', action='store_true', help='whether to use grounding')
    parser.add_argument('--cot', action='store_true', help='Chain of thought')
    parser.add_argument('--mode', type=str, default='base', help='base | noisy | incomplete')
    parser.add_argument('--hard', nargs='?', const='./Alfworld/data/hard_task.json', help='run the hard task(s) of this file')
    parser.add_argument('--continual', action='store_true', help='keep the rules learned in one episode for the next ones')
    parser.add_argument('--revise', action='store_true', help='extension: learned rules can be broken and retracted')
    parser.add_argument('--hints', action='store_true', help='extension: three general guidelines in the rule prompt')
    parser.add_argument('--plus', action='store_true', help='extension: --hints --revise, type predicates, PDDL effects, runtime experience as current evidence')
    parser.add_argument('--v2', action='store_true', help='extension: --plus, the robot is where it went, a stuck planner tests all learned rules')
    parser.add_argument('--asp', action='store_true', help='use ASP solver')
    parser.add_argument('--ilp', action='store_true', help='use ILP rules')
    args = parser.parse_args()
    args.plus = args.plus or args.v2
    if args.plus:
        args.hints = args.revise = True

    print("Argument Specification:")
    arg_spec = [
        ["Argument", "Value", "Description"],
        *[[arg, getattr(args, arg), parser._option_string_actions[f'--{arg}'].help] for arg in vars(args)]
    ]
    print(tabulate(arg_spec, headers="firstrow", tablefmt="grid"))
    print("\n")

    np.random.seed(args.seed)
    random.seed(args.seed)
    pipeline = setup_pipeline(args)
    if args.method == 'ours' and args.procedure in ('general', 'both'):
        pipeline.generalize_external_traj()
    if args.procedure in ('adapt', 'both'):
        save_results([run_experiment(args, pipeline)], args)


if __name__ == "__main__":
    main()
