"""LLM-free checks (the LLM is stubbed). Run from the repo root: pytest"""
import json

from Alfworld import llm_utils
from Alfworld.baselines import ReActAgent, retrieve
from Alfworld import our_pipeline as op
from Alfworld.data.base.base_program import BASE_ASP_PROGRAM, PHYSICS
from Alfworld.our_pipeline import ALPHA, Pipeline, extract_rules, interpret, load_demos
from utils import parse_answer_set

EXPERT = sorted({d['expert_rule'] for d in json.load(open(op.DEMO_PATH))
                 if d['action_type'] in op.TARGET_PREDICATES})
USE_RULE = ':- action(use(O), T), robot_at(L, T), not at(O, L, T).'
# The paper's own counter-example (Appendix D.3.4): it forbids every pick_up.
BAD = ':- action(pick_up(O, L), T), not holding(O, T).'


def stub_llm(monkeypatch, answer):
    prompts = []
    monkeypatch.setattr(Pipeline, 'llm', lambda self, prompt: prompts.append(prompt) or answer(prompt))
    return prompts


def pipeline():
    p = Pipeline()
    p.prompt = {'general_bk': 'BK', 'general_rule': 'RULE', 'goal': 'Predicates:\nXXXXX\nSentence: '}
    p.demos = load_demos()
    return p


def test_demo_fact_sets_agree_with_their_expert_rules():
    demos = load_demos()
    rules, hi, _, wrong = interpret(EXPERT, demos, [])
    assert len(demos) == 31 and hi == ALPHA and not wrong and set(rules) <= set(EXPERT)


def test_extract_rules_keeps_only_constraints_clingo_accepts():
    text = '''```asp
1. :- action(pick_up(O, L), T), not at(O, L, T).
- :- action(put_down(O,
      L), T), not holding(O, T).
holding(O, T+1) :- action(pick_up(O, L), T).
:- action(open(L), T), not robot_at(L, T
:- action(heat(O, L), T), not is_heater(X).
:- action(cool(O, L), T), not is_cooler(L).
Rule 7: :- action(clean(O, L), T), not is_cleaner(L). :- action(open(L), T), not robot_at(L, T).
:- action(heat(potato_1, microwave_1), T).
```'''
    assert extract_rules(text) == [':- action(pick_up(O, L), T), not at(O, L, T).',
                                   ':- action(put_down(O, L), T), not holding(O, T).',
                                   ':- action(cool(O, L), T), not is_cooler(L).',
                                   ':- action(clean(O, L), T), not is_cleaner(L).',
                                   ':- action(open(L), T), not robot_at(L, T).']


def test_interpreter_drops_rules_that_exclude_positive_examples_even_when_they_mask_each_other():
    also_bad = ':- action(pick_up(O, L), T), at(O, L, T).'
    rules, hi, feedback, wrong = interpret(EXPERT + [BAD, also_bad], load_demos(), [])
    assert not {BAD, also_bad} & set(rules) and hi == ALPHA and not wrong and BAD in feedback


def test_rule_that_forbids_an_action_already_taken_in_this_episode_is_never_kept():
    p = pipeline()
    static = ['location(desk_1)', 'is_desk(desk_1)', 'object(bowl_1)', 'object(desklamp_1)']
    p.remember(static + ['robot_at(desk_1,1)', 'at(bowl_1,desk_1,1)', 'action(pick_up(bowl_1,desk_1),1)'], 1, ok=True)
    p.remember(static + ['robot_at(desk_1,2)', 'at(desklamp_1,sidetable_1,2)', 'action(use(desklamp_1),2)'], 2, ok=False)
    at_desk = ':- action(_, T), robot_at(L, T), is_desk(L).'  # excludes the failure, but also the pick_up that worked
    assert interpret([at_desk, USE_RULE], p.demos, p.memory)[0] == [USE_RULE]


def test_algorithm1_feeds_the_interpreter_feedback_back_to_the_llm(monkeypatch):
    bad = ':- action(use(O), T), not holding(O, T).'
    prompts = stub_llm(monkeypatch, lambda p: 'bk.' if 'Background knowledge.' in p
                       else USE_RULE if bad in p else bad)
    p = pipeline()
    demos = [e for e in p.demos if e['batch'] == 'use_precondition1']
    assert p.reformulate(demos, [], [], [(demos[0]['target'], demos)]) == [USE_RULE]
    assert any(bad in prompt for prompt in prompts)


def test_algorithm2_failed_action_re_enters_rule_reformulation(monkeypatch):
    stub_llm(monkeypatch, lambda p: USE_RULE)
    p = pipeline()
    model = ['location(desk_1)', 'object(desklamp_1)', 'is_desklamp(desklamp_1)', 'step(2)', 'robot_at(desk_1,2)',
             'at(desklamp_1,sidetable_1,2)', 'holding(bowl_1,2)', 'action(use(desklamp_1),2)', 'robot_at(desk_1,3)']
    p.remember(model, 2, ok=False)
    assert 'robot_at(desk_1,3)' not in p.memory[0]['facts'] and not p.memory[0]['positive']
    p.refine()
    assert p.rules == [USE_RULE] and p.memory == [] and len(p.experience) == 1


def test_goal_prompt_is_rebuilt_from_the_current_episode_predicates(monkeypatch):
    prompts = stub_llm(monkeypatch, lambda p: ' is_mug(O), at(O, L, T)\nExplanation: ...')
    p = pipeline()
    p.predicates = 'is_mug(X)'
    assert p.gen_goal_state_response('Put a mug away.') == 'goal(T) :- is_mug(O), at(O, L, T).'
    p.predicates = 'is_cd(X)'
    p.gen_goal_state_response('Look at the CD.')
    assert 'is_mug(X)' in prompts[0] and 'is_cd(X)' in prompts[1] and 'is_mug(X)' not in prompts[1]


ROOM = ('location(sink_1). location(counter_1). location(shelf_1). is_shelf(shelf_1). '
        'object(mug_1). object(mug_2). is_mug(mug_1). is_mug(mug_2). ')
GOAL = 'goal(T) :- is_mug(O), is_shelf(L), at(O, L, T).'


def plan(facts):
    p = Pipeline()
    held = ':- action(put_down(O, L), T), not holding(O, T).'  # not among the 12 demo targets; learned online
    models = p.gen_answer_set('\n'.join([BASE_ASP_PROGRAM.replace('{max_step}', '6'), facts, GOAL, held, *EXPERT]))
    return sorted(parse_answer_set(models[0])) if models else None


def test_object_without_a_known_location_is_not_wished_into_the_goal():
    assert plan(ROOM) is None


def test_object_listed_in_two_nested_receptacles_can_still_be_planned_for():
    steps = plan(ROOM + 'at(mug_1, sink_1, 0). at(mug_1, counter_1, 0).')
    assert len(steps) == 4 and steps[-1] == (3, 'put mug 1 in/on shelf 1')


def test_parse_answer_set_skips_atoms_it_cannot_translate():
    assert parse_answer_set(['action(dance(a),0)', 'action(go_to(desk_1),1)', 'at(a,b,1)']) == [(1, 'go to desk 1')]


def test_llm_replays_cached_answers_without_calling_the_api(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_utils, 'CACHE_PATH', str(tmp_path / 'cache.json'))
    monkeypatch.setattr(llm_utils, '_request', lambda *a: 'pong')
    assert llm_utils.llm('ping', 'some-model') == 'pong'
    monkeypatch.setattr(llm_utils, '_request', lambda *a: 1 / 0)
    assert llm_utils.llm('ping', 'some-model') == 'pong'


def test_retrieve_returns_the_three_closest_demonstrations_with_their_outcome():
    demos = retrieve('Your task is to: warm up a bowl of soup in the microwave.')
    assert len(demos) == 3 and 'soup' in demos[0] and all(d.endswith(('(success)', '(fail)')) for d in demos)


def test_react_thoughts_do_not_cost_an_environment_step():
    answers = iter(['think: the mug is on the desk.', '> go to desk 1\nOn the desk 1, you see a mug 1.'])
    agent = ReActAgent(lambda prompt: next(answers))
    agent.start('game', 'You are in a room.\nYour task is to: put a mug in the sink.')
    assert agent.act() == 'go to desk 1'
    agent.observe('go to desk 1', 'On the desk 1, you see a mug 1.')
    assert agent.prompt.endswith('> think: the mug is on the desk.\nOK.\n> go to desk 1\nOn the desk 1, you see a mug 1.\n>')


def test_reflexion_carries_the_plan_of_a_failed_trial_into_the_next_trial_of_the_same_game():
    agent = ReActAgent(lambda prompt: 'Plan: open the fridge first.', reflexion=True)
    agent.start('game-1', 'Your task is to: cool a mug.')
    agent.reflect()
    agent.start('game-1', 'Your task is to: cool a mug.')
    assert 'Trial 0: Plan: open the fridge first.' in agent.prompt
    agent.start('game-2', 'Your task is to: cool a mug.')
    assert 'Your memory' not in agent.prompt


def test_react_never_sends_a_thought_to_the_environment():
    agent = ReActAgent(lambda prompt: 'Think: No mug here, stop and check drawer 1.')
    agent.start('game', 'Your task is to: put a mug in the sink.')
    assert agent.act() == 'look' and agent.prompt.count('OK.') == 3


def test_a_demonstration_is_only_overridden_by_a_failure_in_the_current_environment():
    p = pipeline()
    washed = ':- action(heat(O, L), T), not is_cleaned(O, T).'  # also excludes the demo that heats an unwashed bowl
    assert interpret([washed], p.demos, [])[0] == []
    p.remember(['robot_at(microwave_1,3)', 'holding(potato_1,3)', 'is_heater(microwave_1)',
                'action(heat(potato_1,microwave_1),3)'], 3, ok=False)
    assert interpret([washed], p.demos, p.memory)[0] == [washed]


def test_react_memory_shows_lessons_of_earlier_episodes_in_every_later_one():
    agent = ReActAgent(lambda prompt: 'Open the microwave before heating.', notes=True)
    agent.start('kitchen-1', 'Your task is to: heat a potato.')
    agent.note(solved=False)
    agent.start('kitchen-2', 'Your task is to: heat a potato.')
    assert '- Open the microwave before heating.' in agent.prompt


def test_with_revise_a_wrong_learned_rule_is_broken_instead_of_leaving_no_plan_and_then_dropped():
    wrong = ':- action(put_down(O, L), T), is_shelf(L).'  # forbids the only way to reach the goal
    p = Pipeline()
    p.asp_program = BASE_ASP_PROGRAM.replace('{max_step}', '6')
    p.init_state, p.goal_state = ROOM + 'at(mug_1, sink_1, 0).', GOAL
    p.rules = EXPERT + [':- action(put_down(O, L), T), not holding(O, T).', wrong]
    assert p.eval_single_plan({}) == []
    p.revise, p.learned = True, {wrong}
    plans = p.eval_single_plan({})
    last = max(step for step, _ in parse_answer_set(plans[0]))
    p.remember(plans[0], last, ok=True)  # the put_down that the rule forbade worked
    assert wrong not in p.rules and not p.learned


def test_rule_guidelines_reach_the_prompt_only_with_hints(monkeypatch):
    prompts = stub_llm(monkeypatch, lambda p: '')
    p = pipeline()
    p.prompt['rule_hints'] = 'Guidelines: compare with the successful ones.'
    p.gen_hypotheses(p.demos[:2], 'action(pick_up(O, L), T)', '')
    p.hints = True
    p.gen_hypotheses(p.demos[:2], 'action(pick_up(O, L), T)', '')
    assert 'Guidelines' not in prompts[1] and 'Guidelines' in prompts[3]


def test_with_physics_the_planner_knows_that_chilling_removes_heat_and_heating_removes_cold():
    p = Pipeline()
    room = ('location(fridge_1). location(microwave_1). location(counter_1). is_cooler(fridge_1). is_heater(microwave_1). '
            'object(potato_1). is_potato(potato_1). heatable(potato_1). coolable(potato_1). at(potato_1, counter_1, 0).')
    rules = '\n'.join(EXPERT + [':- action(put_down(O, L), T), not holding(O, T).'])
    def first_plan(goal):
        models = p.gen_answer_set('\n'.join([BASE_ASP_PROGRAM.replace('{max_step}', '8'), PHYSICS, room, goal, rules]))
        return sorted(parse_answer_set(models[0])) if models else None
    # hot and then chilled is not hot any more; chilled and then heated is not cold any more
    assert first_plan('goal(T) :- is_heated(O, T), is_cooled(O, T), is_potato(O).') is None
    plan = first_plan('goal(T) :- is_cooled(O, T), is_potato(O), at(O, fridge_1, T).')
    assert not any(a.startswith('heat') for _, a in plan)


def test_plus_lets_the_rule_prompt_use_the_type_predicates_of_the_examples(monkeypatch):
    prompts = stub_llm(monkeypatch, lambda p: '')
    p = pipeline()
    p.prompt['rule_hints'] = 'Guidelines.'
    example = {'text': '{robot_at(coffeemachine_1, 5), holding(potato_1, 5), is_coffeemachine(coffeemachine_1), '
                       'is_potato(potato_1), action(put_down(potato_1, coffeemachine_1), 5)}', 'positive': False}
    p.gen_hypotheses([example], 'action(put_down(O, L), T)', '')
    p.plus = p.hints = True
    p.gen_hypotheses([example], 'action(put_down(O, L), T)', '')
    assert 'is_coffeemachine' not in prompts[1].split('Positive/Negative Examples')[0]
    allowed = prompts[3].split('Positive/Negative Examples')[0]
    assert 'is_coffeemachine(X)' in allowed and 'is_potato(X)' in allowed and 'Guidelines.' in allowed


def test_with_physics_doors_start_closed_and_closing_closes():
    p = Pipeline()
    room = ('location(fridge_1). location(counter_1). openable(fridge_1). is_fridge(fridge_1). '
            'object(egg_1). is_egg(egg_1). at(egg_1, fridge_1, 0).')
    rules = '\n'.join(EXPERT + [':- action(put_down(O, L), T), not holding(O, T).'])
    def plan(goal):
        models = p.gen_answer_set('\n'.join([BASE_ASP_PROGRAM.replace('{max_step}', '8'), PHYSICS, room, goal, rules]))
        return [a for _, a in sorted(parse_answer_set(models[0]))] if models else None
    assert plan('goal(T) :- holding(O, T), is_egg(O).')[:2] == ['go to fridge 1', 'open fridge 1']
    assert 'open fridge 1' in plan('goal(T) :- holding(O, T), is_egg(O), not is_open(fridge_1, T), T > 4.')


def test_plus_counts_earlier_failures_as_evidence_from_the_current_environment(monkeypatch):
    washed = ':- action(cool(O, L), T), not is_cleaned(O, T).'  # contradicts the demos, which chill unwashed food
    door = ':- action(cool(O, L), T), openable(L), not is_open(L, T).'
    stub_llm(monkeypatch, lambda p: '' if 'Make Constraint Rules' not in p else door if 'is_cleaned(mug_1,7)' in p else washed)
    p = pipeline()
    p.plus = True
    common = ['robot_at(fridge_1,{t})', 'holding(mug_1,{t})', 'is_cooler(fridge_1)', 'openable(fridge_1)', 'cleanable(mug_1)',
              'action(cool(mug_1,fridge_1),{t})']
    p.remember([a.format(t=3) for a in common] + ['is_open(fridge_1,3)'], 3, ok=False)
    p.refine()
    p.remember([a.format(t=7) for a in common] + ['is_cleaned(mug_1,7)'], 7, ok=False)
    p.refine()
    assert washed in p.rules and door in p.rules


def test_a_rule_with_an_aggregate_can_still_be_doubted():
    crowded = ':- action(pick_up(O, L), T), #count{ X : at(X, L, T) } > 1.'  # wrong: forbids the only plan
    p = Pipeline()
    p.asp_program = BASE_ASP_PROGRAM.replace('{max_step}', '6')
    p.init_state, p.goal_state = ROOM + 'at(mug_1, sink_1, 0). at(mug_2, sink_1, 0).', GOAL
    p.rules = EXPERT + [':- action(put_down(O, L), T), not holding(O, T).', crowded]
    p.revise, p.learned = True, {crowded}
    assert p.eval_single_plan({})
