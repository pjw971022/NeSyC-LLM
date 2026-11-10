"""NeSyC: Algorithm 1 (rule reformulation) and the pieces of Algorithm 2 (rule application) that need the LLM or clingo."""
import json
import re

import clingo

from Alfworld.llm_utils import llm
from utils import MAX_STEP_NUM, parse_act_obs

ITER_MAX = 3         # interpretation steps of Algorithm 1 (iter_max)
ALPHA = 0.5          # Eq. 9: weight of TPR against FPR
LAMBDA = 0.3         # Eq. 9: weight of the demonstrations T against the working memory M of the current environment
SOLVE_LIMIT = 50000  # conflicts per planning call (deterministic, unlike a wall-clock cap); then the best plan so far
DEMO_PATH = './Alfworld/data/demo/total_external_data.json'
TARGET_PREDICATES = {  # demo action_type (one batch each) -> target action predicate
    'pick_up_openable': 'action(pick_up(O, L), T)',
    'put_down_openable': 'action(put_down(O, L), T)',
    'pick_up_precondition1': 'action(pick_up(O, L), T)',
    'pick_up_precondition2': 'action(pick_up(O, L), T)',
    'pick_up_precondition3': 'action(pick_up(O, L), T)',
    'put_down_precondition2': 'action(put_down(O, L), T)',
    'heat_precondition3': 'action(heat(O, L), T)',
    'cool_precondition1': 'action(cool(O, L), T)',
    'cool_precondition2': 'action(cool(O, L), T)',
    'clean_precondition2': 'action(clean(O, L), T)',
    'open_precondition1': 'action(open(L), T)',
    'use_precondition1': 'action(use(O), T)',
}
ACTION_PREDICATES = {re.search(r'\((\w+)', p).group(1): p for p in TARGET_PREDICATES.values()}


def solve(program):
    """None if clingo rejects the program, otherwise whether it has a stable model."""
    try:
        ctl = clingo.Control(['--warn=none'], logger=lambda *_: None)
        ctl.add('base', [], program)
        ctl.ground([('base', [])])
        return bool(ctl.solve().satisfiable)
    except RuntimeError:
        return None


def extract_rules(text):
    """The integrity constraints in an LLM answer that clingo accepts; each one is a hypothesis."""
    rules = []
    text = re.sub(r'%.*', '', text.replace('`', ''))
    # a constraint starts a statement (":-" not preceded by a rule head) and runs to its final "."
    for match in re.finditer(r'(?:^|(?<=[^\w)\s]))\s*(:-(?:(?!\n\s*:-)(?:[^.]|\.\.))*\.)', text, re.M):
        rule = ' '.join(match.group(1).split())
        constant = re.search(r'[(,]\s*[a-z]\w*\s*[,)]', rule)  # a rule about one named object is not general
        if rule not in rules and not constant and solve(rule) is not None:
            rules.append(rule)
    return rules


def to_facts(fact_set):
    """'{not p(a, T), q(b, T+1)}' -> 'q(b, 1).': closed world, so only positive literals; T is step 0."""
    body = re.sub(r'\bT\b', '0', re.sub(r'\bT\s*\+\s*1\b', '1', fact_set.strip().strip('{}.')))
    literals = [x.strip() for x in re.split(r',\s*(?![^()]*\))', body)]
    return ' '.join(f'{x}.' for x in literals if x and not x.startswith('not '))


def load_demos(parse=None):
    """The demonstrations T as examples. `parse` (the LLM semantic parser) is only called for a demo without fact_set."""
    examples = []
    with open(DEMO_PATH) as f:
        demos = json.load(f)
    for demo in demos:
        if demo['action_type'] not in TARGET_PREDICATES:
            continue
        text = demo.get('fact_set') or parse(demo)
        if solve(to_facts(text)) is None:
            print(f'Skipping demo with an unparsable fact set: {text}')
            continue
        examples.append({'text': text, 'facts': to_facts(text), 'positive': demo['positive'] == 'true',
                         'batch': demo['action_type'], 'target': TARGET_PREDICATES[demo['action_type']]})
    return examples


def interpret(candidates, E_T, E_M):
    """Hypothesis interpreter. Returns (H, HI(H), feedback, misclassified examples).

    Every candidate is one integrity constraint, so an example is a model of BK and H unless one of the rules fires on
    it (BK is the static facts each example carries). H is a subset of the candidates chosen greedily for HI (Eq. 9):
    add every rule that does not lower the score, then drop rules while that raises it; on equal HI the subset that
    keeps more positive examples wins. Two kinds of rule are never kept, because a rule that forbids an admissible
    action can leave the planner with no plan at all: one that fires on an action which already succeeded in the
    environment, and one that fires on a positive demonstration without also excluding a failure seen in the
    environment (only current evidence may override a demonstration).
    """
    E = E_T + E_M
    out = {r: {i for i, e in enumerate(E) if not solve(e['facts'] + r)} for r in candidates}
    positives = {i for i, e in enumerate(E) if e['positive']}
    done = {i for i in positives if 'step' in E[i]}
    failures = {i for i, e in enumerate(E) if not e['positive'] and 'step' in e}
    usable = [r for r in candidates if not out[r] & done and (not out[r] & positives or out[r] & failures)]

    def score(rules):
        excluded = set().union(*(out[r] for r in rules))

        def rate(positive):  # lambda-weighted share of the positive (negative) examples that H does not exclude
            parts = [(w, [i not in excluded for i in span if E[i]['positive'] == positive])
                     for w, span in ((LAMBDA, range(len(E_T))), (1 - LAMBDA, range(len(E_T), len(E))))]
            parts = [(w, sum(kept) / len(kept)) for w, kept in parts if kept]
            return sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else 0.0

        tpr = rate(True)
        return ALPHA * tpr - (1 - ALPHA) * rate(False), tpr

    best = []
    for rule in usable:
        if score(best + [rule]) >= score(best):
            best.append(rule)
    while best:
        rest = max(([r for r in best if r != drop] for drop in best), key=score)
        if score(rest) <= score(best):
            break
        best = rest
    excluded = set().union(*(out[r] for r in best))
    wrong = [e for i, e in enumerate(E) if e['positive'] == (i in excluded)]
    dropped = [r for r in candidates if r not in best]
    missed = [e['text'] for e in wrong if not e['positive']]
    feedback = f'HI score of the kept rules: {score(best)[0]:.2f} (best possible: {ALPHA}).\n'
    if dropped:
        feedback += 'Rejected rules, each excludes a positive example:\n' + '\n'.join(dropped) + '\n'
    if missed:
        feedback += 'Negative examples that no kept rule excludes yet:\n' + '\n'.join(missed) + '\n'
    return best, score(best)[0], feedback, wrong


def load_rules(path):
    with open(path) as f:
        return extract_rules(f.read())


class Pipeline:
    def __init__(self, args={}):
        self.asp_program = ''
        self.init_state = ''
        self.dynamic_facts = ''
        self.goal_state = ''
        self.predicates = ''   # predicates of the current episode, shown to the goal parser
        self.rules = []        # generalized knowledge R (integrity constraints)
        self.demos = []        # experience set T
        self.experience = []   # working memory already merged into T during this episode
        self.memory = []       # working memory M
        self.hard_tasks = []   # tasks of the file given with --hard
        self.hard_task = None  # the one being run
        self.continual = False  # keep the rules learned in one episode for the next ones
        self.max_steps = MAX_STEP_NUM
        self.prompt = {}
        self.engine = ''
        self.rule_save_path = ''
        self.clingo_seed = 1
        self.temperature = 0.
        self.max_tokens = 256
        for k, v in args.items():
            setattr(self, k, v)

    def clean(self):
        self.init_state = self.dynamic_facts = self.goal_state = ''
        self.experience = self.experience + self.memory if self.continual else []  # T keeps growing when continual
        self.memory = []

    def load_prompt(self, kind_to_path):
        for kind, path in kind_to_path.items():
            with open(path, encoding='utf-8') as f:
                self.prompt[kind] = f.read()

    def llm(self, prompt):
        return llm(prompt, self.engine, self.temperature, self.max_tokens)

    # ---------------------------------------------------------------- Algorithm 1: rule reformulation
    def parse_demo(self, demo):
        """Semantic parser for a demonstration that ships without a fact_set."""
        prompt = (f"{self.prompt['general_fact']}\nExternal Trajectory: {demo['episode']}\n"
                  f"Target Predicate: {TARGET_PREDICATES[demo['action_type']]}\nFact set: ")
        return self.llm(prompt).strip().split('\n')[0]

    def gen_hypotheses(self, examples, target, feedback):
        """Hypothesis generator: background knowledge first (the subsumption rationale), then candidate constraints."""
        shown = ''.join(f"{'Positive' if e['positive'] else 'Negative'}: {e['text']}\n" for e in examples)
        bk = self.llm(f"{self.prompt['general_bk']}\nPositive/Negative Examples:\n{shown}Make Background knowledge.\n")
        return extract_rules(self.llm(
            f"{self.prompt['general_rule']}\nFeedback:\n{feedback or 'None'}\n\nPositive/Negative Examples:\n{shown}\n"
            f"Background Knowledge:\n{bk}\n\nTarget Predicate:\n{target}\n\nMake Constraint Rules.\n"))

    def reformulate(self, E_T, E_M, rules, batches):
        """Algorithm 1. `batches` is [(target predicate, examples)]; returns R, the best-HI rule set."""
        feedback = ''
        for i in range(ITER_MAX):
            for target, examples in batches:
                rules = rules + self.gen_hypotheses(examples, target, feedback)
            rules, hi, feedback, wrong = interpret(list(dict.fromkeys(rules)), E_T, E_M)
            print(f'[Algorithm 1] step {i + 1}: {len(rules)} rules, HI={hi:.2f}, misclassified examples={len(wrong)}')
            batches = [(target, examples) for target, examples in batches if any(e in wrong for e in examples)]
            if not batches:  # every example of every batch is classified correctly
                break
        return rules

    def generalize_external_traj(self):
        """Stage 1: learn R from the demonstrations and save it."""
        batches = [(target, [e for e in self.demos if e['batch'] == batch]) for batch, target in TARGET_PREDICATES.items()]
        rules = self.reformulate(self.demos, [], [], batches)
        with open(self.rule_save_path, 'w') as f:
            f.write('\n'.join(rules) + '\n')
        print(f'Saved {len(rules)} rules to {self.rule_save_path}')
        return rules

    # ---------------------------------------------------------------- Algorithm 2: rule application
    def gen_goal_state_response(self, instruction):
        prompt = f"{self.prompt['goal'].replace('XXXXX', self.predicates)}{instruction.strip()}\nSemantic Parse:"
        body = self.llm(prompt).replace('`', '').replace('Semantic Parse:', '').strip().split('\n')[0]
        body = re.sub(r'^goal\(T\)\s*:-', '', body).strip().rstrip('.')
        body = body.replace('is_coffemachine', 'is_coffeemachine').replace('is_soap(', 'is_soapbar(')
        return f'goal(T) :- {body}.'

    def remember(self, model, step, ok):
        """Error handler: store the transition taken at `step` (believed state, action) with its affordance in M."""
        def time_of(atom):
            match = re.search(r'[,(](\d+)\)$', atom)
            return int(match.group(1)) if match else step  # atoms without a time argument are static

        atoms = [a for a in model if time_of(a) == step]
        name, args = re.match(r'action\((\w+)\((.*)\),\d+\)$', next(a for a in atoms if a.startswith('action('))).groups()
        shown = [a for a in atoms if a.startswith(('robot_at(', 'holding(', 'action('))
                 or set(args.split(',')) & set(re.findall(r'\w+', a))]
        self.memory.append({'text': '{' + ', '.join(shown) + '}', 'facts': ' '.join(f'{a}.' for a in atoms),
                            'positive': ok, 'batch': name, 'step': step,
                            'target': ACTION_PREDICATES.get(name, f'action({name}(L), T)')})

    def refine(self):
        """Algorithm 2, lines 12-15: the last action failed, so re-enter Algorithm 1 with T and M, starting from R."""
        seen = self.demos + self.experience
        target = self.memory[-1]['target']
        batch = [e for e in seen + self.memory if e['target'] == target]
        self.rules = self.reformulate(seen, self.memory, self.rules, [(target, batch)])
        self.experience += self.memory
        self.memory = []

    def gen_answer_set(self, program):
        """Task planner: up to 10 optimal stable models, each a list of atom strings; [] if there is no plan."""
        log = []
        ctl = clingo.Control(['0', '--warn=none', '--opt-mode=optN', f'--seed={self.clingo_seed}', '-t', '1',
                              f'--solve-limit={SOLVE_LIMIT}'], logger=lambda code, message: log.append(message))
        ctl.configuration.solve.models = 10
        try:
            ctl.add('base', [], program)
            ctl.ground([('base', [])])
        except RuntimeError:
            print('clingo rejected the program:\n' + '\n'.join(log))
            return []
        models = []
        ctl.solve(on_model=lambda m: models.append((m.optimality_proven, [str(atom) for atom in m.symbols(atoms=True)])))
        return [atoms for proven, atoms in models if proven] or [atoms for _, atoms in models[-1:]]

    def eval_single_plan(self, example, step=0):
        if 'goal' in example:
            self.goal_state = self.gen_goal_state_response(example['goal'])
            print(f"\n{'goal':=^40}\n{self.goal_state}\n{'':=^40}")
        if 'adapt_fact' in example:
            self.dynamic_facts += '\n' + '\n'.join(parse_act_obs(example['adapt_fact']))
        return self.gen_answer_set('\n'.join(
            [self.asp_program, self.init_state, self.dynamic_facts, self.goal_state, f'now({step}).', *self.rules]))
