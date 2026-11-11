"""ReAct and Reflexion baselines in the paper's open-domain setting (Appendix C).

No task-specific few-shot trajectories: the prompt carries the top-3 task-agnostic demonstrations and the expanded
observation (object locations and attributes) instead. Reflexion is ReAct plus a plan written after every failed
trial and shown again in the next trial of the same game.
"""
import json
import math
import re
from collections import Counter, defaultdict

from Alfworld.our_pipeline import DEMO_PATH

TOP_K = 3             # retrieved demonstrations (paper: k = 3 for ALFWorld)
MAX_THINKS = 2        # thoughts in a row; one more and the step is spent on "look"
REFLEXION_TRIALS = 3  # attempts per game, success in any of them counts; the paper does not give the number
MEMORY = 3            # plans kept per game, as in the Reflexion code

REACT = """Interact with a household to solve a task. Reply with exactly one line: either a thought, written as \
"think: <reasoning>", or one of these commands:
go to <receptacle> | open <receptacle> | close <receptacle> | take <object> from <receptacle> | \
put <object> in/on <receptacle> | heat <object> with <receptacle> | cool <object> with <receptacle> | \
clean <object> with <receptacle> | use <object>
Write objects and receptacles with their number, for example "take mug 1 from countertop 2".

Experiences from other environments:{demos}
{memory}
Here is the task.
{obs}
>"""

# The reflection instruction of Reflexion (Shinn et al.), without its two task-specific examples.
REFLECT = """You will be given the history of a past experience in which you were placed in an environment and \
given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but \
rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of \
action that accounts for your mistake with reference to specific actions that you should have taken. For example, \
if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will \
need this later when you are solving the same task. Give your plan after "Plan".

{history}

New plan:"""


# ReAct with memory across episodes: the counterpart of NeSyC keeping its rules (--continual).
NOTE = """You just finished the episode below ({outcome}). The next episodes take place in other kitchens that follow \
the same hidden rules. In at most 80 words, write what you learned that will help there: which actions failed, \
what made them work, and the order of steps that worked.

{history}

Lessons:"""


def retrieve(query, k=TOP_K):
    """The k demonstrations closest to the query, as prompt lines."""
    # lean: TF-IDF cosine over words; the lost original used MiniLM sentence embeddings (a local model download)
    with open(DEMO_PATH) as f:
        demos = json.load(f)
    docs = [set(re.findall(r'[a-z]+', demo['episode'].lower())) for demo in demos]
    idf = {word: math.log(len(docs) / n) for word, n in Counter(w for doc in docs for w in doc).items()}
    asked = set(re.findall(r'[a-z]+', query.lower()))

    def score(doc):
        return sum(idf[w] ** 2 for w in doc & asked) / math.sqrt(sum(idf[w] ** 2 for w in doc) or 1)

    top = sorted(range(len(demos)), key=lambda i: -score(docs[i]))[:k]
    return [f"\n{demos[i]['episode']} ({'success' if demos[i]['positive'] == 'true' else 'fail'})" for i in top]


class ReActAgent:
    def __init__(self, llm, reflexion=False, notes=False):
        self.llm = llm
        self.reflexion = reflexion
        self.notes = [] if notes else None  # lessons written after every episode (react_memory)
        self.reflections = defaultdict(list)  # game -> plans written after its failed trials
        self.game = self.prompt = ''

    def start(self, game, obs):
        plans = ''.join(f'Trial {i}: {plan}\n' for i, plan in enumerate(self.reflections[game]))
        memory = f'\nYour memory for the task below:\n{plans}' if plans else ''
        if self.notes:
            memory = '\nLessons from earlier episodes in other kitchens:\n' + ''.join(f'- {n}\n' for n in self.notes)
        self.game = game
        task = obs.split('Your task is to:')[-1].split('\nObject Information')[0]  # the room listing would swamp it
        self.prompt = REACT.format(demos=''.join(retrieve(task)), memory=memory, obs=obs)

    def act(self):
        """The next command for the environment; thoughts are answered with "OK." and cost no environment step."""
        for _ in range(MAX_THINKS + 1):
            lines = [line.strip('> ').strip() for line in self.llm(self.prompt).splitlines()]
            line = next((line for line in lines if line), 'look')
            if not line.lower().startswith(('think', 'thought')):
                return line
            self.prompt += f' {line}\nOK.\n>'
        return 'look'  # still thinking: never send a thought to the environment as a command

    def observe(self, action, obs):
        self.prompt += f' {action}\n{obs}\n>'

    def note(self, solved):
        history = self.prompt.split('Here is the task.\n')[-1]
        self.notes.append(self.llm(NOTE.format(outcome='solved' if solved else 'not solved', history=history)).strip())

    def reflect(self):
        """Reflexion: after a failed trial, write a plan for the next trial of this game."""
        plan = self.llm(REFLECT.format(history=self.prompt.split('Here is the task.\n')[-1])).strip()
        self.reflections[self.game] = (self.reflections[self.game] + [plan])[-MEMORY:]
