ORACLE_ASP_PROGRAM = '''
#const max_steps = {max_step}.
% Time steps
step(0..max_steps).

% States
{at(Object, Location, T) : location(Location)} 1 :- object(Object), step(T).
{robot_at(Location, T) : location(Location)} = 1 :- step(T).
{holding(Object, T) : object(Object)} 1 :- step(T).
{is_heated(Object, T) : heatable(Object)} :- step(T).
{is_cooled(Object, T) : coolable(Object)} :- step(T).
{is_cleaned(Object, T) : cleanable(Object)} :- step(T).
{is_open(Object, T) : openable(Object)} :- step(T).
{is_looked(Object1, Object2, T)} :- object(Object1), object(Object2), step(T).

% Actions
0 { action(go_to(L), T) : location(L);
    action(pick_up(O, L), T) : object(O), location(L);
    action(put_down(O, L), T) : object(O), location(L);
    action(heat(O, L), T) : heatable(O), location(L);
    action(cool(O, L), T) : coolable(O), location(L);
    action(clean(O, L), T) : cleanable(O), location(L);
    action(use(O), T) : object(O);
    action(open(L), T) : openable(L);
    action(close(L), T) : openable(L)
} 1 :- step(T), T < max_steps.
:- 2 { action(_, T) }, step(T).

% State inertia
robot_at(L, T+1) :- robot_at(L, T), not action(go_to(_), T), step(T+1).
holding(O, T+1) :- holding(O, T), not action(put_down(O, _), T), step(T+1).
at(O, L, T+1) :- at(O, L, T), not action(pick_up(O, _), T), step(T+1).
is_heated(O, T+1) :- is_heated(O, T), step(T+1).
is_cooled(O, T+1) :- is_cooled(O, T), step(T+1).
is_cleaned(O, T+1) :- is_cleaned(O, T), step(T+1).
is_looked(O1, O2, T+1) :- is_looked(O1, O2, T), step(T+1).
is_open(L, T+1) :- is_open(L, T), not action(close(L), T), step(T+1).
-is_open(L, T+1) :- -is_open(L, T), not action(open(L), T), step(T+1).

% Basic Action effects
is_cooled(O, T+1) :- action(cool(O, L), T), holding(O, T), robot_at(L, T).
is_heated(O, T+1) :- action(heat(O, L), T), holding(O, T), robot_at(L, T).
is_cleaned(O, T+1) :- action(clean(O, L), T), holding(O, T), robot_at(L, T).
is_looked(O1, O2, T+1) :- action(use(O2), T), holding(O1, T), robot_at(L, T), at(O2,L,T).

is_open(L, T+1) :- action(open(L), T), openable(L).
at(O, L, T+1) :- action(put_down(O, L), T).
holding(O, T+1) :- action(pick_up(O, L), T), robot_at(L, T).
-holding(O, T+1) :- action(put_down(O, L), T).
at(O, L, T+1) :- action(put_down(O, L), T).

% Basic Action constraints
:- holding(O, T), not holding(O, T-1), not action(pick_up(O, _), T-1).
:- at(O, L1, T), at(O, L2, T+1), L1 != L2, not action(pick_up(O, L1), T).
:- is_heated(O, T+1), not is_heated(O, T), not action(heat(O, _), T).
:- is_cooled(O, T+1), not is_cooled(O, T), not action(cool(O, _), T).
:- is_cleaned(O, T+1), not is_cleaned(O, T), not action(clean(O, _), T).
:- holding(O, T), at(O, _, T).
:- robot_at(L1, T), robot_at(L2, T+1), L1 != L2, not action(go_to(L2), T).
:- is_open(L, T+1), not is_open(L, T), not action(open(L), T).
:- is_looked(O1, O2, T+1), not is_looked(O1, O2, T), not action(use(O2), T).
:- is_looked(O1, _, T+1), not holding(O1, T).

:- action(pick_up(O, L), T), not at(O, L, T).
:- action(pick_up(_, _), T), holding(_, T).
:- action(pick_up(O, L), T), not robot_at(L, T).

:- action(put_down(O, _), T), not holding(O, T).
:- action(put_down(_, L), T), not robot_at(L, T).

:- action(heat(O, L), T), not holding(O, T).
:- action(heat(_, L), T), not robot_at(L, T).

:- action(cool(O, L), T), not holding(O, T).
:- action(cool(_, L), T), not robot_at(L, T).

:- action(clean(O, L), T), not holding(O, T).
:- action(clean(_, L), T), not robot_at(L, T).

:- action(open(L), T), not robot_at(L, T).
:- action(close(L), T), not robot_at(L, T).
:- action(use(O), T), robot_at(L, T), not at(O, L, T).

% Goal Constraints
goal(T) :- goal(T-1), step(T), T > 0.
:- not goal(max_steps).

% Minimize the number of actions and steps to reach the goal
#minimize { 1@2,T : action(_, T) }.
#minimize { T@1 : goal(T), not goal(T-1) }.

%######## Action Rreconditions from Domain Adaptation #########
:- action(put_down(O, L), T), openable(L), not is_open(L, T).
:- action(pick_up(O, L), T), openable(L), not is_open(L, T).

%######## Action Rreconditions from Domain Generalization #########
:- action(cool(_, L), T), not is_cooler(L).
:- action(clean(_, L), T), not is_cleaner(L).
:- action(heat(_, L), T), not is_heater(L).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial state
location(middle_of_room).
robot_at(middle_of_room, 0).
#show action/2.
'''

BASE_ASP_PROGRAM = '''
#const max_steps = {max_step}.
% Time steps
step(0..max_steps).

% States
{at(Object, Location, T) : location(Location)} 1 :- object(Object), step(T).
{robot_at(Location, T) : location(Location)} = 1 :- step(T).
{holding(Object, T) : object(Object)} 1 :- step(T).
{is_heated(Object, T) : heatable(Object)} :- step(T).
{is_cooled(Object, T) : coolable(Object)} :- step(T).
{is_cleaned(Object, T) : cleanable(Object)} :- step(T).
{is_open(Object, T) : openable(Object)} :- step(T).
{is_looked(Object1, Object2, T)} :- object(Object1), object(Object2), step(T).

% Actions
0 { action(go_to(L), T) : location(L);
    action(pick_up(O, L), T) : object(O), location(L);
    action(put_down(O, L), T) : object(O), location(L);
    action(heat(O, L), T) : heatable(O), location(L);
    action(cool(O, L), T) : coolable(O), location(L);
    action(clean(O, L), T) : cleanable(O), location(L);
    action(use(O), T) : object(O);
    action(open(L), T) : openable(L);
    action(close(L), T) : openable(L)
} 1 :- step(T), T < max_steps.
:- 2 { action(_, T) }, step(T).

% State inertia
at(O, L, T+1) :- at(O, L, T), not action(pick_up(O, _), T), step(T+1).
is_heated(O, T+1) :- is_heated(O, T), step(T+1).
is_cooled(O, T+1) :- is_cooled(O, T), step(T+1).
is_cleaned(O, T+1) :- is_cleaned(O, T), step(T+1).
is_looked(O1, O2, T+1) :- is_looked(O1, O2, T), step(T+1).
is_open(L, T+1) :- is_open(L, T), not action(close(L), T), step(T+1).
-is_open(L, T+1) :- -is_open(L, T), not action(open(L), T), step(T+1).

% Basic Action effects
is_cooled(O, T+1) :- action(cool(O, L), T), holding(O, T), robot_at(L, T).
is_heated(O, T+1) :- action(heat(O, L), T), holding(O, T), robot_at(L, T).
is_cleaned(O, T+1) :- action(clean(O, L), T), holding(O, T), robot_at(L, T).
is_looked(O1, O2, T+1) :- action(use(O2), T), holding(O1, T), robot_at(L, T), at(O2,L,T).

is_open(L, T+1) :- action(open(L), T), openable(L).
at(O, L, T+1) :- action(put_down(O, L), T).
-holding(O, T+1) :- action(put_down(O, L), T).
at(O, L, T+1) :- action(put_down(O, L), T).

% Basic Action constraints
:- holding(O, T), not holding(O, T-1), not action(pick_up(O, _), T-1).
:- at(O, L1, T), at(O, L2, T+1), L1 != L2, not action(pick_up(O, L1), T).
:- is_heated(O, T+1), not is_heated(O, T), not action(heat(O, _), T).
:- is_cooled(O, T+1), not is_cooled(O, T), not action(cool(O, _), T).
:- is_cleaned(O, T+1), not is_cleaned(O, T), not action(clean(O, _), T).
:- holding(O, T), at(O, _, T).
:- robot_at(L1, T), robot_at(L2, T+1), L1 != L2, not action(go_to(L2), T).
:- is_open(L, T+1), not is_open(L, T), not action(open(L), T).
:- is_looked(O1, O2, T+1), not is_looked(O1, O2, T), not action(use(O2), T).
:- is_looked(O1, _, T+1), not holding(O1, T).

% Goal Constraints
goal(T) :- goal(T-1), step(T), T > 0.
:- not goal(max_steps).

% Minimize the number of actions and steps to reach the goal
#minimize { 1@2,T : action(_, T) }.
#minimize { T@1 : goal(T), not goal(T-1) }.

%######## Action Rreconditions from Domain Adaptation #########
:- action(put_down(O, L), T), openable(L), not is_open(L, T).
:- action(pick_up(O, L), T), openable(L), not is_open(L, T).
robot_at(L, T+1) :- robot_at(L, T), action(go_to(L), T), step(T+1).
holding(O, T+1) :- action(pick_up(O, L), T), robot_at(L, T).
holding(O, T+1) :- holding(O, T), not action(put_down(O, _), T), step(T+1).
:- action(clean(O, L), T), not holding(O, T).
:- action(cool(O, L), T), not holding(O, T).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial state
location(middle_of_room).
robot_at(middle_of_room, 0).
#show action/2.
'''