# delayed_traces
Delayed traces to cope with delay in RL


## Setting

There may be 2 options:
  1. Delay affects only the rewards, i.e., consequences in terms of the action taken are immediately experienced by the agent (already from next state) but reward for present transition is given with some delay.
  2. Delay affects both actions and rewards: if the agents chooses an action at time $t$, the corresponding transition actually happens after some time $t+d$, with respect to the state at time $t+d$.


## Details about code

Delayed traces should work as the standard traces but being triggered after some time, corresponding to the delay. So the trace related with state-action pair $(s_t, a_t)$ is actually set to $1$ at time $t+d$. This is however implemented in a different way, which is hopefully equivalent. Namely the state and actions are collected in a buffer and the update is done after $d$ times, when the correct transition is observed. \
\
! This may be wrong. Do we want to update $(s_t, a_t)$ or $(s_{t+d}, a_{t+d})$? \
\
As a start, setting $1$ is considered. \
So far a simple grid world is implemented where an agent and a goal are randomly initialised. Actions corresponds to cardinal points plus a dummy action with which the agent stays. Reward is $0$ for each transition, except if the goal is met, in which case the reward is $10$. \
HPs are basically thrown at random. \
\
Surely the code is wrong somewhere as performance are strictly worse with the additional information of the delay.
