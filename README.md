# **Linear Programming**

Linear programming is a method used to find and achieve the best outcome (maximum utility) in a mathematical model. The constraints of such models are represented by linear relationships.

## Construction of "A" matrix

**Overview**

The A matrix is a 2-D matrix that houses all (STATE, ACTION) pairs which represent the expected number of times an action is taken in a particular state.

Each column of the matrix represents a state-action pair and the rows represent all states of the MDP. Hence the values in the matrix at index (i, j) corresponds to a state that on taking an action (which corresponds to column j) reaches state I. The dimensions of the matrix are 600 x 1936, where 600 is the total number of states:
$(Positions * Arrows*Materials*MMstates*Healths)$

and 1936 is all the actions from each state.

The transition probabilities of the (STATE, ACTION) pairs contribute to the value of the corresponding matrix entries.

**Construction:**

To generate the matrix, first we get the actions from a state S and their probabilities. 

For every action possible from that state, the transition probability is added to the notation of the pair in the 'A' matrix and the probability of getting state S using the action A from the current state is subtracted from the pair value. This can be understood as adding the 'incoming' probabilities and subtracting the 'outgoing' probabilities. The action of 'NOOP' where no action is possible from that state is denoted by $None$. The matrix is then filled with the following logic:

- For non-terminal states, the values of the matrix at (state s, action a) represent the negated probability of transition from state s to state s' on taking action and the absolute value at position (state s', action a)
- For the other non-terminal states having the same start and end states, the corresponding element is set to 1.
- For the terminal states i.e where no action is possible (NOOP),the matrix value is set to 1 for these elements

```python
# GENERATING A
    shape = (NUM_STATES,dims)
    A = np.zeros(shape, dtype=np.float64)
    num = 0
    for i in range(0,NUM_STATES):
        state = State.node(i)
        actions = state.actions()
        if actions is not None:
            for action, results in actions.items():
                node_action.append(action)
                for vals in results:
                    A[i][num] += vals[0]
                    state_nxt = vals[1]
                    A[State(*state_nxt).state_id()][num] -= vals[0]
                num += 1
        else:
            act = "NONE"
            node_action.append(act)
            A[i][num] += 1
            num += 1
            continue
```

## Finding the policy

We solve the LP problem by maximising our total reward ($Rx$), given the following constraints:

$$Ax = \alpha
$$

$$x >= 0$$

The values are stored in $X$ which denote the number of times an action is taking from a particular state. We use these values to find the best action to take from that state and add to it our optimal policy. This process is repeated for each state until we get the optimal policy for each state.

```python
for i in range(NUM_STATES):
        state = list(State.node(i).get_state())
        actions = State.node(i).actions()
        state[0] = IJ_POS[state[0]]
        state[3] = MM_STATES_STRING[state[3]]
        state[4] *= 25
        if actions is None: 
            POLICY.append([state, "NONE"])
            idx += 1
            continue
        ck = len(actions.keys())
        act_idx = np.argmax(X[idx:idx+ck])
        idx += ck
        best_action = list(actions.keys())[act_idx]
        POLICY.append([state, best_action])
```

## Analysis

Given below are a few observations from the policy generated by the linear programming algorithm:

- On comparing the policy with earlier results generated by value iteration, a few disparities could be noticed.
    - There were now instances where IJ would use his blade from the central square which was not seen in the previous part.
    - IJ has become more risk-averse. This is supported by the fact that the frequency of choosing **SHOOT**  by the agent has increased.
    - The agent now could be seen going explicitly to the north in order to craft arrows
    - IJ now tries to safeguard itself. This could be seen using his arrows from the western block where he is safe from any attacks conducted by MM.
- IJ still prefers to use his blade first in the eastern box. When MM's health decreases it moves on to finsh his arrows.
- Due to the STEP COST of -10, IJ uses his **STAY** Option only in South, North, and in Western Square to escape from the ready state of MM or when MM's health is low to prevent a successful attack and subsequently increasing its health
- The level of foresight of the agent has been decreased as The agent is also seen using the **STAY** option in SOUTH square instead of a much fruitful **GATHER** in certain cases even when agents materials are low

The agent IJ is now seen to have an overall tendency towards survival rather than attacking as the final reward obtained by killing MM is absent.

## Generating other policies

Although for a given matrices A, R, and $\alpha$ there cannot be multiple policies since the algorithm is deterministic, depending on the matrices A, R, and $\alpha$, we can have multiple optimal policies. Some of the ways of generating them are:

1. Changing the order (preference) of the actions for each state:
    - In cases where multiple actions have same probability, the current program takes the action given at the top (indexed first) using np.argmax.
    - If the order of actions are changed or if the process of choosing the action from multiple options is changed (eg. randomly choosing an action from the best actions), we will generate a different A matrix due to different index values and thus a different policy
2. In some cases, the order of actions in a policy don't matter as they end up in same state. For such cases, we can have the actions be interchanged with each other by adding a very small penalty to one of the actions. This will change R while A and $\alpha$  remain same. On solving the LP, it will give different policy which will still be optimal
3. Changing the step cost will change R and thus our final objective obtained will be different based on a different policy generated accordingly
4. Changing the initial probability distribution $\alpha$ will change the optimality constraint and will result in a different policy that is optimal according to the new $\alpha$ value
5. Giving more variation to the starting state:
Instead of the current fixed starting state (C,2,3,R,100), if we randomly choose a starting state, it will give rise to variation in $\alpha$, and will thus give a different policy
6. Changing the transition probabilities will change the A matrix as the transition probabilities determine the values in the matrix. This will change the constraint value and a different policy according to the new utilities