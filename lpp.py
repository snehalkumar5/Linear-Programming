import numpy as np
import cvxpy as cp
import json
import os

MDL_TEAM = 79
ARR = [0.5, 1, 2]
Y = ARR[MDL_TEAM % 3]
STEP_COST = -10 / Y

POSITIONS = {
    "WEST" : 0,
    "NORTH" : 1,
    "EAST" : 2,
    "SOUTH" : 3,
    "CENTER" : 4
}
IJ_POS = ['W', 'N', 'E', 'S', 'C']
max_hp = 5
max_num_arr = 3
max_num_mats = 2
max_states_mm = 2
MM_STATES_STRING = ['D', 'R']
STATES_MM = {
    'DORMANT':0,
    'READY':1
}
NUM_STATES = 600
MM_HIT_R = -40

MOVE_CHOOSE = np.array(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])
dormant={
    "STAY": 0.8,
    "GET_READY" : 0.2
}
ready={
    "ATTACK": 0.5,
    "STAY": 0.5
}
ACTIONS = {
    POSITIONS['CENTER']: {
        "UP": {
            "S": (0.85, POSITIONS['NORTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "DOWN": {
            "S": (0.85, POSITIONS['SOUTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "LEFT": {
            "S": (0.85, POSITIONS['WEST']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "RIGHT": {
            "S": (0.85, POSITIONS['EAST']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "STAY": {
            "S": (0.85, POSITIONS['CENTER']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "SHOOT": [0.5,0.5],
        "HIT": [0.1,0.9]
    },
    POSITIONS['NORTH']: {
        "DOWN": {
            "S": (0.85, POSITIONS['CENTER']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "STAY": {
            "S": (0.85, POSITIONS['NORTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "CRAFT": [0.5,0.35,0.15]
    },
    POSITIONS['SOUTH']: {
        "UP": {
            "S": (0.85, POSITIONS['CENTER']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "STAY": {
            "S": (0.85, POSITIONS['SOUTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "GATHER": {
            "S": 0.75,
            "F": 0.25
        }
    },
    POSITIONS['EAST']: {
        "STAY": {
            "S": (1, POSITIONS['EAST']),
            "F": (0, POSITIONS['EAST'])
        },
        "LEFT": {
            "S": (1, POSITIONS['CENTER']),
            "F": (0, POSITIONS['EAST'])
        },
        "SHOOT": [0.9,0.1],
        "HIT": [0.2,0.8]
    },
    POSITIONS['WEST']: {
        "STAY": {
            "S": (1, POSITIONS['WEST']),
            "F": (0, POSITIONS['EAST'])
        },
        "RIGHT": {
            "S": (1, POSITIONS['CENTER']),
            "F": (0, POSITIONS['EAST'])
        },
        "SHOOT": [0.25,0.75]
    }
}
class State:
    def __init__(self, position, materials, arrows, mm_state, mm_health, act = 1):
        self.pos = position
        self.active = act
        self.materials = materials
        self.pos_mm = 0
        self.arrows = arrows
        self.mm_state = mm_state
        self.mm_health = mm_health

    def dec_mm_hp(self, val):
        self.mm_health -= val
        self.mm_health = min(self.mm_health, max_hp-1)
        self.mm_health = max(0, self.mm_health)

    def inc_mm_hp(self, val):
        self.mm_health += val
        self.mm_health = min(self.mm_health, max_hp-1)
        self.mm_health = max(0, self.mm_health)

    def set_arrows(self, arrows):
        self.arrows = max(0, arrows)
        self.arrows = min(max_num_arr, self.arrows)

    def get_state(self):
        if self.active == 1:
            return self.pos, self.materials, self.arrows, self.mm_state, self.mm_health
        return None

    def set_mats(self, count):
        self.materials = max(0, count)
        self.materials = min(max_num_mats, self.materials)
    

    def actions(self):
        # print('hp',self.mm_health)
        new_action_map = {}
        if self.mm_health != 0:
            action_map = {}
            for action, resp in ACTIONS[self.pos].items():
                next_actions = []
                state_nxt = State(*self.get_state())
                if action in MOVE_CHOOSE:
                    # print(resp)
                    for _,(prob, next_pos) in resp.items():
                        state_nxt.pos = next_pos
                        val = state_nxt.get_state()   
                        next_actions.append((prob, val))
                    res = self.response_mm(action,next_actions)
                    action_map[action] = (res)

                elif action == "GATHER":
                    val = state_nxt.get_state()
                    next_actions.append((resp['F'], val))
                    state_nxt.set_mats(state_nxt.materials + 1)
                    val = state_nxt.get_state()
                    next_actions.append((resp['S'], val))
                    res = self.response_mm(action,next_actions)
                    action_map[action] = (res)

                elif action == "HIT":
                    if action not in action_map:
                        action_map[action] = []
                    val = state_nxt.get_state()
                    next_actions.append((resp[1], val))
                    state_nxt.dec_mm_hp(2)
                    val = state_nxt.get_state()
                    next_actions.append((resp[0], val))
                    res = self.response_mm(action,next_actions)
                    action_map[action] = (res)

                elif action == "SHOOT":
                    # print(resp)
                    if self.arrows != 0:
                        state_nxt.set_arrows(self.arrows - 1)
                        val = state_nxt.get_state()
                        next_actions.append((resp[1], val))
                        state_nxt.dec_mm_hp(1)
                        val = state_nxt.get_state()
                        next_actions.append((resp[0], val))
                        res = self.response_mm(action,next_actions)
                        action_map[action] = (res)
                    else:
                        continue

                elif action == "CRAFT":
                    if self.materials != 0:
                        state_nxt.set_mats(state_nxt.materials - 1)
                        # print(resp)
                        for idx,prob in enumerate(resp):
                            state_nxt.set_arrows(state_nxt.arrows+idx+1)
                            # print('S',idx+1,prob)
                            val = state_nxt.get_state()
                            next_actions.append((prob, val))
                            state_nxt.set_arrows(state_nxt.arrows-idx-1)

                    else:
                        continue
                    res = self.response_mm(action,next_actions)
                    action_map[action] = (res)

            for action in action_map.keys():
                take_act = []
                for resp in action_map[action]:
                    if resp[1] == self.get_state() or resp[0] == 0:
                        continue
                    else:
                        take_act.append(resp)
                if take_act:
                    new_action_map[action] = take_act
            return new_action_map
        else:
            return None

    # (p, state_tup) => [(p, state_nxt, got_hurt)]
    def response_mm(self, action, responses):
        final_responses = []

        # STATE IS READY
        if self.mm_state == 1:
            for p, state_nxt in responses:
                state_nxt = State(*state_nxt)
                final_responses.append((p * ready["STAY"], state_nxt.get_state(), 0))

                if self.pos != POSITIONS["EAST"] and self.pos != POSITIONS["CENTER"]:
                    state_nxt.mm_state = STATES_MM["DORMANT"]
                    final_responses.append((p * ready["ATTACK"], state_nxt.get_state(), 0))
         
            if self.pos == POSITIONS["EAST"] or self.pos == POSITIONS["CENTER"]:
                state_nxt = State(*self.get_state())
                state_nxt.set_arrows(0)
                state_nxt.inc_mm_hp(1)
                state_nxt.mm_state = STATES_MM["DORMANT"]
                final_responses.append((ready["ATTACK"], state_nxt.get_state(), 1))
        
        # STATE IS DORMANT
        elif self.mm_state == 0:
            for p, state_nxt in responses:
                state_nxt = State(*state_nxt)
                final_responses.append((p * dormant["STAY"], state_nxt.get_state(), 0))
                state_nxt.mm_state = STATES_MM["READY"]
                final_responses.append((p * dormant["GET_READY"], state_nxt.get_state(), 0))

        return final_responses
            
    def state_id(self):
        indx = 0
        indx += self.pos * ((max_num_mats + 1) * (max_num_arr + 1) * max_states_mm * (max_hp))
        indx += self.materials * ((max_num_arr + 1) * max_states_mm * (max_hp))
        indx += (self.arrows * (max_states_mm * (max_hp))) + (self.mm_state * (max_hp)) + self.mm_health
        return indx

    @classmethod
    def node(self, idx):
        
        ck = ((max_num_mats + 1) * (max_num_arr + 1) * max_states_mm * (max_hp))
        st_0 = idx // ck
        idx = idx % ck

        ck //= (max_num_mats + 1)
        st_1 = idx // ck
        idx = idx % ck

        ck //= (max_num_arr + 1)
        st_2 = idx // ck
        idx = idx % ck

        ck //= (max_states_mm)
        st_3 = idx // ck

        # ret_node = State(st_0, st_1, st_2, st_3, idx)
        return State(st_0, st_1, st_2, st_3, idx%ck)

if __name__ == "__main__":
    node_action = []
    POLICY = []
    OBJECTIVE = 0.0
    # INIT DIMS
    dims = 0
    for i in range(NUM_STATES):
        if State.node(i).actions() is None:
            dims = dims + 1
        else:
            ltgh = len(State.node(i).actions().keys())
            dims = dims + ltgh
    # INITIALISE R
    R = np.zeros((1, dims))
    cnt = 0
    for i in range(NUM_STATES):
        actions = State.node(i).actions()
        if actions is not None:
            for _, results in actions.items():
                R[0][cnt] += (STEP_COST)
                for result in results:
                    if result[2]: 
                        R[0][cnt] += (result[0] * MM_HIT_R)
                cnt += 1
        else:
            R[0][cnt] = 0
            cnt += 1
            continue
    
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
                # print(results)
                for vals in results:
                    # print('yellow',i,result[0])
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

    # INITIALISE ALPHA
    shape = (NUM_STATES,1)
    ALPHA = np.zeros(shape) 
    s1 = State(POSITIONS["CENTER"], 2, 3, STATES_MM["READY"], 4)
    ALPHA[s1.state_id()] = 1
    
    # INITIALISE X
    shape = (dims,1)
    temp = cp.Variable(shape, 'temp')
    constr = [cp.matmul(A, temp) == ALPHA, temp >= 0]
    objt = cp.Maximize(cp.matmul(R, temp))
    prb = cp.Problem(objt, constr)

    solution = prb.solve(verbose = True)
    OBJECTIVE = solution
    
    X = [float(i) for i in list(temp.value)]

    # OBTAIN POLICY
    idx = 0
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

    outputs = {
        "a" : A.tolist(),
        "r" : [float(i) for i in np.transpose(R)],
        "alpha" : [float(i) for i in ALPHA],
        "x" : X,
        "policy" : POLICY,
        "objective" : float(OBJECTIVE)
    }
    os.makedirs('outputs', exist_ok=True)
    ress = json.dumps(outputs, indent=2)
    with open("outputs/part_3_output.json", 'w+') as f:
        f.write(ress)
