import numpy as np
import time
from chessHelper import MyBoard
from helper import generate_action_dict
from constants import N_ACTIONS

from collections import defaultdict


a2m, m2a = generate_action_dict()


class MasterNode():
    def __init__(self):
        self.parent = None
        self.child_n = defaultdict(float)
        self.child_w = defaultdict(float)


class MctsNode():
    def __init__(self, f_action, parent, env=None):
        self.parent = parent

        self.child_n = np.zeros(N_ACTIONS, dtype=np.int)
        self.child_w = np.zeros(N_ACTIONS, dtype=np.float32)

        self.f_action = f_action
        self.is_expanded = False

        # Root Node always copy the environment
        if isinstance(self.parent, MasterNode):
            self.env = env.copy()

    def select(self):
        # Do not select illegal actions
        # TODO: Cput
        Q = -self.child_w / (self.child_n + 1)
        # TODO: Optimise
        U = self.child_p * (np.sqrt(self.parent.child_n[self.f_action]) / (self.child_n + 1))

        L = - 999 * self.illegal_actions
        score = Q + U + L
        a = np.argmax(score)
        return a

    def prep_env(self):
        # Prep evnironment for evalulation
        if not isinstance(self.parent, MasterNode):
            self.env = self.parent.env.copy()
            self.env.push_uci(a2m[self.f_action])

    def expand(self, child_prior):
        assert not self.is_expanded
        # Do not expand illegal node
        legal_actions = get_legal_actions(self.env)
        self.illegal_actions = np.invert(legal_actions)
        self.child = {action: MctsNode(action, self)
                      for action, islegal in enumerate(legal_actions) if islegal}

        self.child_p = child_prior
        self.is_expanded = True

    def backup(self, value):
        self.parent.child_n[self.f_action] += 1
        self.parent.child_w[self.f_action] += value
        if not isinstance(self.parent, MasterNode):
            # In the prespective of the parent
            self.parent.backup(-value)


def get_legal_actions(env):
    legal_actions = [m2a[m.uci()] for m in env.legal_moves]
    action_array = np.zeros(N_ACTIONS, dtype=np.bool)
    action_array[legal_actions] = True
    return action_array


# Old search
def mcts_search(root_node, eval_fn, n_sim=800):
    n_select = 0
    max_depth = 0

    for n in range(n_sim):
        # print('Search:', n)
        current_node = root_node
        depth = 0
        # select
        while current_node.is_expanded:
            # print('Select')
            a = current_node.select()
            depth += 1
            current_node = current_node.child[a]
            n_select += 1
        max_depth = max(depth, max_depth)

        # expand, evaluate, and backup
        # print('_Prep')
        current_node.prep_env()
        value, priors = eval_fn(current_node.env)
        # print('Expand\n', current_node.env)
        current_node.expand(priors)
        # print('Backup:', value)
        current_node.backup(value)

    # TODO: caluclate improved policy
    return (max_depth, n_select)


if __name__ == "__main__":

    from dummy import dummy_material_net
    # board = MyBoard("r1b1r2k/pp2qp2/5n2/4n1p1/2P5/2Q1PNB1/P1B3PP/4RRK1 b - - 0 24")
    board = MyBoard()
    master_node = MasterNode()
    root_node = MctsNode(f_action=None, parent=master_node, env=board)
    tic = time.perf_counter()
    meta = mcts_search(root_node, dummy_material_net, 800)
    print(meta)
    toc = time.perf_counter() - tic
    print(toc)

    print('Total visits based on rt', root_node.child_n.sum())
    print('Master:', master_node.child_n[None])
    pi = root_node.child_n / master_node.child_n[None]
    p = np.argsort(pi)[::-1]
    for i in range(10):
        action = p[i]
        prob = pi[action]
        move = a2m[action]
        print(i, move, prob)
