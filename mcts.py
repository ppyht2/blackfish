from Environments import ChessEnvironment
from collections import defaultdict
from constants import N_ACTIONS
from dummy import generate_action_dict
import numpy as np
import time

from threading import Thread

# TODO: Tidy this up
a2m, m2a = generate_action_dict()

C_PUCT = 10


class MasterNode():
    """ A placeholder node the root node statistics
    """

    def __init__(self):
        self.parent = None
        self.child_n = defaultdict(float)
        self.child_w = defaultdict(float)


class MctsNode():
    """ MCTS Node class

    Args:
      f_action: edge connecting node to this parent
      parent: the parent node
      env: a chess environment
    """

    def __init__(self, f_action, parent, env=None, is_root_node=False):
        self.parent = parent

        self.child_n = np.zeros(N_ACTIONS, dtype=np.int)
        self.child_w = np.zeros(N_ACTIONS, dtype=np.float32)

        self.f_action = f_action
        self.is_expanded = False

        self.is_root_node = is_root_node

        self.locked = False

        # Root Node always copy the environment
        if self.is_root_node:
            self.env = env.copy()

    def select(self):
        """ Select a child node
        """
        Q = -self.child_w / (self.child_n + 1)
        # TODO: Optimise
        U = C_PUCT * self.child_p * (np.sqrt(self.parent.child_n[self.f_action]) / (self.child_n + 1))
        L = - 999 * self.illegal_actions  # Illegal actions are punished
        score = Q + U + L
        a = np.argmax(score)
        return a

    def prep_env(self):
        # Prep environment for evaluation
        if not self.is_root_node:
            self.env = self.parent.env.copy()
            self.env.act(a2m[self.f_action])

    def expand(self, child_prior):
        assert not self.is_expanded
        # only expand legal nodes
        legal_actions = get_legal_actions(self.env)
        self.illegal_actions = np.invert(legal_actions)
        self.child = {action: MctsNode(action, self)
                      for action, islegal in enumerate(legal_actions) if islegal}

        # mask prior by legal moves and normalise
        self.child_p = child_prior * legal_actions
        self.child_p = self.child_p * (1 / self.child_p.sum())
        self.is_expanded = True

    def backup(self, value):
        self.parent.child_n[self.f_action] += 1
        self.parent.child_w[self.f_action] += value
        if not self.is_root_node:
            self.parent.backup(-value)  # backup in the perspective of the parent

    def _apply_loss(self, loss):
        self.parent.child_n[self.f_action] += loss
        if not self.is_root_node:
            self.parent.backup(-loss)

    def lock(self):
        self.locked = True
        # virtual loss
        self._apply_loss(-1)

    def unlock(self):
        self._apply_loss(1)
        self.locked = False

    @property
    def child_q(self):
        return self.child_w / (self.child_n + 1)

    @property
    def _child_u(self):
        return C_PUCT * self.child_p * (np.sqrt(self.parent.child_n[self.f_action]) / (self.child_n + 1))


def get_legal_actions(env):
    legal_actions = [m2a[m.uci()] for m in env.legal_moves]
    action_array = np.zeros(N_ACTIONS, dtype=np.bool)
    action_array[legal_actions] = True
    return action_array


def lap_timer(last_tic):
    now = time.perf_counter()
    lap = now - last_tic
    return lap, now


lock_count = 0


class MctsThread(Thread):
    def __init__(self, root_node, eval_fn):
        Thread.__init__(self)
        self.root_node = root_node
        self.eval_fn = eval_fn

    def run(self):
        current_node = root_node
        # 1. Select
        while current_node.is_expanded:
            a = current_node.select()
            current_node = current_node.child[a]

        # abandon if locked
        if current_node.locked:
            # print('LOCKED!')
            global lock_count
            lock_count += 1
            return

        # 2. Evaluate
        current_node.lock()
        current_node.prep_env()
        value, priors = self.eval_fn(current_node.env)

        # 3. Expand
        current_node.expand(priors)

        current_node.unlock()
        current_node.backup(value)


def mcts_search_new(root_node, eval_fn, n_sim=800):
    statistics = {'select_op': 0,
                  'max_depth': 0,
                  'select_time': 0,
                  'evaluate_time': 0,
                  'expand_time': 0,
                  'backup_time': 0
                  }

    threads = []
    while root_node.parent.child_n[None] < n_sim:
        t = MctsThread(root_node, eval_fn)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return statistics


# Old search
def mcts_search_classic(root_node, eval_fn, n_sim=800):
    statistics = {'select_op': 0,
                  'max_depth': 0,
                  'select_time': 0,
                  'evaluate_time': 0,
                  'expand_time': 0,
                  'backup_time': 0
                  }

    for n in range(n_sim):
        # print('Search:', n)
        current_node = root_node
        depth = 0
        # select
        tic = time.perf_counter()
        while current_node.is_expanded:
            # print('Select')
            a = current_node.select()
            depth += 1
            current_node = current_node.child[a]
            statistics['select_op'] += 1
        statistics['max_depth'] = max(depth, statistics['max_depth'])
        statistics['select_time'] += time.perf_counter() - tic

        # expand, evaluate, and backup
        # print('_Prep')
        tic = time.perf_counter()
        current_node.prep_env()
        value, priors = eval_fn(current_node.env)
        statistics['evaluate_time'] += time.perf_counter() - tic

        tic = time.perf_counter()
        # print('Expand\n', current_node.env)
        current_node.expand(priors)
        statistics['expand_time'] += time.perf_counter() - tic

        tic = time.perf_counter()
        # print('Backup:', value)
        current_node.backup(value)
        statistics['backup_time'] += time.perf_counter() - tic

    return statistics

# Helper function
def mcts_setup(env):
    master_node = MasterNode()
    root_node = MctsNode(f_action=None, parent=master_node, env=env, is_root_node=True)
    return root_node, master_node


def calc_improved_policy(root_node):
    # only basic calculation for now
    total = root_node.parent[None]
    return root_node.child_n / total


if __name__ == "__main__":

    from dummy import dummy_material_net

    N_SIM = 1200

    board = ChessEnvironment("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    root_node, master_node = mcts_setup(board)

    tic = time.perf_counter()
    stats = mcts_search_classic(root_node, dummy_material_net, N_SIM)
    # stats = mcts_search_new(root_node, dummy_material_net, N_SIM)
    toc = time.perf_counter() - tic

    print('INFO: Search statistics')
    for k, v in stats.items():
        print(k, v)
    print('INFO: Main loop time {:.2f}s'.format(toc))

    print('Total visits based on rt', root_node.child_n.sum())
    print('Master:', master_node.child_n[None])
    pi = root_node.child_n / master_node.child_n[None]
    p = np.argsort(pi)[::-1]
    from IPython import embed

    for i in range(10):
        # embed()
        action = p[i]
        prob = pi[action]
        move = a2m[action]
        print(i, move, prob)

    for m in ('a1a8', 'h1h8'):
        a = m2a[m]
        print(m, a)
        print('N', root_node.child_n[a])
        print('W', root_node.child_w[a])
        print('q', root_node.child_q[a])
        print('p', root_node.child_p[a])

        print('u', root_node._child_u[a])

        print('s', root_node._child_u[a] - root_node.child_q[a])
    # embed()
    print('locked', lock_count)
