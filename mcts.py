import numpy as np
from helper import generate_action_dict

a2m, m2a = generate_action_dict()


class MctsNode():
    # TODO: add docs

    def __init__(self, a, p, parent, root_board=None):
        # root node has no parent
        self.parent = parent
        self.a = a
        if parent is None:
            self.board = root_board.copy()
        else:
            self.board = parent.board.copy()
            self.board.push_uci(self.a)

        self.children = dict()

        # node statistics
        self.n = 0
        self.w = 0
        self.p = p
        self.q = 0

    def select(self):
        actions = []
        scores = []
        for a, node in self.children.items():
            actions.append(a)
            # - for value in the prespective of the current player
            s = -node.q + node.p * np.sqrt(self.n) / (1 + node.n)
            scores.append(s)

        i = np.argmax(scores)
        a = actions[i]
        return a

    def expand(self, legal_moves, priors):
        for a, p in zip(legal_moves, priors):
            self.children[a] = MctsNode(a, p, parent=self)
        # board is no longer needed
        self.board = None

    def evaluate(self, v):
        self.n += 1
        self.w += v
        self.q = self.w / self.n
        # backup
        if self.parent is not None:
            self.parent.evaluate(-v)

    def __getitem__(self, arg):
        return self.children[arg]

    def __repr__(self):
        text = "A: {} N: {} W: {:.2f} Q: {:.2f} P: {:.2f}".format(
            self.a, self.n, self.w, self.q, self.p)
        return text


def mcts_search(root_node, eval_fn, n_sim=800):
    # TODO: add docs
    # TODO: added debug and logs

    max_depth = 0
    for n in range(n_sim):
        current_node = root_node
        depth = 0

        # select
        while len(current_node.children) != 0:
            a = current_node.select()
            depth += 1
            current_node = current_node.children[a]

        # expand, evaluate, and backup
        legal_moves = [m.uci() for m in current_node.board.legal_moves]
        value, all_priors = eval_fn([current_node.board.state])
        all_priors = np.squeeze(all_priors)
        legal_actions = [m2a[m] for m in legal_moves]
        legal_priors = [all_priors[actions] for actions in legal_actions]
        current_node.expand(legal_moves, legal_priors)
        current_node.evaluate(value)

    max_depth = max(depth, max_depth)

    # caluclate improved policy
    policy = []
    for a, node in root_node.children.items():
        policy.append((a, node.n / root_node.n))

    # sort it by probability
    policy.sort(key=lambda x: x[1], reverse=True)
    return policy, max_depth


# TODO: Added unit tests
