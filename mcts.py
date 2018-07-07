import numpy as np


class MctsNode():
    # TODO: add docs

    def __init__(self, a, p, parent, root_state=None):
        # root node has no parent
        self.parent = parent
        self.a = a
        if parent is None:
            self.state = root_state.copy()
        else:
            self.state = parent.state.copy()
            self.state.push_uci(self.a)

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


def mcts_search(root_node, value_fn, policy_fun, n_sim=7200):
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
        legal_moves = [m.uci() for m in current_node.state.legal_moves]
        current_node.expand(legal_moves, policy_fun(legal_moves))
        v = value_fn(current_node)
        current_node.evaluate(v)

    max_depth = max(depth, max_depth)
    print('Max Depth', max_depth)

    # caluclate improved policy
    policy = []
    for a, node in root_node.children.items():
        policy.append((a, node.n / root_node.n))

    # sort it by probability
    policy.sort(key=lambda x: x[1], reverse=True)
    return policy


# TODO: Added unit tests
