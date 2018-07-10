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
            # I believe this is the most expensive operation
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
        self.n_root = 0

    # native sort ~ 7 seconds
    def select(self):
        scores = []
        for a, node in self.children.items():
            # - for value in the prespective of the current player
            s = -node.q + node.p * self.n_root / (1 + node.n)
            scores.append((a, s))

        best = max(scores, key=lambda x: x[1])
        return best[0]

    def expand(self, legal_moves, priors):
        for a, p in zip(legal_moves, priors):
            self.children[a] = MctsNode(a, p, parent=self)
        # board is no longer needed
        self.board = None

    def backup(self, v):
        # Increment
        self.n += 1
        self.w += v
        # Update
        self.q = self.w / self.n
        self.n_root = self.n**0.5
        # backup
        if self.parent is not None:
            self.parent.backup(-v)

    def __getitem__(self, arg):
        return self.children[arg]

    def __repr__(self):
        text = "A: {} N: {} W: {:.2f} Q: {:.2f} P: {:.2f}".format(
            self.a, self.n, self.w, self.q, self.p)
        return text


def create_root_node(board):
    return MctsNode(a=None, p=None, parent=None, root_board=board)


def mcts_search(root_node, eval_fn, n_sim=800):
    # TODO: add docs
    # TODO: added debug and logs

    n_select = 0
    n_expand = 0
    max_depth = 0

    for n in range(n_sim):
        current_node = root_node
        depth = 0

        # select
        while len(current_node.children) != 0:
            a = current_node.select()
            depth += 1
            current_node = current_node.children[a]
            n_select += 1

        # expand, evaluate, and backup
        legal_moves = [m.uci() for m in current_node.board.legal_moves]
        value, all_priors = eval_fn([current_node.board.state])
        all_priors = np.squeeze(all_priors)
        legal_actions = [m2a[m] for m in legal_moves]
        legal_priors = [all_priors[actions] for actions in legal_actions]
        current_node.expand(legal_moves, legal_priors)
        n_expand += len(legal_moves)
        current_node.backup(value)

    max_depth = max(depth, max_depth)

    # caluclate improved policy
    policy = []
    for a, node in root_node.children.items():
        policy.append((a, node.n / root_node.n))

    # sort it by probability
    policy.sort(key=lambda x: x[1], reverse=True)
    return policy, (max_depth, n_select, n_expand)


# TODO: Added unit tests

if __name__ == "__main__":
    from dummy import dummy_net
    from chessHelper import MyBoard
    import time
    np.random.seed(42)

    board = MyBoard()
    rt = create_root_node(board)

    tic = time.time()
    _, meta = mcts_search(rt, dummy_net)
    print(meta)
    toc = time.time() - tic
    print(toc)
