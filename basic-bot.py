import chess
import numpy as np
from helper import get_basic_position_value
from mcts import MctsNode, mcts

board = chess.Board()


def value_fn(node):
    return get_basic_position_value(node.state)


def policy_fn(legal_moves):
    n = len(legal_moves)
    policy = np.ones(n)
    policy = policy * 1 / n
    return policy


while not board.is_game_over():
    player_move = input('Player Move: ')
    board.push_uci(player_move)

    root_node = MctsNode(a=None, p=None, parent=None, root_state=board)
    print('root node value:', value_fn(root_node))

    mcts(root_node, value_fn, policy_fn, 3200)

    max_score = 0
    best_action = None
    for a, c in root_node.children.items():
        score = c.n / root_node.n
        if score > max_score:
            max_score = score
            best_action = a

    print('Enemy Move: {} @ {:.2f}%'.format(best_action, max_score * 100))
    board.push_uci(best_action)
    print(board)
