import chess
import numpy as np
from mcts import MctsNode, mcts_search_classic, MasterNode
from dummy import dummy_material_net, get_basic_position_value, generate_action_dict
import time


a2m, m2a = generate_action_dict()

board = chess.Board()

while not board.is_game_over():
    print('------------------------------')
    player_move = input('Player Move: ')
    board.push_uci(player_move)
    print('------------------------------')
    print(board)
    print('------------------------------')

    master = MasterNode()
    root = MctsNode(f_action=None, parent=master, env=board)

    tic = time.perf_counter()
    stat = mcts_search(root, dummy_material_net, 3200)
    toc = time.perf_counter() - tic
    print('MCTS Stat:{} {:.2f}s'.format(stat, toc))

    n = root.child_n
    q = root.child_w / (root.child_n + 1)
    p = root.child_p

    prob = root.child_n / master.child_n[None]
    pi = np.argsort(prob)[::-1]
    a = pi[0]
    move = a2m[a]
    print('------------------------------')
    print('Enemy Move: {} @ {:.2f}% N={} Q={}'.format(move, prob[a] * 100, n[a], q[a]))
    board.push_uci(move)
    print('------------------------------')
    print(board)
