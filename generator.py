from dummy import dummy_material_net
from Environments import ChessEnvironment
import random
from mcts import mcts_setup, mcts_search_classic, calc_improved_policy
import time



# Generate games for the network to learn

# First mover has mate in 7
START_POS = "r3k2r/8/8/8/8/8/8/R2QK2R {} KQkq - 0 1"
COLOUR = ['w', 'b']

if __name__ == "__main__":
    # model_name = sys.argv[1]
    # n_games = int(sys.argv[2])
    model_name = 'PLACEHOLDER'
    n_games = 1
    print('INFO: Generating {} games using {}'.format(n_games, model_name))

    game_hist = {'STATES': [],
                 'POLICY':[],
                 'RESULT':None,
                 'ACTIONS': []
    }

    for game_no in n_games:
        position = START_POS.format(random.choice(COLOUR))
        board = ChessEnvironment(position)
        while not board.is_game_over() or board.can_calim_draw():
            root_node, master_node = mcts_setup(board)
            mcts_pocliy = mcts_search_classic(root_node, dummy_material_net)
            target_policy = calc_improved_policy(root_node)






    # Generate a game
    random.seed(0)
    position = START_POS.format(random.choice(COLOUR))
    board = ChessEnvironment(position)
    while not board.is_game_over() or board.can_calim_draw():
        tic = time.perf_counter()
        root_node, master_node = mcts_setup(board)
        mcts_pocliy = mcts_search_classic(root_node, dummy_material_net)
        print(mcts_pocliy)
        toc = time.perf_counter() - tic
        print('Search Performance:', toc)
        break
