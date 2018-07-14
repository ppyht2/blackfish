import sys
from Environments import ChessEnvironment
from net import Model
import random
from mcts import mcts_setup, mcts_search_classic
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

    model = Model(3)
    # Temp
    model.save()
    model.load()


    def tf_eval(env):
        return model.evaluate([env.state])


    # Generate a game
    position = START_POS.format(random.choice(COLOUR))
    board = ChessEnvironment(position)
    while not board.is_game_over() or board.can_calim_draw():
        tic = time.time()
        root_node, master_node = mcts_setup(board)
        mcts_pocliy = mcts_search_classic(root_node, tf_eval)
        # TODO: addd noise
        print(mcts_pocliy)
        toc = time.time() - tic
        print('Search Time:', toc)
        break
