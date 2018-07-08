import sys
from chessHelper import MyBoard
from net import Model
import random
from mcts import mcts_search, MctsNode
import time
# Generate games for the network to learn

# First mover has mate in 7
START_POS = "r3k2r/8/8/8/8/8/8/R2QK2R {} KQkq - 0 1"
COLOUR = ['w', 'b']

if __name__ == "__main__":
    model_name = sys.argv[1]
    n_games = int(sys.argv[2])
    print('INFO: Generating {} games using {}'.format(n_games, model_name))

    model = Model(3)
    # Temp
    model.save()
    model.load()

    # Generate a game
    position = START_POS.format(random.choice(COLOUR))
    board = MyBoard(position)
    while not board.is_game_over() or board.can_calim_draw():
        tic = time.time()
        rt = MctsNode(a=None, p=None, parent=None, root_board=board)
        mcts_pocliy = mcts_search(rt, model.evaluate)
        # addd noise
        print(mcts_pocliy)
        toc = time.time() - tic
        print('Search Time:', toc)
        break
