import sys
from chessHelper import MyBoard
from net import Model
import random
from mcts import mcts_search, MctsNode
# Generate games for the network to learn

# First move has mate in 7
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

    def value_fn(inputs):
        return model.evaluate_value(inputs)

    def policy_fn(inputs):
        return model.evaluate_policy(inputs)

    # Generate a game
    position = START_POS.format(random.choice(COLOUR))
    board = MyBoard(position)
    while not board.is_game_over() or board.can_calim_draw():
        rt = MctsNode(a=None, p=None, parent=None, root_state=board)
        mcts_pocliy = mcts_search(rt, value_fn, policy_fn)
        print(mcts_pocliy)
        break
