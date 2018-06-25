import chess
import random

# Player is always white

# Setup
board = chess.Board()

while not board.is_game_over():

    # Player
    user_move = input('Your Move: ')
    # TODO assume move is real
    board.push_san(user_move)

    print(board)

    print(' -------- ')

    # Bot
    legal_moves = [m for m in board.legal_moves]
    bot_move = random.choice(legal_moves)
    board.push_uci(bot_move.uci())

    print(board)


print('GAME OVER!')
