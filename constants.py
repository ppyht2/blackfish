import chess

BOARD_SIZE = 8
N_ACTIONS = 8 * 8 * 73

PIECE_NAMES = {chess.PAWN: 'PAWN',
               chess.KNIGHT: 'KNIGHT',
               chess.BISHOP: 'BISHOP',
               chess.ROOK: 'ROOK',
               chess.QUEEN: 'QUEEN',
               chess.KING: 'KING'
               }

DEFAULT_PIECE_ORDER = [i for i in range(1, 7)]
# This is the same as [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

COLOUR_NAMES = ['BLACK', 'WHITE']


N_BOARD_FEATURE = (6 + 6 + 2)
N_FEATURE = 8 * N_BOARD_FEATURE + (1 + 2 + 2)
