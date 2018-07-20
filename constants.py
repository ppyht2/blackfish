import chess
import numpy as np

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

# Used in dummy
PIECE_VALUE = {chess.PAWN: 100,
               chess.KNIGHT: 320,
               chess.BISHOP: 330,
               chess.ROOK: 500,
               chess.QUEEN: 900,
               chess.KING: 20000
               }

UNIFORM_POLICY = np.ones(N_ACTIONS) / N_ACTIONS
