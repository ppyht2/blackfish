import chess
import numpy as np
from constants import BOARD_SIZE

PIECE_VALUE = {chess.PAWN: 100,
               chess.KNIGHT: 320,
               chess.BISHOP: 330,
               chess.ROOK: 500,
               chess.QUEEN: 900,
               chess.KING: 20000
               }


def get_material_value(board, colour):
    """ Get material value with respect to the player

    Args:
      board: current board
      colour: colour of the player

    Returns:
      material_value: material value of the player
    """
    material_value = 0

    # Checkmate modifier
    if board.is_checkmate():
        result = int(board.result()[0])
        material_value += int(result == colour) * 20000

    # Material modifier
    for piece, piece_value in PIECE_VALUE.items():
        # Does not consider kings for now
        if piece == chess.KING:
            continue
        square_set = board.pieces(piece, colour)
        n_piece = len(square_set)
        material_value += n_piece * piece_value

    return material_value


def get_basic_position_value(board, normed=True):
    """ Basic position evalulation with respect to current player
        Only material is considered

    Args:
      board: board
      normal: If True, the position value is between -1 and 1. Absolute value is returned otherwise.

    Returns:
      position_value: position_value with repsect to the current player, higher is better.
    """

    # TODO: add positional evaluations
    # http://chessprogramming.wikispaces.com/Simplified+evaluation+function

    player = board.turn
    player_material_value = get_material_value(board, player)
    enemy_material_value = get_material_value(board, not player)

    if normed:
        position_value = (player_material_value - enemy_material_value) / 500
        position_value = np.tanh(position_value)
    else:
        position_value = player_material_value - enemy_material_value

    return position_value


def generate_action_dict():
    """ Generate a list of all possible uci moves and their action number lookup

    Return:
      actions: lookup dictionary for converting uci into actions(number)
      inverse_actions: lookup dictionary for converting actions to uni
    """
    # normal uci moves
    standard_uci = []
    char_start = ord('a')
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            for u in range(BOARD_SIZE):
                for v in range(BOARD_SIZE):
                    i_char = chr(char_start + i)
                    j_char = str(j + 1)
                    u_char = chr(char_start + u)
                    v_char = str(v + 1)
                    mv = ''.join([i_char, j_char, u_char, v_char])
                    standard_uci.append(mv)

    # pawn promotions
    promotion_uci = []
    for m in standard_uci:
        if (m[1] == '7' and m[3] == '8') or (m[1] == '2' and m[3] == '1'):
            if abs(ord(m[0]) - ord(m[2])) <= 1:
                for p in 'nqrb':
                    promotion_uci.append(m + p)

    complete_uci = standard_uci + promotion_uci
    complete_uci.sort()
    actions = dict()
    inverse_actions = dict()
    for u, a in enumerate(complete_uci):
        actions[u] = a
        inverse_actions[a] = u

    return actions, inverse_actions
