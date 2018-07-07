import numpy as np
from constants import BOARD_SIZE, DEFAULT_PIECE_ORDER, N_FEATURE, N_BOARD_FEATURE


# TODO: default feature planes are typed np.int, perhaps bool will improve efficency
def get_binary_plane(boolean):
    return np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.int) * boolean


def get_piece_plane(idx):
    plane = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int)
    plane[idx] = 1
    plane = plane.reshape((BOARD_SIZE, BOARD_SIZE))
    return plane


def init_state():
    return np.zeros((BOARD_SIZE, BOARD_SIZE, N_FEATURE), dtype=np.int), dict()


def get_new_planes(board, prev_state, rep_counter):
    player = board.turn
    enemy = not player

    board_state = []
    for colour in (player, enemy):
        for piece in DEFAULT_PIECE_ORDER:
            piece_idx = list(board.pieces(piece, colour))
            piece_plane = get_piece_plane(piece_idx)
            board_state.append(piece_plane)

    # Increment state counter
    rep_counter = dict()
    rep_counter[str(board)] = rep_counter.get(str(board), 0) + 1
    rep = rep_counter[str(board)]

    # Repetitions
    for i in range(2):
        rep_plane = get_binary_plane(rep > i)
        board_state.append(rep_plane)

    # transcant planes
    colour_plane = get_binary_plane(player)
    p1_king_castling = get_binary_plane(board.has_kingside_castling_rights(player))
    p1_queen_castling = get_binary_plane(board.has_queenside_castling_rights(player))
    p2_king_castling = get_binary_plane(board.has_kingside_castling_rights(enemy))
    p2_queen_castling = get_binary_plane(board.has_queenside_castling_rights(enemy))

    meta_state = [colour_plane, p1_king_castling,
                  p1_queen_castling, p2_king_castling, p2_queen_castling]

    # roll previouse states
    current_state = np.roll(prev_state, -N_BOARD_FEATURE, axis=2)
    update_state = np.stack(board_state + meta_state, axis=-1)
    # TODO: HARD CODED
    current_state[:, :, -19:] = update_state

    return current_state, rep_counter
