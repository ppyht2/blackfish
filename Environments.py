import numpy as np
import chess
from constants import BOARD_SIZE, DEFAULT_PIECE_ORDER, N_FEATURE, N_BOARD_FEATURE


def get_binary_plane(boolean):
    return np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.bool) * boolean


def get_piece_plane(idx):
    plane = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.bool)
    plane[idx] = True
    plane = plane.reshape((BOARD_SIZE, BOARD_SIZE))
    return plane


def init_state():
    return np.zeros((BOARD_SIZE, BOARD_SIZE, N_FEATURE), dtype=np.bool), dict()


class ChessEnvironment(chess.Board):
    def __init__(self, start=chess.STARTING_FEN):
        chess.Board.__init__(self, start)
        self.state, self.rep_counter = init_state()
        self._update_state()

    def act(self, move, string_move=True):
        if string_move:
            self.push_uci(move)
        else:
            self.push(move)
        self._update_state()

    def _update_state(self):

        player = self.turn
        enemy = not player

        # Part 1: Piece features
        board_state = []
        for colour in (player, enemy):
            for piece in DEFAULT_PIECE_ORDER:
                piece_idx = list(self.pieces(piece, colour))
                piece_plane = get_piece_plane(piece_idx)
                board_state.append(piece_plane)

        # Increment state counter
        self.rep_counter[str(self)] = self.rep_counter.get(str(self), 0) + 1
        rep = self.rep_counter[str(self)]

        # # Part 2: State repretions
        for i in range(2):
            rep_plane = get_binary_plane(rep > i)
            board_state.append(rep_plane)

        # Part 3: Meta features
        colour_plane = get_binary_plane(player)
        p1_king_castling = get_binary_plane(self.has_kingside_castling_rights(player))
        p1_queen_castling = get_binary_plane(self.has_queenside_castling_rights(player))
        p2_king_castling = get_binary_plane(self.has_kingside_castling_rights(enemy))
        p2_queen_castling = get_binary_plane(self.has_queenside_castling_rights(enemy))

        meta_state = [colour_plane, p1_king_castling,
                      p1_queen_castling, p2_king_castling, p2_queen_castling]

        # roll previouse state and update
        self.state = np.roll(self.state, -N_BOARD_FEATURE, axis=2)
        update_state = np.stack(board_state + meta_state, axis=-1)
        # TODO: HARD CODED
        self.state[:, :, -19:] = update_state


if __name__ == "__main__":

    n = 800

    import time
    import random

    random.seed(42)

    print('Measure performance with {} state updates. . . '.format(n))

    env = ChessEnvironment()
    i = 0
    tic = time.perf_counter()
    while i < n:
        if env.is_game_over(claim_draw=True):
            env = ChessEnvironment()
        m = random.choice(list(env.legal_moves))
        env.act(m, False)
        i += 1
    toc = time.perf_counter() - tic
    print(toc)
