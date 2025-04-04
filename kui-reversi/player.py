import time
import numpy as np

type Move = tuple[int, int]
type Direction = tuple[int, int]
type PlayerColor = int
type BoardState = list[list[int]]

DIRECTIONS: list[Direction] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
EMPTY_COLOR = -1
TIMEOUT = 4.98


"""Base weights used for board evaluation"""
BASE_WEIGHTS_6 = np.array([
    [150, -50, 10, 10, -50, 150],
    [-50, -50, -2, -2, -50, -50],
    [ 10,  -2,  0,  0,  -2,  10],
    [ 10,  -2,  0,  0,  -2,  10],
    [-50, -50, -2, -2, -50, -50],
    [100, -50, 10, 10, -50, 100]
])

BASE_WEIGHTS_8 = np.array([
    [150, -50, 10,  5,  5, 10, -50, 150],
    [-50, -50, -2, -2, -2, -2, -50, -50],
    [ 10,  -2,  0,  0,  0,  0,  -2,  10],
    [  5,  -2,  0,  0,  0,  0,  -2,   5],
    [  5,  -2,  0,  0,  0,  0,  -2,   5],
    [ 10,  -2,  0,  0,  0,  0,  -2,  10],
    [-50, -50, -2, -2, -2, -2, -50, -50],
    [100, -50, 10,  5,  5, 10, -50, 100]
])

BASE_WEIGHTS_10 = np.array([
    [150, -50, 10,  5,  3,  3,  5, 10, -50, 150],
    [-50, -50, -2, -2, -1, -1, -2, -2, -50, -50],
    [ 10,  -2,  0,  0,  0,  0,  0,  0,  -2,  10],
    [  5,  -2,  0,  0,  0,  0,  0,  0,  -2,   5],
    [  3,  -1,  0,  0,  0,  0,  0,  0,  -1,   3],
    [  3,  -1,  0,  0,  0,  0,  0,  0,  -1,   3],
    [  5,  -2,  0,  0,  0,  0,  0,  0,  -2,   5],
    [ 10,  -2,  0,  0,  0,  0,  0,  0,  -2,  10],
    [-50, -50, -2, -2, -1, -1, -2, -2, -50, -50],
    [150, -50, 10,  5,  3,  3,  5, 10, -50, 150]
])


def add(a: Move, b: Direction) -> Move:
    return (a[0] + b[0], a[1] + b[1])


class MyPlayer:
    """The one who controls the corners, shall win the game."""

    def __init__(
        self, my_color: PlayerColor, opponent_color: PlayerColor, board_size: int = 8
    ):
        self.name = "Janko Mrkvicka"
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size

    def select_move(self, board: BoardState) -> Move:
        global start, best_move_found

        start = time.time()   # To prevent runtime error
        best_move_found = (0,0)
        alpha = -float("inf")
        beta = float("inf")

        minimax = self.minimax(board, 10, alpha, beta, True, self.my_color)

        return minimax[1]
    
    def minimax(self, board: BoardState, depth: int, alpha, beta, maximazing_player: bool, player_color: PlayerColor):
        """Minimax algorithm with alpha-beta pruning"""
        global best_move_found

        if time.time() - start >= TIMEOUT:  # Handle program running too long
            return [0, best_move_found]
        
        valid_moves = self.get_all_valid_moves(board)

        if depth == 0 or not valid_moves:
            return [self.evaluate_board(board, player_color),best_move_found] # Return static evaluation
        
        if maximazing_player:
            max_eval = -float("inf")

            # Sort moves according to their evaluation for more effective pruning
            for move in sorted(valid_moves, key=lambda m: self.evaluate_board(board, self.my_color), reverse=True):

                flipped = self.make_move(board, player_color, move) # Mark flipped stones for fast recovery
                eval = self.minimax(board, depth - 1, alpha, beta, False, 1 - player_color)[0]
                self.undo_move(board, player_color, move, flipped)
                
                # Pruning
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    best_move_found = move
                    alpha = eval
                if beta <= alpha:
                    break
            
            return [max_eval, best_move]
        
        else:
            min_eval = float("inf")

            for move in sorted(valid_moves, key=lambda m: self.evaluate_board(board, self.my_color), reverse=True):
                flipped = self.make_move(board, 1 - player_color, move)
                eval = self.minimax(board, depth - 1, alpha, beta, True, 1 - player_color)[0]
                self.undo_move(board, 1 - player_color, move, flipped)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                    best_move_found = move
                    beta = eval
                if beta <= alpha:
                    break

            return [min_eval, best_move]
        
    def board_weight(self, board: BoardState):
        """Weigh the board according to position of stones"""

        if self.board_size == 6:
            weights = BASE_WEIGHTS_6

        elif self.board_size == 8:
            weights = BASE_WEIGHTS_8

        elif self.board_size == 10:
            weights = BASE_WEIGHTS_10
        
        unique, counts = np.unique(board, return_counts=True)

        histogram = dict(zip(unique, counts))

        empty_cells = histogram.get(-1, 0)  

        # in early game priorititze mobility rather than position strength
        if empty_cells > (self.board_size) ** 2 - (self.board_size * (1/3)):
            return weights * 0.8

        if empty_cells > (self.board_size) ** 2 - (self.board_size * (2/3)):
            return weights * 1.2

        else:
            return weights * 2.0


    def penalty(self, board: BoardState, weight: BoardState) -> int:
        """Subtract points from position acording to opponent's stones strengths"""

        self.my_color, self.opponent_color = self.opponent_color, self.my_color # Swap colors to get opponent's valid moves
        opponent_moves = self.get_all_valid_moves(board)
        self.my_color, self.opponent_color = self.opponent_color, self.my_color # Swap back colors

        penalty = 0
        for move in opponent_moves:
            row, col = move
            penalty += weight[row][col]
        
        return penalty + 3 * len(opponent_moves)
    
    def evaluate_board(self, board: BoardState, player_color: PlayerColor) -> int:
        """Evaluate the board with an emphasis on mobility early in the game rather then stone count."""
        opponent_color = 1 - player_color

        # Count number of valid moves
        my_moves = len(self.get_all_valid_moves(board))
        opponent_moves = len(self.get_all_valid_moves(board))
        mobility_score = my_moves - opponent_moves  # More moves available is better

        # Count pieces
        my_pieces = sum(row.count(player_color) for row in board)
        opponent_pieces = sum(row.count(opponent_color) for row in board)
        piece_count_score = my_pieces - opponent_pieces

        # Use board weights to evaluate positional strength
        weight_matrix = self.board_weight(board)
        positional_score = np.sum(weight_matrix * (board == player_color)) - np.sum(weight_matrix * (board == opponent_color))

        # Adjust weights dynamically: prioritize mobility early on
        total_pieces = my_pieces + opponent_pieces
        max_pieces = self.board_size ** 2
        early_game_factor = 1 - (total_pieces / max_pieces)  # High early, low late

        evaluation_score = (
            3 * early_game_factor * mobility_score +  # More weight on mobility in early game
            1.5 * positional_score +  
            (1 - early_game_factor) * piece_count_score - # More weight on piece count in late game
            self.penalty(board, weight_matrix)
        )

        return evaluation_score

    def make_move(self, board: BoardState, color: PlayerColor, move: Move):
        """Function places a stone at move and returns all the flipped stones"""
        flipped_pos = []
        board[move[0]][move[1]] = color
        for step in DIRECTIONS:
            if self.__stones_flipped_in_direction(move, step, board) > 0:
                self.__flip_stones(move, board, step, color, flipped_pos)
        return flipped_pos

    def undo_move(self, board: BoardState, color: PlayerColor, move: Move, flipped_pos):
        """Function returs back flipped stones"""
        board[move[0]][move[1]] = EMPTY_COLOR
        opponent_color = 1 - color
        for pos in flipped_pos:
            board[pos[0]][pos[1]] = opponent_color

    def __flip_stones(self, move: Move, board: BoardState, step: Direction, color: PlayerColor, flipped):
        """Find all stones that are affected by placing a stone at move"""
        position = add(move, step)
        while self.__is_on_board(position) and self.__opponent_stone_at(position, board):
            board[position[0]][position[1]] = color
            flipped.append([position[0], position[1]])
            position = add(position, step)


    def get_all_valid_moves(self, board: BoardState) -> list[Move]:
        """Get all valid moves for the player"""
        valid_moves: list[Move] = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                pos = (r, c)
                if self.__is_empty(pos, board) and self.__is_correct_move(pos, board):
                    valid_moves.append(pos)
        return valid_moves

    def __is_empty(self, pos: Move, board: BoardState) -> bool:
        """Check if the position is empty"""
        return board[pos[0]][pos[1]] == EMPTY_COLOR

    def __is_correct_move(self, move: Move, board: BoardState) -> bool:
        for step in DIRECTIONS:
            if self.__stones_flipped_in_direction(move, step, board) > 0:
                return True
        return False

    def __stones_flipped_in_direction(
        self, move: Move, step: Direction, board: BoardState
    ) -> int:
        """Check how many stones would be flipped in a given direction"""
        flipped_stones = 0
        pos = add(move, step)
        while self.__is_on_board(pos) and self.__opponent_stone_at(pos, board):
            flipped_stones += 1
            pos = add(pos, step)
        if not self.__is_on_board(pos):
            # Oponent's stones go all the way to the edge of the game board
            return 0
        if not self.__my_stone_at(pos, board):
            # There is not my stone at the end of opponent's stones
            return 0
        return flipped_stones

    def __my_stone_at(self, pos: Move, board: BoardState) -> bool:
        """Check if the position is occupied by me"""
        return board[pos[0]][pos[1]] == self.my_color

    def __opponent_stone_at(self, pos: Move, board: BoardState) -> bool:
        """Check if the position is occupied by the opponent"""
        return board[pos[0]][pos[1]] == self.opponent_color

    def __is_on_board(self, pos: Move) -> bool:
        """Check if the position exists on the board"""
        return 0 <= pos[0] < self.board_size and 0 <= pos[1] < self.board_size
