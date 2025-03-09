import random, copy

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
DEPTH_OF_SEARCH = 10


def add(a: Move, b: Direction) -> Move:
    return (a[0] + b[0], a[1] + b[1])


class MyPlayer:
    """Template Docstring for MyPlayer, look at the TODOs"""
    # Nic z toho neurobim meheheh

    # TODO replace docstring with a short description of your player

    def __init__(
        self, my_color: PlayerColor, opponent_color: PlayerColor, board_size: int = 8
    ):
        self.name = "username"  # TODO: fill in your username
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size

    def select_move(self, board: BoardState) -> Move:
        # TODO: write you method
        # you can implement auxiliary functions/methods, of course
        print("KHHHKT:", self.get_all_valid_moves(board))
        minimax = self.minimax(board, DEPTH_OF_SEARCH, True, self.my_color)
        print("MiniMax:", minimax)
        return minimax[1]
    
    def minimax(self, board: BoardState, depth: int, maximazing_player: bool, player_color: PlayerColor):
        valid_moves = self.get_all_valid_moves(board)

        if depth == 0 or not valid_moves:
            return [self.evaluate_board(board, player_color),(0,0)] # Return static evaluation
        
        if maximazing_player:
            max_eval = -float("inf")
            #board_copy = [row[:] for row in board]

            for move in valid_moves:
                flipped = self.make_move(board, player_color, move)
                eval = self.minimax(board, depth - 1, False, 1 - player_color)[0]
                self.undo_move(board, player_color, move, flipped)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            
            return [max_eval, best_move]
        
        else:
            min_eval = float("inf")
            #board_copy = [row[:] for row in board]
            for move in valid_moves:
                flipped = self.make_move(board, 1 - player_color, move)
                eval = self.minimax(board, depth - 1, True, 1 - player_color)[0]
                self.undo_move(board, 1 - player_color, move, flipped)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move

            return [min_eval, best_move]
        
    def evaluate_board(self, board: BoardState, my_color: PlayerColor) -> int:
        my_stones = sum(row.count(my_color) for row in board)
        opponent_stones = sum(row.count(my_color - 1) for row in board)
        return my_stones - opponent_stones 
    
    def make_move(self, board: BoardState, color: PlayerColor, move: Move):
        flipped_pos = []
        board[move[0]][move[1]] = color
        for step in DIRECTIONS:
            if self.__stones_flipped_in_direction(move, step, board) > 0:
                self.__flip_stones(move, board, step, color, flipped_pos)
        return flipped_pos

    def undo_move(self, board: BoardState, color: PlayerColor, move: Move, flipped_pos):
        board[move[0]][move[1]] = EMPTY_COLOR
        opponent_color = 1 - color
        for pos in flipped_pos:
            board[pos[0]][pos[1]] = opponent_color

    def __flip_stones(self, move: Move, board: BoardState, step: Direction, color: PlayerColor, flipped):
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
