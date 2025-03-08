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


def add(a: Move, b: Direction) -> Move:
    return (a[0] + b[0], a[1] + b[1])


class MyPlayer:
    """Template Docstring for MyPlayer, look at the TODOs"""

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
        return (0, 0)

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
