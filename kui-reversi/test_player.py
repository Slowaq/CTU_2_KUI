from player import MyPlayer

color_from_char = {".": -1, "x": 0, "o": 1}


def str2board(board_str: str):
    return [[color_from_char[c] for c in row] for row in board_str.split() if row]


def test_noValidMove_inEmptyBoard():
    board = str2board("""
                      ...
                      ...
                      ...
                      """)
    player = MyPlayer(0, 1, len(board))
    assert player.get_all_valid_moves(board) == []


def test_noValidMove_withOpponentsStonesOnly():
    board = str2board("""
                      ...
                      .o.
                      ...
                      """)
    player = MyPlayer(0, 1, len(board))
    assert player.get_all_valid_moves(board) == []


def test_noValidMove_withMyStonesOnly():
    board = str2board("""
                      ...
                      .x.
                      ...
                      """)
    player = MyPlayer(0, 1, len(board))
    assert player.get_all_valid_moves(board) == []


def test_validMoves_myStonesAroundOpponent1():
    board = str2board("""
                      x..
                      xo.
                      xx.
                      """)
    player = MyPlayer(0, 1, len(board))
    assert set(player.get_all_valid_moves(board)) == set(
        [(0, 1), (0, 2), (1, 2), (2, 2)]
    )


def test_validMoves_myStonesAroundOpponent2():
    board = str2board("""
                      .xx
                      .ox
                      ..x
                      """)
    player = MyPlayer(0, 1, len(board))
    assert set(player.get_all_valid_moves(board)) == set(
        [(0, 0), (1, 0), (2, 0), (2, 1)]
    )


def test_noValidMove_withAllLinesFull():
    board = str2board("""
                      xoo
                      oo.
                      o.o
                      """)
    player = MyPlayer(0, 1, len(board))
    assert player.get_all_valid_moves(board) == []


def test_validMoves_forInitialConfig():
    board = str2board("""
                      ....
                      .xo.
                      .ox.
                      ....
                      """)
    player = MyPlayer(0, 1, len(board))
    assert set(player.get_all_valid_moves(board)) == set(
        [(1, 3), (3, 1), (0, 2), (2, 0)]
    )
