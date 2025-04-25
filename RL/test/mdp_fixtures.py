import pytest
from kuimaze2 import MDPProblem, Map

@pytest.fixture
def mdp_G():
    map = "G"
    return MDPProblem(Map.from_string(map))

@pytest.fixture
def mdp_SG():
    map = "SG"
    return MDPProblem(Map.from_string(map))

@pytest.fixture
def mdp_S1G():
    map = "S.G"
    return MDPProblem(Map.from_string(map))

@pytest.fixture
def mdp_S2G():
    map = "S..G"
    return MDPProblem(Map.from_string(map))


@pytest.fixture
def mdp_SG_down():
    map = "S\nG"
    return MDPProblem(Map.from_string(map))

@pytest.fixture
def mdp_S1G_down():
    map = "S\n.\nG"
    return MDPProblem(Map.from_string(map))

@pytest.fixture
def mdp_S2G_down():
    map = "S\n.\n.\nG"
    return MDPProblem(Map.from_string(map))

@pytest.fixture
def mdp_3x3():
    map = "S..\n...\n..G"
    return MDPProblem(Map.from_string(map))

@pytest.fixture
def mdp_3x3_impossible():
    map = "S##\n#.#\n##G"
    return MDPProblem(Map.from_string(map))