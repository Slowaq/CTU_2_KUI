import heapq
from kuimaze2 import SearchProblem, Map, State
from kuimaze2.map_image import map_from_image

def manhattan_distance(state: State, goal: State) -> int:
    """Compute the Manhattan distance between two states."""
    return abs(state.r - goal.r) + abs(state.c - goal.c)



class Agent:
    """Agent that finds the shortest path using the A* algorithm."""

    def __init__(self, environment: SearchProblem):
        self.environment = environment

    def find_path(self) -> list[State]:
        """Finding path using A* algorithm"""

        start_state = self.environment.get_start()
        goal_state = self.environment.get_goals()

        open_list = [] # Min-heap
        heapq.heappush(open_list, (0, id(start_state), start_state, [])) # (f, g,state id, state, path)

        visited = set()
        g_values = {start_state: 0}

        while open_list:
            f, ID, current, path = heapq.heappop(open_list) # ID is mainly for heap sort 

            if current in visited:
                continue

            visited.add(current)

            if current in goal_state:
                path.append(current) # Extend path
                self.environment.render(path=path, wait=True, use_keyboard=True)
                return path
            
            
            for action in self.environment.get_actions(current):
                next_state, cost = self.environment.get_transition_result(current, action)
                
                # Heuristic
                new_g = g_values[current] + cost
                new_h = min(manhattan_distance(next_state, goal) for goal in goal_state)
                new_f = new_g + new_h

                if new_g < g_values.get(next_state, float('inf')):
                    g_values[next_state] = new_g
                    f = new_f
                    heapq.heappush(open_list, (new_f, id(next_state), next_state, path + [current]))


                print(
                f"Transition: {current} -> {next_state}, cost of new state: {g_values[next_state]}"
                )
                # Visualize the state of the algorithm:
                # * Use cost values as texts in cells
                # * Use cost values as colors of cells
                # * Mark current and new states
                # * Wait for keypress (read info in terminal)
                self.environment.render(
                    current_state=current,
                    next_states=[next_state],
                    texts=g_values,
                    colors=g_values,
                    wait=True,
                )

            #self.environment.render(path=path, wait=True, use_keyboard=True)
        return None 



if __name__ == "__main__":
    # Create a Map instance
    MAP = """
    .S...
    .###.
    ...#G
    """
    map = map_from_image("/Users/williambalucha/FEL CVUT/2_semester/KUI/1_HW/maps/normal/normal10b.png")
    # Create an environment as a SearchProblem initialized with the map
    env = SearchProblem(map, graphics=True)
    # Create the agent and find the path
    agent = Agent(env)
    agent.find_path()
