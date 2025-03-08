import random, math, heapq
from kuimaze2 import SearchProblem, Map, State

class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent's row index
        self.parent_j = 0  # Parent's column index
        self.f = float('inf')  # Cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination

# Calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(row, col, dest):
    return ((row - dest.r) ** 2 + (col - dest.c) ** 2) ** 0.5


class Agent:
    """Simple example of a random agent."""

    def __init__(self, environment: SearchProblem):
        self.environment = environment

    def find_path(self) -> list[State]:
        """Finding path using A* algorithm"""

        start_state = self.environment.get_start()
        goal_state = self.environment.get_goals()

        open_list = [] # Priority queue
        heapq.heappush(open_list, (0, id(start_state), start_state, []))

        visited = set()
        g_values = {start_state: 0}

        while open_list:
            f, ID, current, path = heapq.heappop(open_list) 

            if current in visited:
                continue

            visited.add(current)

            if self.environment.is_goal(current):
                print("Goal reached!!!")
                path.append(current)
                self.environment.render(path=path, wait=True, use_keyboard=True)
                return path
            
            print(self.environment.get_actions(current))
            for action in self.environment.get_actions(current):
                print(action)
                next_state, cost = self.environment.get_transition_result(current, action)

                new_g = g_values[current] + cost

                goal_list = [x for x in goal_state]
                new_h = calculate_h_value(next_state.r, next_state.c, goal_list[0])
                
                new_f = new_g + new_h

                if next_state not in visited or new_f < f:
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
                state = next_state 

            #self.environment.render(path=path, wait=True, use_keyboard=True)
        return path 


        """ 

        # Get the start state from the environment
        start_state = self.environment.get_start()
        stack = [(start_state, [], 0)]
        visited = set()
        # Lets track the cost to reach states
        costs = {start_state: 0}
        print(f"Starting random search from state: {start_state}")
        while stack:
            node, path, current_cost = stack.pop()
            
            if node in visited and current_cost >= costs[node]:
                continue

            costs[node] = current_cost


            if self.environment.is_goal(node):
                print("Goal reached!!!")
                path.append(node)
                print("tu")
                #self.environment.render(path=path, wait=True, use_keyboard=True)
                return path
            

            visited.add(node)

            actions = self.environment.get_actions(node)
            for action in actions:
                new_node, cost = self.environment.get_transition_result(
                    node, action
                )

                if new_node not in visited:
                    stack.append((new_node,path + [node], current_cost + cost))

            costs[new_node] = costs[node] + cost
            print(
                f"Transition: {node} -> {new_node}, cost of new state: {costs[new_node]}"
            )
            # Visualize the state of the algorithm:
            # * Use cost values as texts in cells
            # * Use cost values as colors of cells
            # * Mark current and new states
            # * Wait for keypress (read info in terminal)
            self.environment.render(
                current_state=node,
                next_states=[new_node],
                texts=costs,
                colors=costs,
                wait=True,
            )
            state = new_node 

        self.environment.render(path=path, wait=True, use_keyboard=True)
        return path """


if __name__ == "__main__":
    # Create a Map instance
    MAP = """
    .S...
    .###.
    ...#G
    """
    map = Map.from_string(MAP)
    # Create an environment as a SearchProblem initialized with the map
    env = SearchProblem(map, graphics=True)
    # Create the agent and find the path
    agent = Agent(env)
    agent.find_path()
