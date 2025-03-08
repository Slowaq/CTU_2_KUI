import random
from kuimaze2 import SearchProblem, Map, State







class Agent:
    """Simple example of a random agent."""

    def __init__(self, environment: SearchProblem):
        self.environment = environment

    def find_path(self) -> list[State]:
        """
        Find a path from the start state to any goal state.

        This method must be implemented by you.
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
                path.append(state)
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
                    stack.append((new_node,[path + [node]], current_cost + cost))

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

        path = [State(0, 4), State(1, 4)]
        self.environment.render(path=path, wait=True, use_keyboard=True)
        return path


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
