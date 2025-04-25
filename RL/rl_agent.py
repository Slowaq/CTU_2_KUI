import random, time
from numpy import random as random_numpy
from typing import Optional

from kuimaze2 import Action, RLProblem, State
from kuimaze2.typing import Policy, QTable, VTable

T_MAX = 200  # Max steps in episode
TIMEOUT = 20


class RLAgent:
    """Implementation of Q-learning algorithm.

    With the provided code, the agent just walks randomly
    and does not update q-values correctly
    """

    def __init__(
        self,
        env: RLProblem,
        gamma: float = 0.9,
        alpha: float = 0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.decay_factor = 0.30
        self.init_q_table()

    def init_q_table(self) -> None:
        """Create and initialize the q-table

        It is initialized as a dictionary of dictionaries;
        it can be used as 'self.q_table[state][action]'.
        """
        self.q_table = {
            state: {action: 0.0 for action in self.env.get_action_space()}
            for state in self.env.get_states()
        }

    def get_values(self) -> VTable:
        """Return the state values derived from the q-table"""
        return {
            state: max(q_values.values()) for state, q_values in self.q_table.items()
        }

    def render(
        self,
        current_state: Optional[State] = None,
        action: Optional[Action] = None,
        values: Optional[VTable] = None,
        q_values: Optional[QTable] = None,
        policy: Optional[Policy] = None,
        *args,
        **kwargs,
    ) -> None:
        """Visualize the state of the algorithm"""
        values = values or self.get_values()
        q_values = q_values or self.q_table
        # State values will be displayed in the squares
        sq_texts = (
            {state: f"{value:.2f}" for state, value in values.items()} if values else {}
        )
        # State-action value will be displayed in the triangles
        tr_texts = {
            (state, action): f"{value:.2f}"
            for state, action_values in q_values.items()
            for action, value in action_values.items()
        }
        # If policy is given, it will be displayed in the middle
        # of the squares in the "triangular" view
        actions = {}
        if policy:
            # actions = {state: str(action) for state, action in policy.items()}
            actions = {state: action.name[0].upper() for state, action in policy.items()}
        # The current state and chosen action will be displayed as an arrow
        state_action = (current_state, action)
        if current_state is None or action is None:
            state_action = None
        self.env.render(
            *args,
            square_texts=sq_texts,
            square_colors=values,
            triangle_texts=tr_texts,
            triangle_colors=q_values,
            middle_texts=actions,
            state_action_arrow=state_action,
            wait=True,
            **kwargs,
        )

    def extract_policy(self) -> Policy:
        """Extract policy from Q-values"""
        policy = {}
        for state, a2q in self.q_table.items():
            max_q = max(a2q.values())
            best = [a for a, q in a2q.items() if q == max_q]
            policy[state] = random.choice(best)

        return policy

    def learn_policy(self) -> Policy:
        """Run Q-learning algoritm to learn a policy"""
   
        start_time = time.time()
        while time.time() - start_time < TIMEOUT - 0.05:
            self.single_episode()

        self.env.reset()
        return self.extract_policy()
    
    def choose_action(self, state):
        explo = random_numpy.choice(["itation", "ration"], p = [self.epsilon, 1 - self.epsilon]) 

        # :)
        if explo == "itation": 
            return self.env.sample_action()
        elif explo == "ration":
            q_vals = self.q_table[state]
            max_q = max(q_vals.values())
            best_actions = [action for action, value in q_vals.items() if value == max_q]
            return random.choice(best_actions)
   
    def single_episode(self):
        total_reward: float = 0
        episode_finished = False
        t = 0

        # Print a table header
        print(
            f"{'State':^9}{'Action':^9}{'Next state':^11}{'Reward':>9}"
            f"{'Old Q':>9}{'Trial':>9}{'New Q':>9}",
        )

        # Reset the environment and get the initial state
        next_state = self.env.reset()
        path = [next_state]
        while not episode_finished and t < T_MAX:
            t += 1
            state = next_state
            action = self.choose_action(state)
            next_state, reward, episode_finished = self.env.step(action)
            total_reward += reward
            if next_state is not None:
                path.append(next_state)

            # Remember the old q-value for printing it in the table
            old_q = self.q_table[state][action]

            if next_state is None:
                trial = reward
            else:
                bext_next = max(self.q_table[next_state].values())
                trial = reward + self.gamma * bext_next

            delta = trial - old_q

            self.q_table[state][action] = old_q + self.alpha * delta

            print(
                f"{str(state):^9}{str(action):^9}{str(next_state):^9}{reward:>9.2f}"
                f"{old_q:>9.2f}{trial:>9.2f}{self.q_table[state][action]:>9.2f}"
            )

            # Extract the current policy such that we can visualize it
            policy = self.extract_policy()
            self.render(current_state=state, action=action, path=path, policy=policy)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_factor)

        if episode_finished:
            print(f"Episode finished in a terminal state after {t} steps.")
        else:
            print(
                "Episode did not reach any terminal state. Maximum time horizon reached!"
            )
        print("Total reward:", total_reward)

        return total_reward


if __name__ == "__main__":
    from kuimaze2 import Map
    from kuimaze2.map_image import map_from_image

    MAP = """
    ...G
    .#.D
    S...
    """
    #map = Map.from_string(MAP)
    map = map_from_image("RL/maps/normal/normal3.png")
    env = RLProblem(
        map,
        action_probs=dict(forward=0.8, left=0.1, right=0.1, backward=0.0),
        graphics=True,
    )

    agent = RLAgent(env, gamma=0.9, alpha=0.1)
    policy = agent.learn_policy()
    print("Policy found:", policy)
    agent.render(policy=policy, use_keyboard=True)
