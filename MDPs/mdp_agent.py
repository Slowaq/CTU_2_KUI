#!/usr/bin/env python3

import random
from typing import Optional

from kuimaze2 import MDPProblem
from kuimaze2.typing import VTable, QTable, Policy
from kuimaze2.map_image import map_from_image


class MDPAgent:
    """Base class for VI and PI agents"""

    def __init__(self, env: MDPProblem, gamma: float = 0.9, epsilon: float = 0.001):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

    def init_values(self) -> VTable:
        """Initialize all values to 0"""
        return {state: 0 for state in self.env.get_states()}
    
    def init_q_values(self) -> VTable:
        """Initialize all q_values to 0. 
        This method creates nested dictionaries and 
        sets every action of every state to 0
        """
        return {state: {action: 0 for action in self.env.get_actions(state)} for state in self.env.get_states()}
    
    def Bellman_equation(self, state, values: VTable, action: str) -> float:
        """Compute Bellman equation when preforming action from state"""
        return sum((probab * (self.env.get_reward(next_state) + self.gamma * values[next_state]))     # Bellman equation 
               for next_state, probab in self.env.get_next_states_and_probs(state, action) 
               if not self.env.is_terminal(state))
    
    def evaluation(self, values: VTable, qvalues: QTable, policy: Optional[Policy] = None):
        """*** Evaluate states using Bellman equation ***

            If a policy is provided, the function follows the policy (Policy Iteration).
            If no policy is provided, it performs a full Bellman update (Value Iteration).
        """

        while True:
            delta = 0 # Used for termination
            
            for state in values:
                v = values[state]
                if policy:
                    action = policy[state]
                    values[state] = self.Bellman_equation(state, values, action)
                    
                else:
                    for action in self.env.get_actions(state):
                        qvalues[state][action] = self.Bellman_equation(state, values, action)

                    values[state] = max(qvalues[state][action] for action in self.env.get_actions(state))

                delta = max(delta, abs(v - values[state])) 

            
            if delta <= self.epsilon:
                break

    def render(
        self,
        values: Optional[VTable] = None,
        qvalues: Optional[QTable] = None,
        policy: Optional[Policy] = None,
        **kwargs,
    ):
        """Render the environment with added agent's data"""

        value_texts = {state: f"{value:.2f}" for state, value in values.items()}

        qvalue_texts = {
            (state, action): f"{q:.2f}"
            for state, actions in qvalues.items()
            for action, q in actions.items()
        }

        # Prepare policy for rendering
        policy_texts = {state: f"{policy[state]}" for state in self.env.get_states()}

        self.env.render(
            square_colors=values,
            square_texts=value_texts,
            triangle_colors=qvalues,
            triangle_texts=qvalue_texts,
            middle_texts=policy_texts,
            **kwargs,
        )


class ValueIterationAgent(MDPAgent):

    def find_policy(self) -> Policy:
        values = self.init_values()
        q_values = self.init_q_values()
        print(values)
    
        self.evaluation(values, q_values)

        policy = {state: max(q_values[state], key=q_values[state].get) for state in values}     # Policy extraction

        # Store computed values for rendering
        self.values = values
        self.q_values = q_values

        return policy 


class PolicyIterationAgent(MDPAgent):

    def init_policy(self) -> Policy:
        """Create a random policy"""
        return {
            state: random.choice(self.env.get_actions(state))
            for state in self.env.get_states()
        }
    
    def policy_improvement(self, values: VTable, policy: Policy):
        change = False
        new_policy = {}
        
        for state in self.env.get_states():
            if self.env.is_terminal(state):
                new_policy[state] = policy[state] # Terminal states keep the same action
                continue
            
            old_action = policy[state]
            best_action = None
            best_q = float("-inf")

            for action in self.env.get_actions(state):
                q_sa = self.Bellman_equation(state, values, action)

                if q_sa > best_q:
                    best_q = q_sa
                    best_action = action
            
            new_policy[state] = best_action

            if best_action != old_action:
                change = True
        
        return new_policy, change
            
    def find_policy(self) -> Policy:
        policy = self.init_policy()
        values = self.init_values()
        q_values = self.init_q_values()

        stable = False
        while not stable:
            # Policy evaluation
            self.evaluation(values=values, qvalues=q_values, policy=policy)
            
            # Policy improvement
            policy, change = self.policy_improvement(values=values, policy=policy)

            if not change:
                stable = True
                break

        # Store values for rendering
        self.values = values
        self.q_values = q_values

        return policy


if __name__ == "__main__":
    from kuimaze2 import Map
    from kuimaze2.map_image import map_from_image

    MAP = """
    ...G
    .#.D
    S...
    """
    map = Map.from_string(MAP)
    # map = map_from_image("MDPs/maps/normal/normal1.png")
    env = MDPProblem(
        map,
        action_probs=dict(forward=0.8, left=0.1, right=0.1, backward=0.0),
        graphics=True,
    )

    print(env.get_states())
    agent = ValueIterationAgent(env, gamma=0.9, epsilon=0.001)
    #agent = PolicyIterationAgent(env, gamma=0.9, epsilon=0.001)
    policy = agent.find_policy()
    print("Policy found:", policy)
    agent.render(policy=policy, wait=True, values=agent.values, qvalues=agent.q_values)
