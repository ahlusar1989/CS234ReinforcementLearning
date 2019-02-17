
"""
General purpose Monte Carlo model for training on-policy methods.
"""
from base import FiniteModel
import numpy as np

def MonteCaroloFiniteModel(FiniteModel):
    def __init__(self, state_space, action_space, gamma=1.0, epsilon=0.1):
        """MCModel takes in state_space and action_space (finite) 
        Arguments
        ---------
        
        state_space: int OR list[observation], where observation is any hashable type from env's obs.
        action_space: int OR list[action], where action is any hashable type from env's actions.
        gamma: float, discounting factor.
        epsilon: float, epsilon-greedy parameter.
        
        If the parameter is an int, then we generate a list, and otherwise we generate a dictionary.
        >>> m = FiniteMCModel(2,3,epsilon=0)
        >>> m.Q
        [[0, 0, 0], [0, 0, 0]]
        >>> m.Q[0][1] = 1
        >>> m.Q
        [[0, 1, 0], [0, 0, 0]]
        >>> m.pi(1, 0)
        1
        >>> m.pi(1, 1)
        0
        >>> d = m.generate_returns([(0,0,0), (0,1,1), (1,0,1)])
        >>> assert(d == {(1, 0): 1, (0, 1): 2, (0, 0): 2})
        >>> m.choose_action(m.pi, 1)
        0
        """
        super(FiniteMCModel, self).__init__(state_space, action_space, gamma, epsilon)

    def generate_returns(self, episode) :
		"""Backup on returns per time period in an epoch
        Arguments
        ---------
        
        ep: [(observation, action, reward)], an episode trajectory in chronological order.
        """
        G = {} # return on state
        C = 0 # cumulative reward
        for tuple_of_ep in reversed(episode):
        	observation, action, reward = tuple_of_ep
        	G[(observation, action, reward)] = C = reward + self.gamma * C
        return G
	
	def update_Q(self, episode):
        """Performs a action-value update.
        Arguments
        ---------
        
        ep: [(observation, action, reward)], an episode trajectory in chronological order.
        """
        G = generate_returns(episode)
        for s in G:
        	state, action = s
        	q = self.Q[state][action]
        	self.Ql[state][action] += 1
        	N = self.Ql[state][action]
        	self.Q[state][action] = q * (N / N + 1) + G[s] / (N + 1)


