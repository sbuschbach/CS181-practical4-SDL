# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        # Initialize discretizing size
        self.bin_width = 100
        self.bin_height = 10
        self.bin_vel = 5
        self.alpha = 1
        self.gamma = 1
        
        # Initialize Q-matrix to 0s
        num_actions = 2
        num_tree_dist = 600 / self.bin_width
        num_tree_height = 400 / self.bin_height
        num_monkey_height = 400 / self.bin_height
        self.vel_array = [-np.inf ,-40, -30, -20, -10, 0, 10, 20, 30, 40, np.inf]
        num_monkey_vel = len(self.vel_array)-1 # TO DO: FIX THIS TO NOT BE HARDCODED
        self.Q = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
    
    def velocity_bin(self, vel_array, monkey_vel):
        for i in range(len(vel_array)-1):
            #print vel_array[i] < monkey_vel < vel_array[i+1]
            if vel_array[i] < monkey_vel <= vel_array[i+1]:
                vel_bin = i
                return vel_bin
        
    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        # Q Table
        # Discretize states
        # state_T = bottom of tree
        # state_M = bottom of monkey
        # state_V = velocity
        self.state_D = state['tree']['dist'] / self.bin_width # distance to tree
        if self.state_D < 0: # make distance non-negative
            self.state_D =0
        self.state_T = state['tree']['bot'] / self.bin_height # height of bottom of tree
        self.state_M = state['monkey']['bot'] / self.bin_height # height of bottom of monkey
        self.state_V = self.velocity_bin(self.vel_array, state['monkey']['vel']) # monkey's velocity
        
        new_action = npr.rand() < 0.1
        #new_action = 1
        new_state  = state
        
        
        
        #Q_old = self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V]
        #Q_best = max(self.Q[:,0,0,0,0])
        #Q_new = Q_old + self.alpha * (reward + (self.gamma * Q_best) - Q_old)
        #print Q_best

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=True,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 2, 10)
    # (agent, hist, iterations, how fast the game runs)

	# Save history. 
	np.save('hist',np.array(hist))


