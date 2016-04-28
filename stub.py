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
        self.bin_height = 20
        self.bin_vel = 5
        self.alpha = 1
        self.gamma = 1
        
        # Initialize total rewards matrix, total state visit matrix, expected reward matrix, transition count matrix, transition
        # probability matrix, Q-matrix, and V-matrix
        num_tree_dist = 600 / self.bin_width
        num_tree_height = 400 / self.bin_height
        num_monkey_height = 400 / self.bin_height
        self.vel_array = [-np.inf ,-40, -30, -20, -10, 0, 10, 20, 30, 40, np.inf]
        num_monkey_vel = len(self.vel_array)-1 
        self.S = num_tree_dist*num_tree_height*num_monkey_height*num_monkey_vel
        
        self.Rtotal = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.Ntotal = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.R = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        
        self.Ntotal_transition = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel,   num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.transition_prob = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel,   num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        
        self.Q = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.V = np.zeros((num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        
    def velocity_bin(self, vel_array, monkey_vel):
        for i in range(len(vel_array)-1):
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
        
        # Find what state bin we are in
        self.state_D = state['tree']['dist'] / self.bin_width # distance to tree
        if self.state_D < 0: # make distance non-negative
            self.state_D = 0
        self.state_T = state['tree']['bot'] / self.bin_height # height of bottom of tree
        self.state_M = state['monkey']['bot'] / self.bin_height # height of bottom of monkey
        self.state_V = self.velocity_bin(self.vel_array, state['monkey']['vel']) # monkey's velocity
        
        self.Rtotal[self.last_action, += self.last_action
        
        
        new_action = 1

        self.last_action = new_action
        self.last_state  = state
        
        return self.last_action
        
    def value_iterate(self,epsilon): 
        def loop():
            V_old = self.V
            for action in [0,1]:
                self.Q[action] = self.R[action] + self.gamma*np.dot(self.transprob[action],V_old).T
            self.policy = np.argmax(self.Q, axis=0)
            self.V = self.Q[self.policy,range((self.Q).shape[1])]
            return
        
        loop()
        while abs(sum(V_old-self.V)) > epsilon: 
            loop()
            
        return
    
    def explore_action_callback(self,state):
        '''
        This is the action function used during the exploration period of model-learning.
        It's the same as the staff-provided action_callback function, jumping randomly.
        '''
        new_action = 0 #npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward
        
        return self.last_reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    if iters < 20:
        print "I can't learn that fast! Try more iterations."
    
    # DATA-GATHERING PHASE
    for ii in range(15):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.explore_action_callback,
                             reward_callback=learner.reward_callback)
        # Loop until you hit something.
        while swing.game_loop():
            pass  
        # Save score history.
        hist.append(swing.score)
        # Reset the state of the learner.
        learner.reset()
    
    # EXPLOITATION PHASE
    for ii in range(iters)[15:]:
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
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
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


