# Imports.
import numpy as np
import numpy.random as npr
import pdb

from SwingyMonkey import SwingyMonkey


class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        # Initialize discretizing size
        self.bin_width = 50
        self.bin_height = 40
        self.alpha = 1
        self.gamma = 1
        self.gravity = None
        self.epoch = 0
        
        # Initialize Q-matrix to 0s
        num_actions = 2
        self.tree_dist_array = [np.inf] + range(460, -160, -40) + [-np.inf]
        #self.tree_dist_array = [600, 500, 400, 300, 200, 100, 0, -np.inf]
        num_tree_dist = len(self.tree_dist_array) - 1

        self.tree_height_array = [-np.inf] + range(-50, 150, 8) + [np.inf]
        self.num_tree_height = len(self.tree_height_array) - 1

        self.vel_array = [-np.inf , -30, -10, 0, 3,7, np.inf]
        num_monkey_vel = len(self.vel_array)-1
        num_gravity = 3
        self.gravity_dict = {1:0, 4:1, None:2}
        self.Q = np.zeros((num_actions,num_tree_dist,self.num_tree_height, num_gravity))
    
    def asc_bin(self, interval_array, val):
        for i in range(len(interval_array)-1):
            if interval_array[i] < val <= interval_array[i+1]:
                bin = i
                return bin
    
    def desc_bin(self, interval_array, val):
        for i in range(len(interval_array)-1):
            if interval_array[i] >= val > interval_array[i+1]:
                bin = i
                return bin
    
    '''
    def tree_dist_bin(self, tree_dist_array, tree_dist):
        for i in range(len(tree_dist_array)-1):
            if tree_dist_array[i] < tree_dist <= tree_dist_array[i+1]:
                regular_bins = 600 / self.bin_width
                tree_dist_bin = i + regular_bins
                return tree_dist_bin
    '''
    
    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None
        self.epoch = 0

    def action_callback(self, state):
        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
    
        # Get Q-old from previous time
        if self.last_action != None:
            Q_old = self.Q[self.last_action, self.state_D, self.state_T, self.state_G]

        # Get state space
        '''
        self.state_D = state['tree']['dist'] # distance to tree
        if self.state_D >= 0:
            self.state_D = self.state_D / self.bin_width
        else: 
            self.state_D = self.tree_dist_bin(self.tree_dist_array, self.state_D)
        '''
        self.state_D = self.desc_bin(self.tree_dist_array, state['tree']['dist'])
        self.state_T = self.asc_bin(self.tree_height_array, (state['monkey']['bot'] - state['tree']['bot'])) # monkey's position in relation to tree

        """
        if self.state_T >= 0:
            self.state_T = self.state_T / self.bin_height 
        else:
            self.state_T = (abs(self.state_T) / self.bin_height) + self.num_tree_height/2
        """

        '''
        self.state_T = state['tree']['bot'] / self.bin_height # height of bottom of tree
        self.state_M = state['monkey']['bot'] / self.bin_height # height of bottom of monkey
        '''
        self.state_V = self.asc_bin(self.vel_array, state['monkey']['vel']) # monkey's velocity
        self.state_G = self.gravity_dict[self.gravity]

        if self.last_action != None:
            Q_best = max(self.Q[:,self.state_D, self.state_T, self.state_G])
            Q_new = Q_old + self.alpha * (self.last_reward + (self.gamma * Q_best) - Q_old)
            print self.epoch, self.Q[self.last_action, self.state_D, self.state_T, self.state_G]
            self.Q[self.last_action, self.state_D, self.state_T, self.state_G] = Q_new
            print self.epoch, self.Q[self.last_action, self.state_D, self.state_T, self.state_G]
            '''
            Q_best = max(self.Q[:,self.state_D, self.state_T, self.state_M, self.state_V, self.state_G])
            Q_new = Q_old + self.alpha * (self.last_reward + (self.gamma * Q_best) - Q_old)
            print self.epoch, self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G] = Q_new
            print self.epoch, self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            '''
        
        # Check if this is the very beginning of the game
        if self.last_state == None or self.epoch == 1:
            # we will choose to not jump in this state so we can figure out gravity
            self.gravity1 = state['monkey']['bot']
            new_action = 0
        else:
            #print True
            # Calculate the Qs for jumping or not
            Q_stay = self.Q[0, self.state_D, self.state_T, self.state_G]
            Q_jump = self.Q[1, self.state_D, self.state_T, self.state_G]
            '''
            Q_stay = self.Q[0, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            Q_jump = self.Q[1, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            '''
            if Q_stay == Q_jump: # if equal, choose randomly
                new_action = npr.rand() < 0.1
            else:
                new_action = np.argmax([Q_stay,Q_jump])
        
        # If we are on second state, calculate gravity
        if self.epoch == 2:
            gravity2 = state['monkey']['bot']
            self.gravity = self.gravity1 - gravity2

        self.epoch += 1
        new_state = state
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
    
    for ii in range(iters):
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
        
        if ii is (iters - 1):
            pdb.set_trace()

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 400, 70)

	# Save history. 
	np.save('hist',np.array(hist))


