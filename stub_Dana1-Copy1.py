# Imports.
import numpy as np
import numpy.random as npr
import pdb

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
        self.bin_width = 50
        self.bin_height = 40
        self.alpha = .5
        self.gamma = .5
        self.gravity = None
        self.epoch = 0
        
        # Initialize Q-matrix to 0s
        num_actions = 2
        num_tree_dist = 3
        self.num_tree_height = (400 / self.bin_height) * 2
        self.monkey_height_array = [-np.inf, 30, np.inf]
        num_monkey_height = len(self.monkey_height_array) - 1
        self.vel_array = [-np.inf , -30, -10, 0, 3,7, np.inf]
        num_monkey_vel = len(self.vel_array)-1
        num_gravity = 3
        self.gravity_dict = {1:0, 4:1, None:2}
        self.Q = np.zeros((num_actions,num_tree_dist,self.num_tree_height,num_monkey_height,num_monkey_vel, num_gravity))
        
    def velocity_bin(self, vel_array, monkey_vel):
        for i in range(len(vel_array)-1):
            if vel_array[i] < monkey_vel <= vel_array[i+1]:
                vel_bin = i
                return vel_bin
    
    def monkey_height_bin(self, monkey_height_array, monkey_height):
        for i in range(len(monkey_height_array)-1):
            if monkey_height_array[i] < monkey_height <= monkey_height_array[i+1]:
                height_bin = i
                return height_bin
    
    def tree_dist_bin(self, tree_dist):
        if 45 < tree_dist <= 295:
            dist_bin = 0
        elif 295 < tree_dist <= np.inf:
            dist_bin = 1
        else:
            dist_bin = 2
        return dist_bin
    
    
    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None
        self.epoch = 0

    def action_callback(self, state):
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        #pdb.set_trace()
        
        # Get Q-old from previous time
        if self.last_action != None:
            Q_old = self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]

        # Get state space
        self.state_D = self.tree_dist_bin(state['tree']['dist'])
        self.state_T = (state['tree']['bot'] - state['monkey']['bot']) # height of bottom of tree
        if self.state_T >= 0:
            self.state_T = self.state_T / self.bin_height 
        else:
            self.state_T = (abs(self.state_T) / self.bin_height) + self.num_tree_height/2
        self.state_M = self.monkey_height_bin(self.monkey_height_array, state['monkey']['bot'])
        self.state_V = self.velocity_bin(self.vel_array, state['monkey']['vel']) # monkey's velocity
        self.state_G = self.gravity_dict[self.gravity]

        if self.last_action != None:
            print self.epoch, self.Q[:,self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            Q_best = max(self.Q[:,self.state_D, self.state_T, self.state_M, self.state_V, self.state_G])
            Q_new = Q_old + self.alpha * (self.last_reward + (self.gamma * Q_best) - Q_old)
            print self.epoch, self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G] = Q_new
            print self.epoch, self.Q[self.last_action, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
        
        # Check if this is the very beginning of the game
        if self.last_state == None or self.epoch == 1:
            # we will choose to not jump in this state so we can figure out gravity
            self.gravity1 = state['monkey']['bot']
            new_action = 0
        else:
            #print True
            # Calculate the Qs for jumping or not
            Q_stay = self.Q[0, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            Q_jump = self.Q[1, self.state_D, self.state_T, self.state_M, self.state_V, self.state_G]
            if Q_stay == Q_jump: # if equal, choose randomly
                new_action = npr.rand() < 0.08
            else:
                # e-greedy
                e_greedy = np.random.rand() < 0.15
                if e_greedy == True:
                    new_action = npr.rand() < 0.5
                    print True
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


