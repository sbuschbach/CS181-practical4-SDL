# Imports.
import numpy as np
import numpy.random as npr
import pdb
from sklearn.preprocessing import normalize

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_state_dict = None
        
        # Initialize discretizing size
        self.gamma = 0.8
        self.gravity = None
        self.iteration = 0 
        
        # Initialize total rewards matrix, total state visit matrix, expected reward matrix, transition count matrix, transition
        # probability matrix, Q-matrix, V-matrix, and policy
        self.tree_dist_array = [np.inf] + range(460, -160, -80) + [-np.inf]
        num_tree_dist = len(self.tree_dist_array) - 1
        
        self.tree_height_array = [-np.inf] + range(-50, 150, 16) + [np.inf]
        self.num_tree_height = len(self.tree_height_array) - 1
        
        self.vel_array = [-np.inf , 0, np.inf]
        num_monkey_vel = len(self.vel_array)-1
        
        num_gravity = 3
        self.gravity_dict = {1:0, 4:1, None:2}

        self.S = num_tree_dist*self.num_tree_height*num_monkey_vel*num_gravity
        
        self.Rtotal = np.zeros((2,num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))
        self.Ntotal = np.zeros((2,num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))
        self.R = np.zeros((2,num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))
        
        self.Ntotal_transition = np.zeros((2,num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity, 
                                           num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))
        self.transition_prob = np.empty((2,num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity,
                                         num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))
        (self.transition_prob).fill(float(1)/float(self.S))
        
        self.Q = np.zeros((2,num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))
        self.V = np.zeros((num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))
        self.policy = np.random.choice([0,1],(num_tree_dist,self.num_tree_height,num_monkey_vel,num_gravity))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_state_dict = None
        self.gravity = None
        self.iteration = 0
        
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
    
    def get_state_dict(self, state):  
        state_D = self.desc_bin(self.tree_dist_array, state['tree']['dist']) # distance from tree
        state_T = self.asc_bin(self.tree_height_array, (state['monkey']['bot'] - state['tree']['bot'])) # distance from bottom of tree
        state_V = self.asc_bin(self.vel_array, state['monkey']['vel']) # monkey's velocity
        state_G = self.gravity_dict[self.gravity] # gravity
       
        return {'treedist':state_D, 'treebot':state_T, 'monkeyvel':state_V, 'gravity':state_G}
        
    
    def update_params(self, current_state):
        # Update count and total reward for last action in last state
        self.Rtotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                    self.last_state_dict['monkeyvel'],self.last_state_dict['gravity']] += self.last_reward
        self.Ntotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                    self.last_state_dict['monkeyvel'],self.last_state_dict['gravity']] += 1
        
        # Update MLE estimate for expected reward
        totalr = self.Rtotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                             self.last_state_dict['monkeyvel'],self.last_state_dict['gravity']]
        totaln = self.Ntotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                             self.last_state_dict['monkeyvel'],self.last_state_dict['gravity']]
        self.R[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
               self.last_state_dict['monkeyvel'],self.last_state_dict['gravity']] = float(totalr)/float(totaln)
        
        # Update count of transitions
        self.Ntotal_transition[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'], 
                               self.last_state_dict['monkeyvel'], self.last_state_dict['gravity'], current_state['treedist'],
                               current_state['treebot'], current_state['monkeyvel'], current_state['gravity']] += 1
        
        # Update MLE for transition probabilities
        totaln_transition = self.Ntotal_transition[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                                                   self.last_state_dict['monkeyvel'], self.last_state_dict['gravity'], 
                                                   current_state['treedist'], current_state['treebot'],
                                                   current_state['monkeyvel'], current_state['gravity']]
        self.transition_prob[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                             self.last_state_dict['monkeyvel'], self.last_state_dict['gravity'], current_state['treedist'],
                             current_state['treebot'], current_state['monkeyvel'], 
                             current_state['gravity']] = float(totaln_transition)/float(totaln)
        
        # Normalize transition probabilities
        array = self.transition_prob[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                                     self.last_state_dict['monkeyvel'], self.last_state_dict['gravity']]
        array1 = np.reshape(array, (self.S,1))
        array2 = normalize(array1, axis=1, norm='l1')
        array3 = np.reshape(array2,(array.shape))
        self.transition_prob[self.last_action, self.last_state_dict['treedist'], 
                             self.last_state_dict['treebot'], 
                             self.last_state_dict['monkeyvel'], self.last_state_dict['gravity']] = array3
        
        return
        
    def value_iterate(self): 
        pdb.set_trace()
        # Reshape multidimensional arrays for easier manipulation
        self.flatQ = np.reshape(self.Q, (2,self.S))
        self.flatR = np.reshape(self.R, (2,self.S))
        self.flatV = np.reshape(self.V, (self.S,1))
        self.flatpolicy = np.reshape(self.policy,(1,self.S))
        self.flat_transition_prob = np.reshape(self.transition_prob,(2,self.S,self.S))
        
        # Define function for 1 loop of value iteration
        def loop():
            self.V_old = self.flatV
            for action in [0,1]:
                reward = np.reshape(self.flatR[action],(1,self.S))
                transition = np.reshape(self.flat_transition_prob[action],(self.S,self.S))
                values = reward + self.gamma*np.dot(transition, self.V_old).T
                self.flatQ[action] = values #np.reshape(values,(1,self.S))
            self.flatpolicy = np.argmax(self.flatQ, axis=0)
            self.flatpolicy = np.reshape(self.flatpolicy, (1,self.S))
            self.flatV = self.flatQ[self.flatpolicy, range((self.flatQ).shape[1])]
            self.flatV = np.reshape(self.flatV,(self.S, 1))
            return
        
        # Loop until values stop changing
        loop()
        while not np.allclose(self.V_old, self.flatV, atol=10):
            loop()       
            
        # Reshape arrays back into multidimensional form for easier use
        self.Q = np.reshape(self.flatQ, (self.Q).shape)
        self.R = np.reshape(self.flatR, (self.R).shape)
        self.V = np.reshape(self.flatV, (self.V).shape)
        self.policy = np.reshape(self.flatpolicy, (self.policy).shape)
        self.transition_prob = np.reshape(self.flat_transition_prob, (self.transition_prob).shape)
        
        return     

    
    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        # Get dictionary of bins for current state
        state_dict = self.get_state_dict(state)
        
        if self.last_state != None: 
            # Update iteration value
            self.iteration += 1.

            # Update MLE estimates of reward and transition probability
            self.update_params(state_dict)        
            
            # Update optimal policy with new MLE estimates if it's not our first epoch
            self.value_iterate()
            
        # Get best action from updated policy
        new_action = self.policy[state_dict['treedist'],state_dict['treebot'],state_dict['monkeyvel'],state_dict['gravity']]
            
        # Update last action/state for next iteration
        self.last_action = new_action
        self.last_state  = state
        self.last_state_dict = state_dict
        
        return self.last_action
    
    
    def explore_action_callback(self, state):
        '''
        This is the action function used during the exploration period of model-learning.
        It's the same as the staff-provided action_callback function, jumping randomly.
        '''
        # Get dictionary of bins for current state
        state_dict = self.get_state_dict(state)
            
        if self.last_state != None:
            # Update MLE estimates of reward and transition probability
            self.update_params(state_dict)    
            
            # Update optimal policy with new MLE estimates if it's not our first epoch
            self.value_iterate()
            
            # Randomly choose new action
            new_action = npr.rand() < 0.1
            
        # Don't jump first first iteration, use difference between first and second iteration to calculate gravity
        if self.last_state == None or self.iteration == 1:
            self.gravity1 = state['monkey']['bot']
            new_action = 0       
        if self.iteration == 2:
            gravity2 = state['monkey']['bot']
            self.gravity = self.gravity1 - gravity2

        # Update last action/state for next round
        self.last_action = new_action
        self.last_state  = state
        self.last_state_dict = state_dict

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
    for ii in range(30):
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
    for ii in range(iters)[30:]:
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
	run_games(agent, hist, 1000, 10)

	# Save history. 
	np.save('hist',np.array(hist))


