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
        self.last_state_dict = None
        
        # Initialize discretizing size
        self.bin_width = 200
        self.bin_height = 100
        self.gamma = 1
        
        # Initialize total rewards matrix, total state visit matrix, expected reward matrix, transition count matrix, transition
        # probability matrix, Q-matrix, V-matrix, and policy
        num_tree_dist = 600 / self.bin_width
        num_tree_height = 400 / self.bin_height
        num_monkey_height = 400 / self.bin_height
        self.vel_array = [-np.inf ,-40, -20, -10, 0, 10, 20, 40, np.inf]
        num_monkey_vel = len(self.vel_array)-1 
        self.S = num_tree_dist*num_tree_height*num_monkey_height*num_monkey_vel
        
        self.Rtotal = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.Ntotal = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.R = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        
        self.Ntotal_transition = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel,  
                                           num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.transition_prob = np.empty((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel,  
                                         num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        (self.transition_prob).fill(float(1)/float(self.S))
        
        self.Q = np.zeros((2,num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.V = np.zeros((num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))
        self.policy = np.random.choice([0,1],(num_tree_dist,num_tree_height,num_monkey_height,num_monkey_vel))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_state_dict = None
        
    def velocity_bin(self, vel_array, monkey_vel):
        for i in range(len(vel_array)-1):
            if vel_array[i] < monkey_vel <= vel_array[i+1]:
                vel_bin = i
                return vel_bin
    
    def get_state_dict(self,state):   
        state_D = state['tree']['dist'] / self.bin_width # distance to tree
        if state_D < 0: # make distance non-negative
            state_D = 0
        state_T = state['tree']['bot'] / self.bin_height # height of bottom of tree
        state_M = state['monkey']['bot'] / self.bin_height # height of bottom of monkey
        state_V = self.velocity_bin(self.vel_array, state['monkey']['vel']) # monkey's velocity
        
        return {'treedist':state_D, 'treebot':state_T, 'monkeybot':state_M, 'monkeyvel':state_V}
        
    
    def update_params(self, current_state):
        # Update count and total reward for last action in last state
        self.Rtotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                    self.last_state_dict['monkeybot'], self.last_state_dict['monkeyvel']] += self.last_reward
        self.Ntotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                    self.last_state_dict['monkeybot'], self.last_state_dict['monkeyvel']] += 1
        
        # Update MLE estimate for expected reward
        totalr = self.Rtotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                             self.last_state_dict['monkeybot'], self.last_state_dict['monkeyvel']]
        totaln = self.Ntotal[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                             self.last_state_dict['monkeybot'], self.last_state_dict['monkeyvel']]
        self.R[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'], self.last_state_dict['monkeybot'],
               self.last_state_dict['monkeyvel']] = float(totalr)/float(totaln)
        
        # Update count of transitions
        self.Ntotal_transition[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'], 
                               self.last_state_dict['monkeybot'], self.last_state_dict['monkeyvel'], current_state['treedist'],
                               current_state['treebot'], current_state['monkeybot'], current_state['monkeyvel']] += 1
        
        # Update MLE for transition probabilities
        totaln_transition = self.Ntotal_transition[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                                                   self.last_state_dict['monkeybot'], self.last_state_dict['monkeyvel'],
                                                   current_state['treedist'], current_state['treebot'], current_state['monkeybot'],
                                                   current_state['monkeyvel']]
        self.transition_prob[self.last_action, self.last_state_dict['treedist'], self.last_state_dict['treebot'],
                                                   self.last_state_dict['monkeybot'], self.last_state_dict['monkeyvel'],
                                                   current_state['treedist'], current_state['treebot'], current_state['monkeybot'],
                                                   current_state['monkeyvel']] = float(totaln_transition)/float(totaln)
        
        return
        
    def value_iterate(self): 
        # Reshape multidimensional arrays for easier manipulation
        self.flatQ = np.reshape(self.Q, (2,self.S))
        self.flatR = np.reshape(self.R, (2,self.S))
        self.flatV = np.reshape(self.V, (1,self.S))
        self.flatpolicy = np.reshape(self.policy,(1,self.S))
        self.flat_transition_prob = np.reshape(self.transition_prob,(2,self.S,self.S))
        
        # Define function for 1 loop of value iteration
        def loop():
            self.V_old = np.reshape(self.flatV,(self.S,1))
            for action in [0,1]:
                values = np.add(self.flatR[action], np.dot(self.flat_transition_prob[action],self.V_old).T)
                self.flatQ[action] = np.reshape(values,(1,self.S))
            self.flatpolicy = np.argmax(self.flatQ, axis=0)
            self.flatpolicy = np.reshape(self.flatpolicy, (1,self.S))
            self.flatV = self.flatQ[self.flatpolicy,range((self.flatQ).shape[1])]
            self.flatV = np.reshape(self.flatV,(self.S,1))
            return
        
        # Loop until values stop changing
        loop()
        while not np.allclose(self.V_old,self.flatV,20):
            loop()       
            
        # Reshape arrays back into multidimensional form for easier use
        self.Q = np.reshape(self.flatQ, (self.Q).shape)
        self.R = np.reshape(self.flatR, (self.R).shape)
        self.V = np.reshape(self.flatV, (self.V).shape)
        self.policy = np.reshape(self.flatpolicy, (self.policy).shape)
        print self.policy.shape
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
        
        if self.last_state_dict == None:
            # Get best action from current policy
            new_action = self.policy[state_dict['treedist'],state_dict['treebot'],state_dict['monkeybot'],state_dict['monkeyvel']]
            
            # Update last action/state for next round
            self.last_action = new_action

            self.last_state  = state
            self.last_state_dict = state_dict
        
        else: 
            # Update MLE estimates of reward and transition probability
            self.update_params(state_dict)        
            
            # Update optimal policy with new MLE estimates if it's not our first epoch
            self.value_iterate()
            
            # Get best action from updated policy
            new_action = self.policy[state_dict['treedist'],state_dict['treebot'],state_dict['monkeybot'],state_dict['monkeyvel']]
            
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
        
        if self.last_state_dict == None:
            # Don't jump for first action
            new_action = 0
            
            # Update last action/state for next round
            self.last_action = new_action
            self.last_state  = state
            self.last_state_dict = state_dict
            
        else:
            # Update MLE estimates of reward and transition probability
            self.update_params(state_dict)    
            
            # Update optimal policy with new MLE estimates if it's not our first epoch
            self.value_iterate()
            
            # Randomly choose new action
            new_action = npr.rand() < 0.1
            
            # Update last action/state for next round
            self.last_action = new_action
            self.last_state  = state
            self.last_state_dict = state_dict

        self.last_action = new_action
        self.last_state  = state

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


