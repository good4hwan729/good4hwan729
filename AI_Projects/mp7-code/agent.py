import numpy as np
import utils
import random


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def discretize(self, state):                                                #Helper function to discretize state
        if state[0] == 40:
            adjoining_wall_x = 1
        elif state[0] == 480:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        if state[1] == 40:
            adjoining_wall_y = 1
        elif state[1] == 480:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        if state[0] > state[3]:
            food_dir_x = 1
        elif state[0] < state[3]:
            food_dir_x = 2
        else:
            food_dir_x = 0

        if state[1] > state[4]:
            food_dir_y = 1
        elif state[1] < state[4]:
            food_dir_y = 2
        else:
            food_dir_y = 0

        if (state[0] - 40, state[1]) in state[2]:
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0

        if (state[0] + 40, state[1]) in state[2]:
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0

        if (state[0], state[1] - 40) in state[2]:
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0

        if (state[0], state[1] + 40) in state[2]:
            adjoining_body_bottom = 1
        else:
            adjoining_body_bottom = 0

        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)

    def explore(self, state, action):                                           #Helper function for exploration function

        if self.N[state + (action,)] < self.Ne:
            return 1
        else:
            return self.Q[state + (action,)]


    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        real_state = self.discretize(state)
        #print(points - self.points)                                 #Discretize the state using helper function
        if points - self.points == 1:                                           #Compute R(s) based off points and dead
            Rs = 1
            self.points += 1
        elif dead == True:
            Rs = -1
            self.points = 0
        else:
            Rs = -0.1

        if self._train == True and self.s != None and self.a != None:           #If we are training and we aren't at our first train, update Q-table
            #self.N[self.s + (self.a,)] += 1
            alpha = self.C / (self.C + self.N[self.s + (self.a,)])                 #Compute learning rate

            trials = []
            for i in range(4):
                trials.append(self.Q[real_state + (i,)])

            #if list(self.s + (self.a,)) == [0, 0, 1, 0, 0, 0, 0, 0, 3]:
            #    print(self.Q[self.s + (self.a,)])
            #    print(alpha)
            #    print(Rs)
                #self.Q[self.s + (self.a,)] = -0.1

            self.Q[self.s + (self.a,)] += alpha*(Rs + self.gamma * max(trials) - self.Q[self.s + (self.a,)])


        tests = []
        for action in range(3,-1, -1):                                          #Find best actions (tiebreaker applied)
            tests.append(self.explore(real_state, action))

        final = 3 - np.argmax(tests)                                            #Reverse to consider tiebreaker
        #print(final)
        #self.N[real_state + (final,)] += 1
        self.s = real_state                                                     #Set new values for s and a
        self.a = final

        if dead == True:                                                        #Return nothing if dead
            self.reset()
            return

        self.N[real_state + (final,)] += 1                                      #Increment N(s',a')

        return final
