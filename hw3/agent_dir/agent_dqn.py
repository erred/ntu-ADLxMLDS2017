import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from agent_dir.agent import Agent

MODELDIR = 'model/ddqn'
MODELFILE = MODELDIR + '/model'
LOGFILE = MODELDIR + '/log'

class Agent_DQN(Agent):
    def __init__(self, env, args):

        super(Agent_DQN,self).__init__(env)

        self.epsilon = 0.01
        # self.epsilon_min = 0.00001
        # self.epsilon_decay = 0.995

        self.episode = 0
        self.episode_max = 70000

        self.gamma = 0.99

        self.batchsize = 64
        self.memory = deque(maxlen=50000)
        self.reward_all = []
        self.reward_avg = []

        self.lr = 0.001
        self.model = self.buildModel()
        self.target = self.buildModel()

        if not os.path.exists(MODELDIR):
            os.makedirs(MODELDIR)

        if os.path.exists(MODELFILE):
            self.model.load_weights(MODELFILE)
            self.target.load_weights(MODELFILE)
        pass


    def buildModel(self):
        inp = Input(shape=[84, 84, 4])
        mask = Input(shape=[3])
        model = Conv2D(32, 3, activation='relu', padding='same')(inp)
        model = MaxPooling2D(2)(model)
        model = Conv2D(64, 3, activation='relu', padding='same')(model)
        model = MaxPooling2D((2,2))(model)
        model = Flatten()(model)
        model = Dense(512, activation='relu')(model)
        q_values = Dense(3)(model)
        q_values = Multiply()([q_values, mask])

        opt = Adam(self.lr)
        loss = tf.losses.huber_loss
        # loss = 'mean_squared_error'
        m = Model([inp, mask], q_values)
        m.compile(opt, loss)
        return m

    def init_game_setting(self):
        self.done = False
        self.episode += 1
        self.reward = 0
        self.state = self.env.reset()
        pass

    def train(self):
        while self.episode < self.episode_max:
            self.init_game_setting()
            while not self.done:
                act = self.make_action(self.state, test=False)
                nState, reward, self.done, _ = self.env.step(act)
                # remember stuff
                self.reward += reward
                self.memory.append((self.state, act-1, reward, nState, self.done))
                self.state = nState
            # log stuff
            self.reward_all.append(self.reward)
            avg = np.mean(self.reward_all[-31:])
            self.reward_avg.append(avg)
            print('episode: {} score: {} avg: {}'.format(self.episode, self.reward, avg))

            # replay
            batchsize = len(self.memory) if len(self.memory) < self.batchsize else self.batchsize
            self.replay(batchsize)

            # conditional log/update
            if self.episode % 100 == 0:
                self.model.save_weights(MODELFILE)
                self.target.load_weights(MODELFILE)
                if self.episode % 1000 == 0:
                    self.model.save_weights(MODELFILE +'-'+ str(self.episode))
                with open(LOGFILE, 'a') as f:
                    for i in reversed(range(100)):
                        e = self.episode-i-1
                        avg = self.reward_avg[-i-1]
                        f.write('{},{}\n'.format(e, avg))
                print('saved model')
        pass

    def replay(self, batchsize):
        batch = np.array(random.sample(self.memory, batchsize))
        nStates = np.stack(batch[:, 3])

        q_select = np.argmax(self.model.predict_on_batch([nStates, np.ones([batchsize, 3])]), 1)
        target_q = self.gamma * self.target.predict_on_batch([nStates, np.ones([batchsize, 3])])[range(batchsize), q_select]
        target_q = target_q * (1 - batch[:, 4]) + batch[:, 2]

        mask = to_categorical(batch[:, 1], 3)
        targets = mask * np.expand_dims(target_q, 1)
        self.model.train_on_batch([np.stack(batch[:, 0]), mask], targets)
        pass

    def make_action(self, state, test=True):
        if test:
            self.epsilon = 0.005
        if np.random.rand() < self.epsilon:
            return random.randrange(1, 4)
        act = self.model.predict_on_batch([np.expand_dims(state, 0), np.ones((1, 3))])
        return np.argmax(act) + 1
