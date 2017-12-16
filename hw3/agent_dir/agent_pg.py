import os

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam

from agent_dir.agent import Agent

MODELDIR = 'model/pg33'

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)

        # ====================== Model init
        self.actions = 3
        self.all_reward = []
        self.avg_reward = []
        self.episode = 2000
        self.gamma = 0.997
        self.lr = 0.001

        def build():
            inp = Input([80, 80, 1])
            m = Conv2D(64, 6, strides=6, padding='same', activation='relu', kernel_initializer='he_uniform')(inp)
            m = Conv2D(32, 2, strides=1, padding='same', activation='relu', kernel_initializer='he_uniform')(m)
            m = Flatten()(m)
            m = Dense(64, activation='relu', kernel_initializer='he_uniform')(m)
            m = Dense(32, activation='relu', kernel_initializer='he_uniform')(m)
            m = Dense(self.actions, activation='softmax')(m)

            model = Model(inputs=inp, outputs=m)
            opt = Adam(self.lr)
            model.compile(loss='categorical_crossentropy', optimizer=opt)
            model.summary()
            return model

        self.model = build()

        if os.path.exists(MODELDIR + '/model'):
            self.model = load_model(MODELDIR + '/model')
        else:
            if os.path.exists(MODELDIR):
                with open(MODELDIR + '/log', 'w') as f:
                    f.write('')


    def init_game_setting(self):
        self.lastState = None
        self.hist_x = []
        self.hist_g = []
        self.hist_p = []
        self.hist_r = []
        self.episode += 1
        pass


    def train(self):
        while True:
            self.init_game_setting()
            state = self.env.reset()

            done = False
            while not done:
                act, act_prob = self.make_action(state, test=False)
                state, reward, done, info = self.env.step(act)

                self.hist_r.append([reward])
                self.hist_x.append([self.lastState])
                self.hist_p.append([act_prob])
                self.hist_g.append(np.array([1 if i == act-1 else 0 for i in range(3)], dtype=np.float32) - act_prob)

            # ========== Update
            reward = np.sum(self.hist_r)
            self.all_reward.append(reward)
            avg = sum(self.all_reward[-30:])/len(self.all_reward[-30:])
            self.avg_reward.append(avg)

            print('episode: {}\tscore: {}/{}\treward: {}\taverage: {:0.2f}'.format(self.episode, self.hist_r.count([1]), self.hist_r.count([-1]), reward, avg))

            grads = np.squeeze(np.vstack(self.hist_g))
            h_prob = np.squeeze(np.vstack(self.hist_p))

            rewards = np.vstack(self.hist_r)
            discounted_rewards = np.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(0, rewards.size)):
                if rewards[t] != 0:
                    running_add = 0
                running_add = running_add * self.gamma + rewards[t]
                discounted_rewards[t] = running_add
            rewards = discounted_rewards

            rewards = (rewards-np.mean(rewards)) / np.std(rewards)
            gradients = grads * rewards
            y = h_prob + self.lr * gradients
            x = np.vstack(self.hist_x)

            self.model.train_on_batch([x], y)

            if self.episode % 10 == 0:
                if not os.path.exists(MODELDIR):
                    os.makedirs(MODELDIR)
                self.model.save(MODELDIR + '/model')
                if self.episode % 500 == 0:
                    self.model.save(MODELDIR + '/model' + '-' + str(self.episode))
                with open(MODELDIR + '/log', 'a') as f:
                    for i in reversed(range(10)):
                        f.write(str(self.episode-i) + ',' +  str(self.avg_reward[-(i+1)]) + '\n')

                print('saved model: ', self.episode)
        pass


    def make_action(self, state, test=True):
        # Input: RGB Screen: (210, 160, 3)
        # Return: action(index): ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

        state = state[35:195:2, ::2, 0]
        state[state == 144] = 0
        state[state == 109] = 0
        state[state != 0] = 1
        state = np.expand_dims(state, -1)

        if self.lastState is None:
            self.lastState = state
        diffState = state - self.lastState
        self.lastState = state

        state = np.expand_dims(state, 0)
        act_prob = self.model.predict(state, batch_size=1)
        # print(act_prob[0], end=' ')
        action = np.random.choice(3, p=act_prob[0])
        if test:
            return action + 1

        return action + 1, act_prob[0]
