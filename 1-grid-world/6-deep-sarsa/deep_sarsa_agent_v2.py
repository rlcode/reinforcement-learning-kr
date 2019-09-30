from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import pylab
import random
import numpy as np
from environment import Env

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
from tensorflow.keras import backend
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optim

EPISODES = 1000


class DeepSARSA(tf.keras.Model):
    def __init__(self, state_size, action_size, learning_rate,
                 epsilon, epsilon_min, epsilon_decay, discount_factor):
        super(DeepSARSA, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor

        self.model = models.Sequential()
        self.model.add(layers.Dense(30, activation='relu'))
        self.model.add(layers.Dense(30, activation='relu'))
        self.model.add(layers.Dense(self.action_size, activation='linear'))

        self.compile(loss='mse', optimizer=optim.Adam(lr=self.learning_rate))
        self.build(input_shape=(1, self.state_size))
        self.summary()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        q_values = self.model(x)
        #x = self.hidden1(x)
        #x = self.hidden2(x)

        #q_values = self.linear(x)

        return q_values

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            q_values = self.call(state)
            return np.argmax(q_values[0])

    def update(self, state, action, reward, next_state, next_action, done):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)

        target = self.predict(state)[0]
        # 살사의 큐함수 업데이트 식
        if done:
            target[action] = reward
        else:
            next_q = self.predict(next_state)[0]
            # tf.assign(target[action], [(reward + self.discount_factor * next_q[next_action])])
            target[action] = (reward + self.discount_factor * next_q[next_action])

        # 출력 값 reshape
        target = np.reshape(target, [1, 5])
        # 인공신경망 업데이트
        self.fit(state, target, epochs=1, verbose=0)


# 그리드월드 예제에서의 딥살사 에이전트
class Trainer:
    def __init__(self, env, load_model=None):
        self.env = env
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.agent = DeepSARSA(state_size=self.state_size,
                               action_size=self.action_size,
                               learning_rate=self.learning_rate,
                               epsilon=self.epsilon,
                               epsilon_min=0.01,
                               epsilon_decay=.9999,
                               discount_factor=0.99)

        self.global_step = 0
        self.scores, self.episodes = [], []

        if load_model is not None:
            self.epsilon = 0.05
            self.agent = models.load_model('./save_model/deep_sarsa_pretrained.h5')

    def train(self):

        for e in range(EPISODES):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state, [1, 15])

            while not done:

                self.global_step += 1

                action = self.agent.get_action(state)

                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, 15])
                next_action = self.agent.get_action(next_state)

                self.agent.update(state, action, reward,
                                  next_state, next_action, done)

                state = next_state
                score += reward

                state = copy.deepcopy(next_state)

                if done:
                    self.scores.append(score)
                    self.episodes.append(e)
                    pylab.plot(self.episodes, self.scores, 'b')
                    pylab.savefig("./save_graph/deep_sarsa_plot.png")

                    print("Episodes : {:04d}, Score: {:.2f}, Global_step : {:03d}, Epsilon : {:.4f}"
                          .format(e, score, self.global_step, self.agent.epsilon))

                if e % 100 == 0:
                    self.agent.model.save('./save_model/deep_sarsa_pretrained.h5')


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    trainer = Trainer(env)
    trainer.train()
