import os
import gym
import time
import threading
import random
import numpy as np
import tensorflow as tf

from skimage.color import rgb2gray
from skimage.transform import resize

from tensorflow.keras.layers import Conv2D, Flatten, Dense


# ActorCritic 인공신경망
class ActorCritic(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(ActorCritic, self).__init__()

        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                            input_shape=state_size)
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.shared_fc = Dense(512, activation='relu')

        self.policy = Dense(action_size, activation='linear')
        self.value = Dense(1, activation='linear')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.shared_fc(x)

        policy = self.policy(x)
        value = self.value(x)
        return policy, value


class A3CTestAgent:
    def __init__(self, action_size, state_size, model_path='./save_model/a3c'):
        self.action_size = action_size

        self.model = ActorCritic(action_size, state_size)
        self.model.load_weights(model_path)

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.model(history)[0][0]
        policy = tf.nn.softmax(policy)
        action_index = np.random.choice(self.action_size, 1, p=policy.numpy())[0]
        return action_index, policy


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = gym.make("BreakoutDeterministic-v4")
    state_size = (84, 84, 4)
    action_size = 3
    model_path = './save_model/model'

    agent = A3CTestAgent(action_size, state_size, model_path)
    action_dict = {0:1, 1:2, 2:3, 3:3}

    num_episode = 30
    for e in range(num_episode):
        done = False
        dead = False

        score, start_life = 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, 30)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        history = np.stack([state, state, state, state], axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            # env.render()
            action, policy = agent.get_action(history)

            real_action = action_dict[action]

            if dead:
                action, real_action, dead = 0, 1, False

            next_observe, reward, done, info = env.step(real_action)

            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead, start_life = True, info['ale.lives']

            score += reward

            if dead:
                history = np.stack((next_state, next_state,
                                    next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            if done:
                print("episode: {:3d} | score : {:3.2f}".format(e, score))
