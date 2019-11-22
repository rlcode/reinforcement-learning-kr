import os
import gym
import time
import threading
import multiprocessing
import random
import numpy as np
import tensorflow as tf
from queue import Queue


from skimage.color import rgb2gray
from skimage.transform import resize

from tensorflow.python import keras

global episode
episode = 0
# 환경 생성
env_name = "BreakoutDeterministic-v4"

class ActorCritic(tf.keras.Model):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()

        self.action_size = action_size

        self.conv1 = tf.keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.shared_fc = tf.keras.layers.Dense(256, activation='relu')

        self.policy = tf.keras.layers.Dense(self.action_size, activation='linear')
        self.value = tf.keras.layers.Dense(1, activation='linear')


    def call(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.shared_fc(x)

        policy = self.policy(x)
        value = self.value(x)
        
        return policy, value

class A3CTestAgent:
    def __init__(self, action_size, model_path):
        