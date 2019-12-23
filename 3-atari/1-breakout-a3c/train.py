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
from tensorflow.compat.v1 import nn, train

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000
# 환경 생성
env_name = "BreakoutDeterministic-v4"

# ActorCritic 인공신경망
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

# 브레이크아웃에서의 A3CAgent 클래스 (글로벌신경망)    
class A3CAgent():
    def __init__(self, action_size):
        self.env_name = 'BreakoutDeterministic-v4'
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.lr = 2.5e-4
        # 쓰레드의 갯수
        self.threads = 1
        
        # 글로벌 인공신경망 생성 
        self.global_model = ActorCritic(self.action_size)
        self.global_model(tf.convert_to_tensor(np.random.random((1, *self.state_size)), dtype=tf.float32))
        
        # 인공신경망 업데이트하는 옵티마이저 함수 생성
        self.optimizer = train.RMSPropOptimizer(self.lr,
                                                decay=0.99,
                                                epsilon=0.01,
                                                use_locking=True)
        
        self.writer = tf.summary.create_file_writer('summary/breakout_a3c')
        
        self.model_path = os.path.join(os.getcwd(), 'save_model')
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        
    # 쓰레드를 만들어 학습을 하는 함수 
    def train(self):
        
        # 쓰레드 수 만큼 Runner 클래스 생성 
        runners = [Runner(self.action_size, self.state_size, 
                         self.global_model, self.optimizer,
                         self.discount_factor, self.env_name,
                         self.writer) for i in range(self.threads)]
        
        # 각 쓰레드 시작
        for i, runner in enumerate(runners):
            print("Start worker #{:d}".format(i))
            runner.start()
        
        # 10분 (600초)에 한 번씩 모델을 저장
        while True:
            self.save_model(os.path.join(self.model_path, "breakout_a3c"))
            time.sleep(60 * 10)
    
    def save_model(self, name):
        self.global_model.save_weights(name + "_model.h5")


class Runner(threading.Thread):
    global_episode = 0
    save_lock = threading.Lock()

    def __init__(self, action_size, state_size, global_model,
                 optimizer, discount_factor, env_name, writer):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.global_model = global_model
        self.optimizer = optimizer
        self.discount_factor = discount_factor

        self.states, self.actions, self.rewards = [], [], []

        self.local_model = ActorCritic(self.action_size)
        self.env = gym.make(env_name)
        self.writer = writer

        self.avg_p_max = 0
        self.avg_loss = 0

        self.t_max = 20
        self.t = 0
    
    def run(self):
        global episode
        step = 0
        
        while episode < EPISODES:
            done = False
            dead = False
            
            score, start_life = 0, 5
            observe = self.env.reset()
            next_observe = observe
            
            for _ in range(random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = self.env.step(1)
                
            state = pre_processing(next_observe, observe)
            history = np.stack([state, state, state, state], axis=2)
            history = np.reshape([history], (1, 84, 84, 4))
            
            while not done:
                step += 1
                self.t += 1
                observe = next_observe
                action, policy = self.get_action(history)
                
                # 1: 정지, 2: 왼쪽, 3: 오른쪽
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                # 선택한 행동으로 한 스텝을 실행
                next_observe, reward, done, info = self.env.step(real_action)
                
                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(next_observe, observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3],
                                         axis=3)
                
                action_prob = tf.nn.softmax(self.local_model(np.float32(history / 255.))[0])
                self.avg_p_max += np.amax(action_prob[0].numpy())
                
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                reward = np.clip(reward, -1., 1.)

                # 샘플을 저장
                self.append_sample(history, action, reward)
                
                if dead:
                    history = np.stack((next_state, next_state,
                                        next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.t = 0
                    
                if done:
                    episode += 1
                    
                    stats = [score, self.avg_p_max / float(step),
                             step]
                    print("episode:", episode, "  score:", score, "  step:", step)

                    with self.writer.as_default():
                        tf.summary.scalar('Total Reward/Episode', score, step=episode)
                        tf.summary.scalar('Average Max Prob/Episode', self.avg_p_max / float(step), step=episode)
                        tf.summary.scalar('Duration/Episode', step, step=episode)
                    
                    self.avg_p_max = 0
                    step = 0
                    
                    
    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_model(history)[0][0]
        policy = nn.softmax(policy)
        action_index = np.random.choice(self.action_size, 1, p=policy.numpy())[0]
        return action_index, policy
    
    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)
                
    
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            # value function
            running_add = self.local_model(np.float32(self.states[-1] / 255.))[-1][0].numpy()

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    def compute_loss(self, done):
        
        discounted_prediction = self.discounted_prediction(self.rewards, done)
        discounted_prediction = tf.convert_to_tensor(discounted_prediction[:, None], dtype=tf.float32)
        
        states = np.zeros((len(self.states), 84, 84, 4))
        
        for i in range(len(self.states)):
            states[i] = self.states[i]
        states = np.float32(states / 255.)

        policy, values = self.local_model(states)

        # 가치 신경망 업데이트
        advantages = discounted_prediction - values
        value_loss = tf.square(advantages)

        # 정책 신경망 업데이트
        action = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        policy_prob = nn.softmax(policy)
        action_prob = tf.reduce_sum(action * policy_prob, axis=1, keepdims=True)
        cross_entropy = tf.math.log(action_prob + 1e-10) * tf.stop_gradient(advantages)
        cross_entropy = -tf.reduce_sum(cross_entropy)
        
        entropy = tf.reduce_sum(policy_prob * tf.math.log(policy_prob + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)
        
        policy_loss = cross_entropy + 0.01 * entropy
        
        total_loss = tf.reduce_mean(0.5 * value_loss + policy_loss)
        
        return total_loss
    
    
    def train_model(self, done):
        
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done)
        
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
        
        self.local_model.set_weights(self.global_model.get_weights())
        
        self.states, self.actions, self.rewards = [], [], []
            


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe
        

if __name__ == "__main__":
    
    global_agent = A3CAgent(action_size=3)
    global_agent.train()




        
