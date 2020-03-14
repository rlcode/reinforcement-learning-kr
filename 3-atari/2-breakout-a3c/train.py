import os
import gym
import time
import threading
import random
import numpy as np
import tensorflow as tf

from skimage.color import rgb2gray
from skimage.transform import resize

from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# 멀티쓰레딩을 위한 글로벌 변수
global episode, score_avg, score_max
episode, score_avg, score_max = 0, 0, 0
num_episode = 8000000


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


# 브레이크아웃에서의 A3CAgent 클래스 (글로벌신경망)
class A3CAgent():
    def __init__(self, action_size, env_name):
        self.env_name = env_name
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.lr = 1e-4
        # 쓰레드의 갯수
        self.threads = 16

        # 글로벌 인공신경망 생성
        self.global_model = ActorCritic(self.action_size, self.state_size)
        # 글로벌 인공신경망의 가중치 초기화
        self.global_model.build(tf.TensorShape((None, *self.state_size)))

        # 인공신경망 업데이트하는 옵티마이저 함수 생성
        self.optimizer = AdamOptimizer(self.lr, use_locking=True)

        # 텐서보드 설정
        self.writer = tf.summary.create_file_writer('summary/breakout_a3c')
        # 학습된 글로벌신경망 모델을 저장할 경로 설정
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        # 쓰레드 수 만큼 Runner 클래스 생성
        runners = [Runner(self.action_size, self.state_size,
                          self.global_model, self.optimizer,
                          self.discount_factor, self.env_name,
                          self.writer) for i in range(self.threads)]

        # 각 쓰레드 시정
        for i, runner in enumerate(runners):
            print("Start worker #{:d}".format(i))
            runner.start()

        # 10분 (600초)에 한 번씩 모델을 저장
        while True:
            self.global_model.save_weights(self.model_path, save_format="tf")
            time.sleep(60 * 10)


# 액터러너 클래스 (쓰레드)
class Runner(threading.Thread):
    global_episode = 0

    def __init__(self, action_size, state_size, global_model,
                 optimizer, discount_factor, env_name, writer):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 넘겨준 하이준 파라미터 설정
        self.action_size = action_size
        self.state_size = state_size
        self.global_model = global_model
        self.optimizer = optimizer
        self.discount_factor = discount_factor

        self.states, self.actions, self.rewards = [], [], []

        # 환경, 로컬신경망, 텐서보드 생성
        self.local_model = ActorCritic(action_size, state_size)
        self.env = gym.make(env_name)
        self.writer = writer

        # 학습 정보를 기록할 변수
        self.avg_p_max = 0
        self.avg_loss = 0
        # k-타임스텝 값 설정
        self.t_max = 20
        self.t = 0
        # 불필요한 행동을 줄여주기 위한 dictionary
        self.action_dict = {0:1, 1:2, 2:3, 3:3}

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, e):
        avg_p_max = self.avg_p_max / float(step)
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=e)
            tf.summary.scalar('Average Max Prob/Episode', avg_p_max, step=e)
            tf.summary.scalar('Duration/Episode', step, step=e)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_model(history)[0][0]
        policy = tf.nn.softmax(policy)
        action_index = np.random.choice(self.action_size, 1, p=policy.numpy())[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # k-타임스텝의 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            # value function
            last_state = np.float32(self.states[-1] / 255.)
            running_add = self.local_model(last_state)[-1][0].numpy()

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 저장된 샘플들로 A3C의 오류함수를 계산
    def compute_loss(self, done):

        discounted_prediction = self.discounted_prediction(self.rewards, done)
        discounted_prediction = tf.convert_to_tensor(discounted_prediction[:, None],
                                                     dtype=tf.float32)

        states = np.zeros((len(self.states), 84, 84, 4))

        for i in range(len(self.states)):
            states[i] = self.states[i]
        states = np.float32(states / 255.)

        policy, values = self.local_model(states)

        # 가치 신경망 업데이트
        advantages = discounted_prediction - values
        critic_loss = 0.5 * tf.reduce_sum(tf.square(advantages))

        # 정책 신경망 업데이트
        action = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        policy_prob = tf.nn.softmax(policy)
        action_prob = tf.reduce_sum(action * policy_prob, axis=1, keepdims=True)
        cross_entropy = - tf.math.log(action_prob + 1e-10)
        actor_loss = tf.reduce_sum(cross_entropy * tf.stop_gradient(advantages))

        entropy = tf.reduce_sum(policy_prob * tf.math.log(policy_prob + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)
        actor_loss += 0.01 * entropy

        total_loss = 0.5 * critic_loss + actor_loss

        return total_loss

    # 로컬신경망을 통해 그레이디언트를 계산하고, 글로벌 신경망을 계산된 그레이디언트로 업데이트
    def train_model(self, done):

        global_params = self.global_model.trainable_variables
        local_params = self.local_model.trainable_variables

        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done)

        # 로컬신경망의 그레이디언트 계산
        grads = tape.gradient(total_loss, local_params)
        # 안정적인 학습을 위한 그레이디언트 클리핑
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        # 로컬신경망의 오류함수를 줄이는 방향으로 글로벌신경망을 업데이트
        self.optimizer.apply_gradients(zip(grads, global_params))
        # 로컬신경망의 가중치를 글로벌신경망의 가중치로 업데이트
        self.local_model.set_weights(self.global_model.get_weights())
        # 업데이트 후 저장된 샘플 초기화
        self.states, self.actions, self.rewards = [], [], []

    def run(self):
        # 액터러너끼리 공유해야하는 글로벌 변수
        global episode, score_avg, score_max

        step = 0
        while episode < num_episode:
            done = False
            dead = False

            score, start_life = 0, 5
            observe = self.env.reset()

            # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = self.env.step(1)

            # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
            state = pre_processing(observe)
            history = np.stack([state, state, state, state], axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                step += 1
                self.t += 1

                # 정책 확률에 따라 행동을 선택
                action, policy = self.get_action(history)
                # 1: 정지, 2: 왼쪽, 3: 오른쪽
                real_action = self.action_dict[action]
                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action, real_action, dead = 0, 1, False

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                observe, reward, done, info = self.env.step(real_action)

                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                # 정책확률의 최대값
                self.avg_p_max += np.amax(policy.numpy())

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
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
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1
                    score_max = score if score > score_max else score_max
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score

                    log = "episode: {:5d} | score : {:4.1f} | ".format(episode, score)
                    log += "score max : {:4.1f} | ".format(score_max)
                    log += "score avg : {:.3f}".format(score_avg)
                    print(log)

                    self.draw_tensorboard(score, step, episode)

                    self.avg_p_max = 0
                    step = 0


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    global_agent = A3CAgent(action_size=3, env_name="BreakoutDeterministic-v4")
    global_agent.train()
