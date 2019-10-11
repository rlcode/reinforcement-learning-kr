import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf

EPISODES = 10


# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class Reinforce(tf.keras.Model):
    def __init__(self, action_size):
        super(Reinforce, self).__init__()
        self.fc1 = tf.keras.layers.Dense(24, activation='relu')
        self.fc2 = tf.keras.layers.Dense(24, activation='relu')
        self.fc_out = tf.keras.layers.Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy


# 그리드월드 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self):
        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15

        self.model = Reinforce(self.action_size)
        self.model.load_weights('save_model/reinforce')

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.05)
    agent = ReinforceAgent()

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            score += reward

            state = next_state

            if done:
                print("episode: {:3d} | score: {:3d}".format(e, score))