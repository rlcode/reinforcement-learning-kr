import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf

EPISODES = 10


# 딥살사 인공신경망
class DeepSARSA(tf.keras.Model):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = tf.keras.layers.Dense(30, activation='relu')
        self.fc2 = tf.keras.layers.Dense(30, activation='relu')
        self.fc_out = tf.keras.layers.Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self):
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15

        self.epsilon = 0.01
        self.model = DeepSARSA(self.action_size)
        self.model.load_weights('save_model/trained/deep_sarsa')

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            q_values = self.model(state)
            return np.argmax(q_values[0])


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.05)
    agent = DeepSARSAgent()

    scores, episodes = [], []

    for e in range(EPISODES):
        score = 0
        done = False
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            state = next_state
            score += reward

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d}".format(e, score))