import gym
import time
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

# 브레이크아웃에서의 테스트를 위한 A3C 에이전트 클래스


class A3CTestAgent:
    def __init__(self, action_size, state_size, model_path):
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
    # 테스트를 위한 환경, 모델 생성
    env = gym.make("BreakoutDeterministic-v4")
    state_size = (84, 84, 4)
    action_size = 3
    model_path = './save_model/trained/model'
    render = True

    agent = A3CTestAgent(action_size, state_size, model_path)
    action_dict = {0:1, 1:2, 2:3, 3:3}

    num_episode = 10
    for e in range(num_episode):
        done = False
        dead = False

        score, start_life = 0, 5
        observe = env.reset()

        # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
        for _ in range(random.randint(1, 30)):
            observe, _, _, _ = env.step(1)

        # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
        state = pre_processing(observe)
        history = np.stack([state, state, state, state], axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if render:
                env.render()
                time.sleep(0.05)

            # 정책 확률에 따라 행동을 선택
            action, policy = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            real_action = action_dict[action]
            # 죽었을 때 시작하기 위해 발사 행동을 함
            if dead:
                action, real_action, dead = 0, 1, False

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)

            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
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
                # 각 에피소드 당 학습 정보를 기록
                print("episode: {:3d} | score : {:4.1f}".format(e, score))
