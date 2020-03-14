import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability import distributions as tfd


# 정책 신경망과 가치 신경망 생성
class ContinuousA2C(tf.keras.Model):
    def __init__(self, action_size):
        super(ContinuousA2C, self).__init__()
        self.actor_fc1 = Dense(24, activation='tanh')
        self.actor_mu = Dense(action_size,
                              kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.actor_sigma = Dense(action_size, activation='softplus',
                                 kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1,
                                kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc1(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)
        sigma = sigma + 1e-5

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return mu, sigma, value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class ContinuousA2CAgent:
    def __init__(self, action_size, max_action):
        # 행동의 크기 정의
        self.action_size = action_size
        self.max_action = max_action

        # 정책신경망과 가치신경망 생성
        self.model = ContinuousA2C(self.action_size)
        self.model.load_weights("./save_model/trained/model")

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    gym.envs.register(
        id='CartPoleContinuous-v0',
        entry_point='env:ContinuousCartPoleEnv',
        max_episode_steps=500,
        reward_threshold=475.0)

    env = gym.make('CartPoleContinuous-v0')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # 액터-크리틱(A2C) 에이전트 생성
    agent = ContinuousA2CAgent(action_size, max_action)

    scores, episodes = [], []

    num_episode = 10
    for e in range(num_episode):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state

            if done:
                print("episode: {:3d} | score: {:3d}".format(e, int(score)))