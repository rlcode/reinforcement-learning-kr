import gym
import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.models import Model

EPISODES = 10


class A2CAgent:
    def __init__(self, state_size, action_size, max_action):
        self.render = True
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.max_action = max_action

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor.load_weights("./save_model/pendulum_actor.h5")
        self.critic.load_weights("./save_model/pendulum_critic.h5")

    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        state = Input(batch_shape=(None, self.state_size))
        hidden_layer = Dense(30, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform')(state)
        actor_mu = Dense(self.action_size, activation='tanh', 
                         kernel_initializer='he_uniform')(hidden_layer)
        actor_sigma = Dense(self.action_size, activation='softplus', 
                         kernel_initializer='he_uniform')(hidden_layer)
        actor_mu = Lambda(lambda x: self.max_action * x)(actor_mu)
        actor_sigma = Lambda(lambda x: x + 10e-5)(actor_sigma)
        actor = Model(inputs=state, outputs=(actor_mu, actor_sigma))
        actor.summary()
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        state = Input(batch_shape=(None, self.state_size))
        hidden_layer = Dense(30, input_dim=self.state_size, activation='relu', 
                         kernel_initializer='he_uniform')(state)
        hidden_layer = Dense(30, activation='relu', 
                         kernel_initializer='he_uniform')(hidden_layer)
        state_value = Dense(1, activation='linear', 
                         kernel_initializer='he_uniform')(hidden_layer)
        critic = Model(inputs=state, outputs=state_value)
        critic.summary()
        return critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma = self.actor.predict(state)
        epsilon = np.random.randn(self.action_size)
        action = mu[0] + sigma[0] * epsilon
        action = np.clip(action, -self.max_action, self.max_action)
        return action


if __name__ == "__main__":
    # Pendulum-v0 환경, 최대 타임스텝 수가 200
    env = gym.make('Pendulum-v0')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    
    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size, max_action)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state

            if done:
                print("episode: {} | score : {:.3f}".format(e, score))