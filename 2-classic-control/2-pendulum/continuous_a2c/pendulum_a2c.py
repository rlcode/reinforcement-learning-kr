import gym
import pylab
import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 1000


class A2CAgent:
    def __init__(self, state_size, action_size, max_action):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.max_action = max_action

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.9
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

        if self.load_model:
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
        entropy = np.sum(np.log(sigma * np.sqrt(2. * np.pi * np.e)))
        return action, entropy

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, 1))

        mu, sigma = self.actor.output
        pdf = 1. / (sigma * np.sqrt(2. * np.pi)) * \
              K.exp(-K.square(action - mu) / (2. * sigma * sigma))
        log_pdf = K.sum(K.log(pdf + K.epsilon()), axis=-1)
        entropy = K.sum(K.log(sigma * np.sqrt(2. * np.pi * np.e)))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, 
                                        [], actor_loss)

        train = K.function([self.actor.input, action, advantages], 
                            [], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, 
                                        [], loss)
        train = K.function([self.critic.input, discounted_reward], 
                           [], updates=updates)
        return train

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, 1))

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor_updater([state, action, advantages])
        self.critic_updater([state, target])


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
    score_avg = 0

    for e in range(EPISODES):
        done = False
        score = 0
        entropy_list = []    
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action, entropy = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.train_model(state, action, reward/10, next_state, done)

            score += reward
            entropy_list.append(entropy)
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                score_avg = 0.99 * score_avg + 0.01 * score if score_avg != 0 else score
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/pendulum_a2c.png")
                print("episode: {} | score : {:d} | score_avg : {:d} | entropy : {:.3f}".format(
                       e, int(score), int(score_avg), np.mean(entropy)))

        # save the model
        if e % 50 == 0:
            agent.actor.save_weights("./save_model/pendulum_actor.h5")
            agent.critic.save_weights("./save_model/pendulum_critic.h5")
