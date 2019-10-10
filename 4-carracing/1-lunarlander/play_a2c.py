import gym
import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.models import Model

EPISODES = 10


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = True
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.discount_factor = .9

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        self.actor.load_weights("./save_model/lunarlander_actor.h5")
        self.critic.load_weights("./save_model/lunarlander_critic.h5")

    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))
        actor_input = Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
        mu_0 = Dense(self.action_size, activation='tanh', kernel_initializer='he_uniform')(actor_input)
        sigma_0 = Dense(self.action_size, activation='softplus', kernel_initializer='he_uniform')(actor_input)

        mu = Lambda(lambda x: x)(mu_0)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        critic_input = Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
        # value_hidden = Dense(self.hidden2, activation='relu')(critic_input)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(critic_input)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        actor.summary()
        critic.summary()

        return actor, critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.state_size]))
        mu = mu[0]
        sigma_sq = sigma_sq[0]
        epsilon = np.random.randn(self.action_size)
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -1, 1)
        return action, np.mean(sigma_sq)


if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action, sigma = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state

            if done:
                print("episode: {} | score : {:.3f}".format(e, score))