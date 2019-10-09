import gym
import pylab
import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 10000


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.discount_factor = .9

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        if self.load_model:
            self.actor.load_weights("./save_model/pendulum_actor_trained.h5")
            self.critic.load_weights("./save_model/pendulum_critic_trained.h5")

    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))
        actor_input = Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
        mu_0 = Dense(self.action_size, activation='tanh', kernel_initializer='he_uniform')(actor_input)
        sigma_0 = Dense(self.action_size, activation='softplus', kernel_initializer='he_uniform')(actor_input)

        mu = Lambda(lambda x: x * 2)(mu_0)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        critic_input = Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
        # value_hidden = Dense(self.hidden2, activation='relu')(critic_input)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(critic_input)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))

        mu, sigma_sq = self.actor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)

        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.state_size]))
        mu = mu[0]
        sigma_sq = sigma_sq[0]
        epsilon = np.random.randn(self.action_size)
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -2, 2)
        return action

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.optimizer[0]([state, action, advantages])
        self.optimizer[1]([state, target])


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    # env = gym.make('LunarLanderContinuous-v2')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

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
            reward /= 10
            next_state = np.reshape(next_state, [1, state_size])
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                score_avg = 0.99 * score_avg + 0.01 * score if score_avg != 0 else score
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/continuous_control.png")
                print("episode: {} | score : {:.3f} | score_avg : {:.3f}".format(
                       e, score, score_avg))

        # save the model
        # if e % 50 == 0:
            # agent.actor.save_weights("./save_model/lunarlander_actor.h5")
            # agent.critic.save_weights("./save_model/lunarlander_critic.h5")
