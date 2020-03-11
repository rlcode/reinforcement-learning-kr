import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

from skimage.color import rgb2gray
from skimage.transform import resize

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                            input_shape=state_size)
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.fc = Dense(512, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self, action_size, state_size=(84, 84, 4)):
        self.render = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000

        self.memory = deque(maxlen=100000)
        self.no_op_steps = 30

        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(self.learning_rate, clipnorm=10.)

        self.avg_q_max, self.avg_loss = 0, 0

        self.update_target_model()

        self.writer = tf.summary.create_file_writer('summary/breakout_dqn')

        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')
        os.makedirs(self.model_path, exist_ok=True)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def huber_loss(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.array([sample[0][0] / 255. for sample in mini_batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_history = np.array([sample[3][0] / 255. for sample in mini_batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(model_params)

            predicts = self.model(history)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_history)

            max_q = np.amax(target_predicts, axis=1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q

            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
            self.avg_loss += loss.numpy()

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)

    global_step = 0
    score_avg = 0
    score_max = 0
    action_dict = {0:1, 1:2, 2:3, 3:3}

    num_episode = 50000
    for e in range(num_episode):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
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

            agent.avg_q_max += np.amax(agent.model(np.float32(history / 255.))[0])

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += reward
            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if (global_step % agent.update_target_rate == 0 and global_step > agent.train_start):
                agent.update_target_model()

            if dead:
                history = np.stack((next_state, next_state,
                                    next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:

                    with agent.writer.as_default():
                        tf.summary.scalar('Total Reward/Episode', score, step=e)
                        tf.summary.scalar('Average Max Q/Episode', agent.avg_q_max / float(step), step=e)
                        tf.summary.scalar('Duration/Episode', step, step=e)
                        tf.summary.scalar('Average Loss/Episode', agent.avg_loss / float(step), step=e)

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                score_max = score if score > score_max else score_max

                print("episode: {:3d} | score: {:3.2f} | score max : {:3.2f} | score avg: {:3.2f} | memory length: {:4d} | epsilon: {:.4f} | q avg : {:3.2f} | avg loss : {:3.2f}".format(
                      e, score, score_max, score_avg, len(agent.memory), agent.epsilon, agent.avg_q_max / float(step), agent.avg_loss / float(step)))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/model", save_format="tf")
