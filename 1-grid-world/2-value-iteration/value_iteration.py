import numpy as np
from environment import GraphicDisplay, Env


class ValueIteration:
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.env = env
        # 가치 함수를 2차원 리스트로 초기화
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # 할인율
        self.discount_factor = 0.9

    # 벨만 최적 방정식을 통해 다음 가치 함수 계산
    def value_iteration(self):
        # 다음 가치함수 초기화
        next_value_table = [[0.0] * self.env.width 
                           for _ in range(self.env.height)]

        # 모든 상태에 대해서 벨만 최적방정식을 계산                           
        for state in self.env.get_all_states():
            # 마침 상태의 가치 함수 = 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue

            # 벨만 최적 방정식
            value_list = []
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))

            # 최댓값을 다음 가치 함수로 대입
            next_value_table[state[0]][state[1]] = max(value_list)

        self.value_table = next_value_table

    # 현재 가치 함수로부터 행동을 반환
    def get_action(self, state):
        if state == [2, 2]:
            return []

        # 모든 행동에 대해 큐함수 (보상 + (감가율 * 다음 상태 가치함수))를 계산
        value_list = []
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)
            value_list.append(value)

        # 최대 큐 함수를 가진 행동(복수일 경우 여러 개)을 반환
        max_idx_list = np.argwhere(value_list == np.amax(value_list))
        action_list = max_idx_list.flatten().tolist()
        return action_list

    def get_value(self, state):
        return self.value_table[state[0]][state[1]]


if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
