<p align="center"><img width="90%" src="images/Reinforcement-Learning.png" /></p>

--------------------------------------------------------------------------------

> [RLCode](https://rlcode.github.io)팀이 직접 만든 강화학습 예제들을 모아놓은 Repo 입니다. [영문 (English)](https://github.com/rlcode/reinforcement-learning)
>
> Maintainers - [이웅원](https://github.com/dnddnjs), [이영무](https://github.com/zzing0907), [양혁렬](https://github.com/Hyeokreal), [이의령](https://github.com/wooridle), [김건우](https://github.com/keon)

[Pull Request](https://github.com/rlcode/reinforcement-learning-kr/pulls)는 언제든 환영입니다.
문제나 버그, 혹은 궁금한 사항이 있으면 [이슈](https://github.com/rlcode/reinforcement-learning-kr/issues)에 글을 남겨주세요.


## 필요한 라이브러리들 (Dependencies)
1. Python 3.6
2. Tensorflow 2.1.0
3. numpy 1.17.0
4. gym 0.16.0
5. pillow 6.1.0
6. matplot 3.1.1
7. scikit-image 0.15.0

### 설치 방법 (Install Requirements)
```
pip install -r requirements.txt
```


## 목차 (Table of Contents)

**Grid World** - 비교적 단순한 환경인 그리드월드에서 강화학습의 기초를 쌓기

- [정책 이터레이션 (Policy Iteration)](./1-grid-world/1-policy-iteration)
- [가치 이터레이션 (Value Iteration)](./1-grid-world/2-value-iteration)
- [살사 (SARSA)](./1-grid-world/3-sarsa)
- [큐러닝 (Q-Learning)](./1-grid-world/4-q-learning)
- [Deep SARSA](./1-grid-world/5-deep-sarsa)
- [REINFORCE](./1-grid-world/6-reinforce)

**CartPole** - 카트폴 예제를 이용하여 여러가지 딥러닝을 강화학습에 응용한 알고리즘들을 적용해보기

- [Deep Q Network](./2-cartpole/1-dqn)
- [Actor Critic (A2C)](./2-cartpole/2-actor-critic)
- [Continuous Actor Critic (A2C)](./2-cartpole/3-continuous-actor-critic)

**Atari 브레이크아웃** - 딥러닝을 응용하여 좀더 복잡한 Atari 브레이크아웃 게임을 마스터하는 에이전트 만들기

- [Deep Q Network](./3-atari/1-breakout-dqn)
- [Asynchronous Advantage Actor Critic(A3C)](./3-atari/1-breakout-a3c)