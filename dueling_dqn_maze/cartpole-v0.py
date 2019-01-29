import gym
from dueling_dqn import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(num_features=env.observation_space.shape[0],
                  num_actions=env.action_space.n,
                  learning_rate=0.01,
                  reward_decay = 0.9,
                  e_greedy=0.9,
                  replace_target_iter=100,
                  memory_size=2000,
                  batch_size=32,
                  e_greedy_increment=0.0008,
                  output_graph=True)
total_steps = 0

for i_episode in range(100):
    observation = env.reset()
    print("observation:",observation)
    ep_r = 0
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = observation_   # 细分开, 为了修改原配的 reward
        # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()  # 学习

        ep_r += reward
        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
# 最后输出 cost 曲线
RL.plot_cost()
