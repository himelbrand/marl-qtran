from pettingzoo.butterfly import knights_archers_zombies_v6 as kaz
from supersuit import black_death_v1
import numpy as np
import cv2


#  Init parallel env
env = kaz.parallel_env(spawn_rate=20, num_knights=2, num_archers=2,
                       killable_knights=True, killable_archers=True, line_death=True, pad_observation=True,
                       max_cycles=900)
#  Constant number of agents - when dead observation is all black
env = black_death_v1(env)

#  Env used for rendering
visual_env = env.aec_env.env

# Reset env + initial render
observations = env.reset()
visual_env.render()

# scoring and steps of episode
scores = {a: 0 for a in env.agents}
steps = 0

# Running a single episode
while not env.env_done:
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    observations, rewards, dones, infos = env.step(actions)
    for a in rewards:
        scores[a] += rewards[a]
    steps += 1
    visual_env.render()
visual_env.render()
visual_env.close()

# print results of episode
print(scores)
print(steps)
print(f'total score: {sum(scores.values())}')






# for agent in env.agent_iter():
#     env.render()
#     observation, reward, done, info = env.last()
#     print(agent, done, steps[agent])
#     cv2.imwrite(f'out/{agent}-{steps[agent]}.png', cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
#     scores[agent] += reward
#     if not done:
#         action = env.action_spaces[agent].sample()
#     else:
#         action = None
#     env.step(action)
#     steps[agent] += 1

