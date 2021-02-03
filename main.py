from pettingzoo.butterfly import knights_archers_zombies_v6 as kaz
import numpy as np
import cv2
env = kaz.env(spawn_rate=20, num_knights=2, num_archers=2,
              killable_knights=True, killable_archers=True, line_death=True, pad_observation=True,
              max_cycles=900)

env.reset()
scores = {a: 0 for a in env.agents}
steps = {a: 0 for a in env.agents}

for agent in env.agent_iter():
    env.render()
    observation, reward, done, info = env.last()
    print(agent, done, steps[agent])
    cv2.imwrite(f'{agent}-{steps[agent]}.png', cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
    scores[agent] += reward
    if not done:
        action = env.action_spaces[agent].sample()
    else:
        action = None
    env.step(action)
    steps[agent] += 1
print(scores)
print(steps)
env.render()
env.close()

