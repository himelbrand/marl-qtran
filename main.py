import torch
from pettingzoo.butterfly import knights_archers_zombies_v6 as kaz
from supersuit import black_death_v1
import numpy as np
import cv2
from nn import SingleQ, CentralNN
from pettingzoo.utils.to_parallel import to_parallel
from qtran import QTran


def choose_actions(policy_net, observations):
    with torch.no_grad():
        q_values = {agent: singleQ(observations[agent])[0].numpy() for singleQ, agent in
                    zip(policy_net.singleQs, observations)}
        actions = {agent: q_values[agent].argmax() for agent in observations}
        return actions


#  Init parallel env
env = kaz.env(spawn_rate=20, num_knights=2, num_archers=2,
              killable_knights=True, killable_archers=True, line_death=True, pad_observation=True,
              max_cycles=900)
env = black_death_v1(env)
env = to_parallel(env)

# init Qtran
q_tran = QTran()

q_tran.train(env)
print('Done training...')
policy = q_tran.get_policy()


def run_policy():
    # scoring and steps of episode
    observations = env.reset()
    scores = {a: 0 for a in env.agents}
    steps = 0
    env.aec_env.env.render()
    # Running a single episode
    dones = [False] * 4
    while not all(dones):
        actions = choose_actions(policy, observations)
        observations, rewards, dones, _ = env.step(actions)
        for a in rewards:
            scores[a] += rewards[a]
        steps += 1
        env.aec_env.env.render()

    env.aec_env.env.close()
    print(f'Episode done with total score: {sum(scores.values())} and total steps of: {steps}')
    print('Individual scores:')
    for a in scores:
        print(f'{a}: {scores[a]}')


run_policy()

