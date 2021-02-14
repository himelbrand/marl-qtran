from random import randint

import torch
from pettingzoo.butterfly import pistonball_v3 as pistonball
import matplotlib.pyplot as plt
from qtran import QTran

def plot(epochs, score, scores, steps, loss):
    plt.figure()
    plt.plot(epochs, score)
    plt.title('Epochs / Total score')
    plt.xlabel('Epochs')
    plt.ylabel('Total score')

    plt.figure()
    scores_tmp = {}
    for ss in scores:
        for a in ss:
            if a in scores_tmp:
                scores_tmp[a].append(ss[a])
            else:
                scores_tmp[a] = [ss[a]]
    for a in scores_tmp:
        plt.plot(epochs, scores_tmp[a], label=a)
    plt.title('Epochs / Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.figure()
    plt.plot(epochs, steps)
    plt.title('Epochs / Steps')
    plt.xlabel('Epochs')
    plt.ylabel('Steps')

    plt.figure()
    plt.plot(epochs, loss)
    plt.title('Epochs / Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()
    input('done?')
    plt.close('all')


def load_checkpoint(epoch, path='out/epochs/epoch'):
    filepath = f'{path}{epoch}.pt'
    checkpoint = torch.load(filepath)
    print(checkpoint)
    return checkpoint


def plot_all(s_epoch=0, e_epoch=5):
    epochs = []
    score = []
    scores = []
    steps = []
    loss = []
    for epoch in range(s_epoch, e_epoch+1):
        checkpoint = load_checkpoint(epoch)
        epochs.append(epoch)
        score.append(checkpoint['eval_score'])
        scores.append(checkpoint['eval_scores'])
        steps.append(checkpoint['eval_steps'])
        loss.append(checkpoint['loss'].item())
    plot(epochs, score, scores, steps, loss)


def choose_actions(policy_net, observations):
    with torch.no_grad():
        q_values = {agent: singleQ(observations[agent])[0].numpy() for singleQ, agent in
                    zip(policy_net.singleQs, observations)}
        actions = {agent: q_values[agent].argmax() for agent in observations}
        return actions


# plot_all()
# exit(0)
#  Init parallel env
env = pistonball.parallel_env(n_pistons=15, local_ratio=0.2, time_penalty=-0.1, continuous=False, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900)
# env = black_death_v1(env)
# env = to_parallel(env)
# env.reset()
# env_done = False
# score = 0
# while not env_done:
#     env.render()
#     actions = {a: randint(0, 2) for a in env.agents}
#     _, rewards, dones, _ = env.step(actions)
#     score += sum(rewards.values())
#     env_done = all(dones.values())
# env.render()
# env.close()
# print(score)
# init Qtran
q_tran = QTran(piston_n=15)

q_tran.train(env, start_epoch=0, epochs_n=250)
print('Done training...')
exit(0)
for i in range(10):
    policy, checkpoint = q_tran.get_policy(epoch=i, cpu=True)
    print(checkpoint['epsilon'])
    def run_policy():
        # scoring and steps of episode
        observations = env.reset()
        scores = {a: 0 for a in env.agents}
        steps = 0
        env.aec_env.env.render(epoch=i, scores=scores)
        # Running a single episode
        dones = [False] * 4
        while not all(dones):
            actions = choose_actions(policy, observations)
            observations, rewards, dones, _ = env.step(actions)
            dones = dones.values()
            for a in rewards:
                scores[a] += rewards[a]
            steps += 1
            env.aec_env.env.render(epoch=i, scores=scores)

        env.aec_env.env.close()
        print(f'Epoch {i}')
        print(f'Episode done with total score: {sum(scores.values())} and total steps of: {steps}')
        print('Individual scores:')
        for a in scores:
            print(f'{a}: {scores[a]}')


    run_policy()
    # print(f'Starting eval of epoch {i}:')
    # eval_score, eval_steps, eval_scores = q_tran.evaluate(env, episodes_n=1000, debug=True)
    # print(f'Finished eval of epoch {i}:')
    # print(f'eval_score: {eval_score}')
    # print(f'eval_steps: {eval_steps}')
    # print('=== separate scores ===')
    # print(eval_scores)
    # filepath = f'out/epochs/epoch{i}.pt'
    # checkpoint['eval_score'] = eval_score
    # checkpoint['eval_steps'] = eval_steps
    # checkpoint['eval_scores'] = eval_scores
    # torch.save(checkpoint,filepath)

