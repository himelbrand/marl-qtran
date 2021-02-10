import torch
from pettingzoo.butterfly import knights_archers_zombies_v6 as kaz
from supersuit import black_death_v1
import matplotlib.pyplot as plt
from pettingzoo.utils.to_parallel import to_parallel
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
env = kaz.env(spawn_rate=20, num_knights=2, num_archers=2,
              killable_knights=True, killable_archers=True, line_death=True, pad_observation=True,
              max_cycles=900)
env = black_death_v1(env)
env = to_parallel(env)

# init Qtran
q_tran = QTran()

q_tran.train(env)
print('Done training...')
exit(0)
for i in range(11, 12):
    policy = q_tran.get_policy(epoch=i)


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
            dones = dones.values()
            for a in rewards:
                scores[a] += rewards[a]
            steps += 1
            env.aec_env.env.render()

        env.aec_env.env.close()
        print(f'Epoch {i}')
        print(f'Episode done with total score: {sum(scores.values())} and total steps of: {steps}')
        print('Individual scores:')
        for a in scores:
            print(f'{a}: {scores[a]}')


    run_policy()

