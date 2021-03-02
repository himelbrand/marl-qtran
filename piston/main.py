import torch
from pettingzoo.butterfly import pistonball_v3 as pistonball
import matplotlib.pyplot as plt
from qtran import QTran
import time


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


def plot_all(s_epoch=0, e_epoch=310):
    epochs = []
    score = []
    scores = []
    steps = []
    loss = []
    for epoch in range(s_epoch, e_epoch + 1):
        checkpoint = load_checkpoint(epoch)
        epochs.append(epoch)
        score.append(checkpoint['eval_score'])
        scores.append(checkpoint['eval_scores'])
        steps.append(checkpoint['eval_steps'])
        loss.append(checkpoint['loss'].item())
    plot(epochs, score, scores, steps, loss)


def choose_actions(policy_net, observations):
    with torch.no_grad():
        q_values = policy_net(observations)[0]
        actions = {agent: torch.argmax(q_values[agent]).item() for agent in observations}
        return actions


#  Init parallel env
env = pistonball.parallel_env(n_pistons=10, local_ratio=0.2, time_penalty=-0.1, continuous=False, random_drop=False,
                              random_rotate=False, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5,
                              max_cycles=900)

q_tran = QTran(piston_n=10)


# q_tran.train(env, start_epoch=7, epochs_n=250)


def evaluate_epoch(epoch=245, episodes_n=50):
    q_tran.get_policy(epoch=epoch, cpu=True)
    print(f'Starting eval of epoch {epoch}:')
    start = time.time()
    eval_score, eval_steps, eval_scores = q_tran.evaluate(env, episodes_n=episodes_n, debug=False)
    end = time.time()
    print(f'Finished eval of epoch {epoch} ({episodes_n} episodes):')
    print(f'This run took {end - start} seconds')
    print(f'eval_score: {eval_score}')
    print(f'eval_steps: {eval_steps}')
    print('=== separate scores ===')
    print(eval_scores)


def train(start_epoch=0, learn_epochs=250):
    q_tran.train(env, start_epoch=start_epoch, epochs_n=learn_epochs)
    print('done training...')


def run_policy(epoch=245):
    # scoring and steps of episode
    policy, _ = q_tran.get_policy(epoch=epoch, cpu=True)
    observations = env.reset()
    scores = {a: 0 for a in env.agents}
    steps = 0
    env.aec_env.env.render()
    # Running a single episode
    env_done = False
    while not env_done:
        actions = choose_actions(policy, observations)
        observations, rewards, dones, _ = env.step(actions)
        env_done = all(dones.values())
        for a in rewards:
            scores[a] += rewards[a]
        steps += 1
        env.aec_env.env.render()
    env.aec_env.env.close()
    print(f'Epoch {epoch}')
    print(f'Episode done with total score: {sum(scores.values())} and total steps of: {steps}')
    print('Individual scores:')
    for a in scores:
        print(f'{a}: {scores[a]}')
    return sum(scores.values())


def prompt_menu(title, options):
    num2option = list(options.keys())
    print(title)
    for i, k in enumerate(options):
        print(f'({i}) - {k}')
    while True:
        ans = input('\nPlease enter one of the numeric keys: ')
        try:
            i = int(ans)
            key = num2option[i]
            return options[key]
        except ValueError:
            print(f'Invalid choice: {ans}.\nTry again')


if __name__ == '__main__':
    ans = prompt_menu('Choose option', {'Train': 0, 'Evaluate': 1, 'Run simulation': 2})
    if ans == 0:
        ans = prompt_menu('Start from scratch?', {'Yes': 0, 'No': 1})
        if ans == 0:
            train(start_epoch=ans)
        else:
            while True:
                x = input('enter starting epoch: ')
                try:
                    x = int(x)
                    break
                except:
                    print('not a valid option! max is last_epoch+1 min is 0')
            train(start_epoch=x)
    elif ans == 1:
        ans = prompt_menu('use default epoch?', {'Yes': 0, 'No': 1})
        if ans == 0:
            evaluate_epoch()
        else:
            while True:
                x = input('enter epoch for evaluation: ')
                try:
                    x = int(x)
                    break
                except:
                    print('not a valid option! max is last_epoch+1 min is 0')
            evaluate_epoch(epoch=x)
    else:
        ans = prompt_menu('use default epoch?', {'Yes': 0, 'No': 1})
        if ans == 0:
            run_policy()
        else:
            while True:
                x = input('enter epoch for evaluation: ')
                try:
                    x = int(x)
                    break
                except:
                    print('not a valid option! max is last_epoch+1 min is 0')
            run_policy(epoch=x)
