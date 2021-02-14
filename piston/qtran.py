from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
from nn import CentralNN, convert_to_one_hot_actions
import torch

Transition = namedtuple('Transition',
                        ('tau', 'u', 'r', 'tau_tag'))


def plot(epochs, score, scores, steps, loss):
    plt.figure()
    plt.plot(epochs, score)
    plt.title('Epochs / Total score')
    plt.xlabel('Epochs')
    plt.ylabel('Total score')

    plt.figure()
    for a in scores:
        plt.plot(epochs, scores[a], label=a)
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


def actions2index(u):
    i = 0
    if len(u) > 15:
        ans = []
        for a in u:
            i = 0
            for n, e in zip(a, range(12)):
                i += n * 3 ** e
            ans.append(i)
        return ans
    else:
        for n, e in zip(u, range(12)):
            i += n * 3 ** e
        return i


class ReplayBuffer(object):
    def __init__(self, buffer_size=50000, batch_size=16):
        self.replay_memory_capacity = buffer_size
        self.batch_size = batch_size
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)

    def add_to_memory(self, experience):
        self.replay_memory.append(Transition(*experience))

    def sample_from_memory(self):
        if len(self.replay_memory) >= self.batch_size:
            return random.sample(self.replay_memory, self.batch_size)
        else:
            return None

    def erase(self):
        self.replay_memory.popleft()

    def clear_buffer(self):
        self.replay_memory.clear()


class QTran:
    def __init__(self, piston_n=20, fresh_start=True, epsilon=1, epsilon_decay=0.99995, cpu=False,
                 gamma=1):
        self.using_cuda = torch.cuda.is_available()
        self.device = torch.device("cpu" if not self.using_cuda or cpu else "cuda")
        self.agents_n = piston_n
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.theta = None
        self.t_theta = self.theta
        self.fresh_start = fresh_start
        self.policy_net = CentralNN(using_cuda=self.using_cuda, agents_n=piston_n)
        self.target_net = copy.deepcopy(self.policy_net)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())

    def choose_actions(self, observations, greedy=False):
        flip = random.random()
        if flip > self.epsilon or greedy:
            with torch.no_grad():
                q_values = {agent: singleQ(observations[agent])[0] for singleQ, agent in
                            zip(self.policy_net.singleQs, observations)}
                actions = {agent: q_values[agent].argmax().item() for agent in observations}
                return actions
        else:
            actions = {a: random.randint(0, 2) for a in observations}
            return actions

    def train(self, env, start_epoch=0, epochs_n=100):
        print('Running QTran training!')
        if self.using_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        if start_epoch > 0:
            self.load_model(start_epoch - 1)
        else:
            print('Started evaluating initial policy...')
            self.policy_net.eval()
            eval_score, eval_steps, eval_scores = self.evaluate(env, episodes_n=10)
            self.policy_net.train()

            epoch_data = {
                'epoch': 0,
                'eval_score': eval_score,
                'eval_steps': eval_steps,
                'eval_scores': eval_scores,
                'epoch_duration_ms': 0,
                'epsilon': self.epsilon,
                'loss': torch.tensor(0),
                'loss_td': torch.tensor(0),
                'loss_opt': torch.tensor(0),
                'loss_nopt': torch.tensor(0)
            }
            self.save_model(0, epoch_data)
            print(f'Finished evaluating initial policy... average total score was: {eval_score}')
            start_epoch = 1
        torch.cuda.empty_cache()
        for epoch in range(start_epoch, start_epoch + epochs_n):
            print(f'Running Qtran epoch #{epoch}')
            if self.using_cuda:
                start_event.record()
            losses = self.train_epoch(env, epoch=epoch)
            if self.using_cuda:
                end_event.record()
                torch.cuda.synchronize()
            epoch_duration_ms = start_event.elapsed_time(end_event)
            print(f'Running evaluation...')
            self.policy_net.eval()
            eval_score, eval_steps, eval_scores = self.evaluate(env, episodes_n=1)
            self.policy_net.train()
            epoch_data = {
                'epoch': epoch,
                'eval_score': eval_score,
                'eval_steps': eval_steps,
                'eval_scores': eval_scores,
                'epoch_duration_ms': epoch_duration_ms,
                'epsilon': self.epsilon
            }
            for loss_name in losses:
                epoch_data[loss_name] = losses[loss_name]
            print(
                f'Finished epoch {epoch} - took {epoch_duration_ms / 1000} seconds or {epoch_duration_ms / 60000} minutes - current loss is {losses["loss"].item()} - total average score from evaluation is {eval_score} - average step for episode in evaluation is {eval_steps}')
            self.save_model(epoch, epoch_data)

    def train_epoch(self, env, steps_n=1000, epoch=0, update_step=250):
        print(f'Starting epoch {epoch} training')
        episode = 0
        steps = 0
        episode_steps = 0
        observations = env.reset()
        actions = {a: 1 for a in env.agents}
        losses = {'loss': None, 'loss_td': None, 'loss_opt': None, 'loss_nopt': None}
        while steps <= steps_n:
            episode += 1
            steps += episode_steps
            episode_steps = 0
            env_done = False
            while not env_done and steps + episode_steps <= steps_n:
                episode_steps += 1
                actions = self.choose_actions(observations)
                observations_n, rewards, dones, _ = env.step(actions)
                env_done = all(dones.values())
                reward = sum(rewards.values())
                if not env_done:
                    self.replay_buffer.add_to_memory((observations, actions, reward, observations_n))
                sample = self.replay_buffer.sample_from_memory()
                if sample:
                    batch = Transition(*zip(*sample))
                    # batch_idx = range(self.replay_buffer.batch_size)
                    tau_batch = {a: np.array([tau_tag[a] for tau_tag in batch.tau]) for a in observations}
                    actions_batch = batch.u # torch.tensor([actions2index(x.values()) for x in batch.u], device=self.device)
                    reward_batch = torch.tensor(batch.r, device=self.device)
                    tau_tag_batch = {a: np.array([tau_tag[a] for tau_tag in batch.tau_tag]) for a in observations}
                    q_singles_target, q_jt_target, _, _, q_jt_target_opt = self.target_net(tau_tag_batch, actions_batch)
                    # u_bar_tag = torch.tensor(
                    #     actions2index(torch.stack([torch.argmax(q_singles_target[a], dim=1) for a in q_singles_target], dim=1)),
                    #     device=self.device)
                    # u_bar_tag = {a: torch.argmax(q_singles_target[a], dim=1).cpu().numpy() for a in q_singles_target}
                    # _, q_jt_target_opt, _ = self.target_net(tau_tag_batch, u_bar_tag)
                    y_dqn = reward_batch + self.gamma * q_jt_target_opt
                    q_singles, q_jt, v_jt, q_jt_tag_nopt, q_jt_tag_opt = self.policy_net(tau_batch, actions_batch)
                    q_jt_hat = q_jt.detach()
                    u_bar = {a: torch.argmax(q_singles[a], dim=1).cpu().numpy() for a in q_singles}
                    _, q_jt_opt, _ = self.policy_net(tau_batch, u_bar)
                    q_jt_hat_opt = q_jt_opt.detach()
                    loss_td = torch.sum((q_jt - y_dqn) ** 2)
                    # q_jt_tag_opt = sum([torch.max(q_singles[a]) for a in q_singles])
                    loss_opt = torch.sum((q_jt_tag_opt - q_jt_hat_opt + v_jt) ** 2)
                    # q_jt_tag_nopt = torch.tensor(
                    #     [sum([q_singles[a][0][us[a]] for a in q_singles]) for us in batch.u], device=self.device)
                    loss_nopt = torch.sum(torch.min(q_jt_tag_nopt - q_jt_hat + v_jt,
                                                    torch.tensor(0).to(self.device)) ** 2)
                    loss = loss_td + loss_opt + loss_nopt
                    self.optimizer.zero_grad()
                    loss.backward()
                    # for param in self.policy_net.parameters():
                    #     param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
                    losses = {'loss': loss, 'loss_td': loss_td, 'loss_opt': loss_opt, 'loss_nopt': loss_nopt}
                if (steps + episode_steps) % update_step == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
        return losses

    def load_model(self, epoch, path='out/epochs/epoch', training=True):
        filepath = f'{path}{epoch}.pt'
        torch.cuda.empty_cache()
        checkpoint = torch.load(filepath, map_location='cpu')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
        loss_td = checkpoint['loss_td']
        loss_opt = checkpoint['loss_opt']
        loss_nopt = checkpoint['loss_nopt']
        losses = {'loss': loss, 'loss_td': loss_td, 'loss_opt': loss_opt, 'loss_nopt': loss_nopt}
        self.epsilon = checkpoint['epsilon']
        if training:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()
            self.target_net.eval()
        else:
            self.policy_net.eval()
        return losses, checkpoint

    def save_model(self, epoch, epoch_data):
        data = epoch_data.copy()
        data['optimizer_state_dict'] = self.optimizer.state_dict()
        data['model_state_dict'] = self.policy_net.state_dict()
        data['epoch'] = epoch
        filepath = f'out/epochs/epoch{epoch}.pt'
        torch.save(data, filepath)

    def get_policy(self, training=False, epoch=-1, cpu=False):
        checkpoint = None
        if epoch >= 0:
            _, checkpoint = self.load_model(epoch, training=training)
        policy = CentralNN(agents_n=self.agents_n)
        policy.load_state_dict(self.policy_net.state_dict())
        device = torch.device("cpu" if not self.using_cuda or cpu else "cuda")
        policy.to(device)
        if training:
            policy.train()
        else:
            policy.eval()
        return policy, checkpoint

    def evaluate(self, env, episodes_n=10, debug=False):
        # scoring and steps of episode
        env.reset()
        score = []
        scores = {a: [] for a in env.agents}
        steps = []
        for episode in range(episodes_n):
            # Reset env + initial render
            observations = env.reset()
            curr_scores = {a: 0 for a in env.agents}
            actions = {a: 1 for a in env.agents}
            curr_steps = 0
            if debug and episode == episodes_n - 1:
                env.aec_env.env.render()
            # dones = [False] * 4
            env_done = False
            # Running a single episode
            while not env_done:
                actions = self.choose_actions(observations, greedy=True)
                # try:
                observations, rewards, dones, _ = env.step(actions)
                env_done = all(dones.values())
                # except Exception as e:
                #     print(actions)
                #     print(e.tr)
                #     break
                for a in rewards:
                    curr_scores[a] += rewards[a]
                curr_steps += 1
                if debug and episode == episodes_n - 1:
                    env.aec_env.env.render()
            if debug and episode == episodes_n - 1:
                env.aec_env.env.render()
                env.aec_env.env.close()
            steps.append(curr_steps)
            score.append(sum(curr_scores.values()))
            for a in curr_scores:
                scores[a].append(curr_scores[a])
            # print results of episode
            if debug and episode % 100 == 0:
                print(f'Episode {episode} done with total score: {sum(curr_scores.values())}')
        scores = {a: np.array(scores[a]).mean() for a in scores}
        return np.array(score).mean(), np.array(steps).mean(), scores
