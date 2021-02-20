import pickle
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
from nn import CentralNN, convert_to_one_hot_actions, to_tensor
import torch

Transition = namedtuple('Transition',
                        ('tau', 'u', 'r', 'tau_tag', 'done'))


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
    def __init__(self, buffer_size=5000, batch_size=16):
        self.replay_memory_capacity = buffer_size
        self.batch_size = batch_size
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)

    def add_to_memory(self, experience):
        # print(experience)
        self.replay_memory.append(Transition(*experience))

    def sample_from_memory(self):
        if len(self.replay_memory) >= self.batch_size * 10:
            ans = []
            for _ in range(self.batch_size):
                i = random.randint(0, len(self.replay_memory) - 1)
                ans.append(self.replay_memory[i])
            return ans
        else:
            return None

    def erase(self):
        self.replay_memory.popleft()

    def clear_buffer(self):
        self.replay_memory.clear()


class QTran:
    def __init__(self, piston_n=20, fresh_start=True, epsilon=1, epsilon_decay=0.99999, cpu=False,
                 gamma=0.95):
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
        for p in self.target_net.parameters(recurse=True):
            p.requires_grad = False
        for p in self.target_net.jointV.parameters():
            p.requires_grad = False
        for p in self.target_net.jointQ.parameters():
            p.requires_grad = False
        for sq in self.target_net.singleQs:
            for p in sq.parameters():
                p.requires_grad = False
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def choose_actions(self, observations, greedy=False):
        flip = random.random()
        if flip > self.epsilon or greedy:
            self.policy_net.eval()
            for sq in self.policy_net.singleQs:
                sq.eval()
            with torch.no_grad():
                q_values = {agent: singleQ(observations[agent])[0] for singleQ, agent in
                            zip(self.policy_net.singleQs, observations)}
                actions = {agent: q_values[agent].argmax().item() for agent in observations}
                self.policy_net.train()
                for sq in self.policy_net.singleQs:
                    sq.train()
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

        self.target_net.load_state_dict(self.policy_net.state_dict())
        for sq_target, sq in zip(self.target_net.singleQs, self.policy_net.singleQs):
            sq_target.load_state_dict(sq.state_dict())
        self.target_net.jointQ.load_state_dict(self.policy_net.jointQ.state_dict())
        self.target_net.jointV.load_state_dict(self.policy_net.jointV.state_dict())
        torch.cuda.empty_cache()
        steps = 0
        for epoch in range(start_epoch, start_epoch + epochs_n):
            print(f'Running Qtran epoch #{epoch}')
            if self.using_cuda:
                start_event.record()
            losses, steps_prev = self.train_epoch(env, epoch=epoch)
            steps += steps_prev
            if self.using_cuda:
                end_event.record()
                torch.cuda.synchronize()
            # if steps % 10000 == 0:
            #     self.target_net.load_state_dict(self.policy_net.state_dict())
            #     for sq_target, sq in zip(self.target_net.singleQs, self.policy_net.singleQs):
            #         sq_target.load_state_dict(sq.state_dict())
            #     self.target_net.jointQ.load_state_dict(self.policy_net.jointQ.state_dict())
            #     self.target_net.jointV.load_state_dict(self.policy_net.jointV.state_dict())
            epoch_duration_ms = start_event.elapsed_time(end_event)
            print(
                f'Finished epoch {epoch} - took {epoch_duration_ms / 1000} seconds or {epoch_duration_ms / 60000} minutes')
            print(f'current loss is {losses["loss"].item()}')
            print(f'Running evaluation...')
            if self.using_cuda:
                start_event.record()
            self.policy_net.eval()
            eval_score, eval_steps, eval_scores = self.evaluate(env, episodes_n=1)
            self.policy_net.train()
            if self.using_cuda:
                end_event.record()
                torch.cuda.synchronize()
            eval_duration_ms = start_event.elapsed_time(end_event)
            print(f'Finished evaluation....')
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
            print(f'took {eval_duration_ms / 1000} seconds or {eval_duration_ms / 60000} minutes')
            print(f'total average score from evaluation is {eval_score}')
            print(f'average step for episode in evaluation is {eval_steps}')
            self.save_model(epoch, epoch_data)

    def train_epoch(self, env, steps_n=1000, epoch=0, update_step=250):
        print(f'Starting epoch {epoch} training')
        start_env_step = torch.cuda.Event(enable_timing=True)
        end_env_step = torch.cuda.Event(enable_timing=True)
        start_get_batch = torch.cuda.Event(enable_timing=True)
        end_get_batch = torch.cuda.Event(enable_timing=True)
        start_backward = torch.cuda.Event(enable_timing=True)
        end_backward = torch.cuda.Event(enable_timing=True)
        start_opt_step = torch.cuda.Event(enable_timing=True)
        end_opt_step = torch.cuda.Event(enable_timing=True)
        start_target = torch.cuda.Event(enable_timing=True)
        end_target = torch.cuda.Event(enable_timing=True)
        start_policy = torch.cuda.Event(enable_timing=True)
        end_policy = torch.cuda.Event(enable_timing=True)
        start_act = torch.cuda.Event(enable_timing=True)
        end_act = torch.cuda.Event(enable_timing=True)
        # torch.cuda.empty_cache()
        episode = 0
        steps = 0
        episode_steps = 0
        observations = env.reset()
        observations = {a: to_tensor(observations[a], False) for a in observations}
        actions = {a: 1 for a in env.agents}
        losses = {'loss': None, 'loss_td': None, 'loss_opt': None, 'loss_nopt': None}
        while steps <= steps_n:
            episode += 1
            steps += episode_steps
            episode_steps = 0
            env_done = False
            while not env_done and steps + episode_steps <= steps_n:

                episode_steps += 1
                # start_act.record()
                actions = self.choose_actions(observations)
                # end_act.record()
                # start_env_step.record()
                observations_n, rewards, dones, _ = env.step(actions)

                # end_env_step.record()
                env_done = all(dones.values())
                if env_done and episode_steps < 899:
                    print(f'\n\n\nGot to left wall! at epoch {epoch}, E{episode}-S{episode_steps}\n\n\n')
                    self.epsilon = max(0.05, self.epsilon * 0.9)




                reward = sum(rewards.values())
                if env_done:
                    observations_n = observations
                else:
                    observations_n = {a: to_tensor(observations_n[a], False) for a in observations_n}
                self.replay_buffer.add_to_memory((observations, actions, reward,
                                                  observations_n, int(not env_done)))

                sample = self.replay_buffer.sample_from_memory()
                if sample:
                    # start_get_batch.record()
                    batch = Transition(*zip(*sample))
                    # end_get_batch.record()
                    # batch_idx = range(self.replay_buffer.batch_size)
                    tau_batch = {a: torch.cat([tau_tag[a] for tau_tag in batch.tau]).to(device=self.device) for a in
                                 observations}
                    actions_batch = batch.u  # torch.tensor([actions2index(x.values()) for x in batch.u], device=self.device)
                    reward_batch = torch.tensor(batch.r, device=self.device)
                    done_batch = torch.tensor(batch.done, device=self.device)
                    tau_tag_batch = {a: torch.cat([tau_tag[a] for tau_tag in batch.tau_tag]).to(device=self.device) for
                                     a in observations}
                    # start_target.record()
                    _, _, _, _, _, q_jt_target_opt = self.target_net(tau_tag_batch, actions_batch)
                    # end_target.record()
                    # u_bar_tag = torch.tensor(
                    #     actions2index(torch.stack([torch.argmax(q_singles_target[a], dim=1) for a in q_singles_target], dim=1)),
                    #     device=self.device)
                    # u_bar_tag = {a: torch.argmax(q_singles_target[a], dim=1).cpu().numpy() for a in q_singles_target}
                    # _, q_jt_target_opt, _ = self.target_net(tau_tag_batch, u_bar_tag)
                    y_dqn = reward_batch + self.gamma * done_batch * q_jt_target_opt
                    y_dqn = y_dqn.detach()
                    # start_policy.record()
                    q_singles, q_jt, q_jt_opt, v_jt, q_jt_tag_nopt, q_jt_tag_opt = self.policy_net(tau_batch,
                                                                                                   actions_batch)
                    # end_policy.record()
                    q_jt_hat = q_jt.detach()
                    q_jt_hat_opt = q_jt_opt.detach()
                    # u_bar = {a: torch.argmax(q_singles[a], dim=1).cpu().numpy() for a in q_singles}
                    # start_policy2.record()
                    # _, q_jt_opt, _, _, _ = self.policy_net(tau_batch, u_bar)
                    # end_policy2.record()

                    loss_td = torch.sum((q_jt - y_dqn) ** 2)
                    # q_jt_tag_opt = sum([torch.max(q_singles[a]) for a in q_singles])
                    loss_opt = torch.sum((q_jt_tag_opt - q_jt_hat_opt + v_jt) ** 2)
                    # q_jt_tag_nopt = torch.tensor(
                    #     [sum([q_singles[a][0][us[a]] for a in q_singles]) for us in batch.u], device=self.device)
                    loss_nopt = torch.sum(torch.min(q_jt_tag_nopt - q_jt_hat + v_jt,
                                                    torch.tensor(0).to(self.device)) ** 2)
                    loss = loss_td + loss_opt + loss_nopt
                    self.optimizer.zero_grad(set_to_none=True)
                    # start_backward.record()
                    loss.backward()
                    # end_backward.record()
                    count = 0
                    for param in self.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                        count += 1
                    for sq in self.policy_net.singleQs:
                        for param in sq.parameters():
                            param.grad.data.clamp_(-1, 1)
                    # start_opt_step.record()
                    self.optimizer.step()
                    # end_opt_step.record()
                    if steps + episode_steps > steps_n:
                        losses = {'loss': loss.detach(), 'loss_td': loss_td.detach(), 'loss_opt': loss_opt.detach(),
                                  'loss_nopt': loss_nopt.detach()}

                    # if steps + episode_steps % 10 == 0:
                    #     torch.cuda.synchronize()
                    #     print(
                    #         f'====== Episode #{episode} - step #{episode_steps} - total steps #{steps + episode_steps} =====')
                    #     print(f'choose action time: {start_act.elapsed_time(end_act)}ms')
                    #     print(f'env step time: {start_env_step.elapsed_time(end_env_step)}ms')
                    #     print(f'sample mini-batch time: {start_get_batch.elapsed_time(end_get_batch)}ms')
                    #     print(f'forward target time: {start_target.elapsed_time(end_target)}ms')
                    #     print(f'forward policy time: {start_policy.elapsed_time(end_policy)}ms')
                    #     print(f'backward loss time: {start_backward.elapsed_time(end_backward)}ms')
                    #     print(f'optimizer step time: {start_opt_step.elapsed_time(end_opt_step)}ms')
                    #     print(f'length of policy params is: {count}')
                if (steps + episode_steps) % update_step == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
                if env_done:
                    observations = env.reset()
                    observations = {a: to_tensor(observations[a], False) for a in observations}
                else:
                    observations = observations_n
        return losses, steps

    def load_model(self, epoch, path='out/epochs/epoch', training=True, epsilon=None):
        filepath = f'{path}{epoch}.pt'
        torch.cuda.empty_cache()
        checkpoint = torch.load(filepath, map_location='cpu')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        # for i, sq in enumerate(self.policy_net.singleQs):
        #     sq.load_state_dict(checkpoint[f'single_q_{i}_state_dict'])
        # self.policy_net.jointQ.load_state_dict(checkpoint[f'joint_q_state_dict'])
        # self.policy_net.jointV.load_state_dict(checkpoint[f'joint_v_state_dict'])
        loss = checkpoint['loss']
        loss_td = checkpoint['loss_td']
        loss_opt = checkpoint['loss_opt']
        loss_nopt = checkpoint['loss_nopt']
        losses = {'loss': loss, 'loss_td': loss_td, 'loss_opt': loss_opt, 'loss_nopt': loss_nopt}
        self.epsilon = checkpoint['epsilon'] if not epsilon else epsilon
        # self.epsilon = 0.6
        if training:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()
            self.target_net.eval()
        else:
            self.policy_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False
        for p in self.target_net.jointQ.parameters():
            p.requires_grad = False
        for p in self.target_net.jointV.parameters():
            p.requires_grad = False
        for sq in self.target_net.singleQs:
            for p in sq.parameters():
                p.requires_grad = False

        # with open(f'/content/drive/MyDrive/piston_buffers/epoch{epoch}.txt', 'rb') as rb_file:
        #     replay_buffer = pickle.load(rb_file)
        #     self.replay_buffer.clear_buffer()
        #     for exp in replay_buffer:
        #         self.replay_buffer.add_to_memory(exp)
        return losses, checkpoint

    def save_model(self, epoch, epoch_data):
        data = epoch_data.copy()
        data['optimizer_state_dict'] = self.optimizer.state_dict()
        data['model_state_dict'] = self.policy_net.state_dict()
        # for i, sq in enumerate(self.policy_net.singleQs):
        #     data[f'single_q_{i}_state_dict'] = sq.state_dict()
        # data[f'joint_q_state_dict'] = self.policy_net.jointQ.state_dict()
        # data[f'joint_v_state_dict'] = self.policy_net.jointV.state_dict()
        data['epoch'] = epoch
        filepath = f'out/epochs/epoch{epoch}.pt'
        torch.save(data, filepath)
        # with open(f'/content/drive/MyDrive/piston_buffers/epoch{epoch}.txt', 'wb') as rb_file:
        #     pickle.dump(self.replay_buffer.replay_memory, rb_file)

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
