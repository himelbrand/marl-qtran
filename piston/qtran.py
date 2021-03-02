from collections import deque, namedtuple
import random
import numpy as np
import copy
from nn import CentralNN, to_tensor
import torch

Transition = namedtuple('Transition',
                        ('tau', 'u', 'r', 'tau_tag', 'done'))


def deep_eq(_v1, _v2):
    import operator

    def _deep_dict_eq(d1, d2):
        k1 = sorted(d1.keys())
        k2 = sorted(d2.keys())
        if k1 != k2:  # keys should be exactly equal
            return False
        return sum(deep_eq(d1[k], d2[k]) for k in k1) == len(k1)

    def _deep_iter_eq(l1, l2):
        if len(l1) != len(l2):
            return False
        return sum(deep_eq(v1, v2) for v1, v2 in zip(l1, l2)) == len(l1)

    op = operator.eq
    c1, c2 = (_v1, _v2)

    # guard against strings because they are also iterable
    # and will consistently cause a RuntimeError (maximum recursion limit reached)
    for t in [str]:
        if isinstance(_v1, t):
            break
    else:
        if isinstance(_v1, dict):
            op = _deep_dict_eq
        else:
            try:
                c1, c2 = (list(iter(_v1)), list(iter(_v2)))
            except TypeError:
                c1, c2 = _v1, _v2
            else:
                op = _deep_iter_eq

    return op(c1, c2)


class ReplayBuffer(object):
    def __init__(self, buffer_size=10000, batch_size=25):
        self.replay_memory_capacity = buffer_size
        self.batch_size = batch_size
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)

    def add_to_memory(self, experience):
        self.replay_memory.append(Transition(*experience))

    def sample_from_memory(self):
        if len(self.replay_memory) >= self.batch_size:
            ans = []
            for _ in range(self.batch_size-1):
                i = random.randint(0, len(self.replay_memory) - 1)
                ans.append(self.replay_memory[i])
            ans.append(self.replay_memory[-1])
            return ans
        else:
            return None

    def erase(self):
        self.replay_memory.popleft()

    def clear_buffer(self):
        self.replay_memory.clear()


class QTran:
    def __init__(self, piston_n=20, fresh_start=True, epsilon=1, epsilon_decay=0.99999, cpu=False,
                 gamma=0.5, debug=False):
        self.using_cuda = torch.cuda.is_available()
        self.debug = debug
        self.device = torch.device("cpu" if not self.using_cuda or cpu else "cuda")
        self.agents_n = piston_n
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.theta = None
        self.t_theta = self.theta
        self.fresh_start = fresh_start
        self.policy_net = CentralNN(using_cuda=self.using_cuda, agents_n=piston_n)
        self.target_net = copy.deepcopy(self.policy_net)
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
                if not greedy:
                    self.policy_net.train()
                    for sq in self.policy_net.singleQs:
                        sq.train()
                return actions
        else:
            actions = {a: random.randint(0, 2) for a in observations}
            return actions

    def train(self, env, start_epoch=0, epochs_n=100):
        print('Running QTran training!')
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if start_epoch > 0:
            self.load_model(start_epoch - 1)
        else:
            print('Started evaluating initial policy...')
            self.policy_net.eval()
            eval_score, eval_steps, eval_scores = self.evaluate(env, episodes_n=20)
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
            start_event.record()
            losses, steps_prev = self.train_epoch(env, epoch=epoch)
            steps += steps_prev
            end_event.record()
            torch.cuda.synchronize()
            epoch_duration_ms = start_event.elapsed_time(end_event)
            print(
                f'Finished epoch {epoch} - took {epoch_duration_ms / 1000} seconds or {epoch_duration_ms / 60000} minutes')
            print(f'Running evaluation...')
            if self.using_cuda:
                start_event.record()
            self.policy_net.eval()
            eval_score, eval_steps, eval_scores = self.evaluate(env, episodes_n=15)
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

    def train_epoch(self, env, steps_n=1000, epoch=0, update_step=50):
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
            env.random_drop = random.random() < 0.3
            env.random_rotate = random.random() < 0.3
            while not env_done and steps + episode_steps <= steps_n:

                episode_steps += 1
                start_act.record()
                actions = self.choose_actions(observations)
                end_act.record()
                start_env_step.record()
                observations_n, rewards, dones, _ = env.step(actions)

                end_env_step.record()
                env_done = all(dones.values())
                if env_done and episode_steps < 900:
                    print(f'\n\n\nGot to left wall! at epoch {epoch}, E{episode}-S{episode_steps}\n\n\n')

                reward = sum(rewards.values())
                if env_done:
                    observations_n = observations
                else:
                    observations_n = {a: to_tensor(observations_n[a], False) for a in observations_n}
                self.replay_buffer.add_to_memory((observations, actions, reward,
                                                  observations_n, int(not (env_done and episode_steps < 900))))
                if reward > 0:
                    for _ in range(20):
                        self.replay_buffer.add_to_memory((observations, actions, reward,
                                                          observations_n, int(not env_done)))
                sample = self.replay_buffer.sample_from_memory()
                if sample:
                    start_get_batch.record()
                    batch = Transition(*zip(*sample))
                    end_get_batch.record()

                    tau_batch = {a: torch.cat([tau_tag[a] for tau_tag in batch.tau]).to(device=self.device) for a in
                                 observations}
                    actions_batch = batch.u
                    reward_batch = torch.tensor(batch.r, device=self.device)
                    done_batch = torch.tensor(batch.done, device=self.device)
                    tau_tag_batch = {a: torch.cat([tau_tag[a] for tau_tag in batch.tau_tag]).to(device=self.device) for
                                     a in observations}
                    start_target.record()
                    _, _, _, _, _, q_jt_target_opt = self.target_net(tau_tag_batch, actions_batch)
                    end_target.record()

                    y_dqn = reward_batch + self.gamma * done_batch * q_jt_target_opt
                    y_dqn = y_dqn.detach()
                    start_policy.record()
                    q_singles, q_jt, q_jt_opt, v_jt, q_jt_tag_nopt, q_jt_tag_opt = self.policy_net(tau_batch,
                                                                                                   actions_batch)
                    end_policy.record()
                    q_jt_hat = q_jt.detach()
                    q_jt_hat_opt = q_jt_opt.detach()

                    loss_td = torch.sum((q_jt - y_dqn) ** 2)
                    loss_opt = torch.sum((q_jt_tag_opt - q_jt_hat_opt + v_jt) ** 2)
                    loss_nopt = torch.sum(torch.min(q_jt_tag_nopt - q_jt_hat + v_jt,
                                                    torch.tensor(0).to(self.device)) ** 2)

                    loss = loss_td + loss_opt + loss_nopt
                    self.optimizer.zero_grad(set_to_none=True)
                    start_backward.record()
                    loss.backward()
                    end_backward.record()
                    count = 0
                    for param in self.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                        count += 1
                    for sq in self.policy_net.singleQs:
                        for param in sq.parameters():
                            param.grad.data.clamp_(-1, 1)
                    for param in self.policy_net.jointQ.parameters():
                        param.grad.data.clamp_(-1, 1)
                    for param in self.policy_net.jointV.parameters():
                        param.grad.data.clamp_(-1, 1)
                    start_opt_step.record()
                    self.optimizer.step()
                    end_opt_step.record()
                    if steps + episode_steps > steps_n:
                        losses = {'loss': loss.detach(), 'loss_td': loss_td.detach(), 'loss_opt': loss_opt.detach(),
                                  'loss_nopt': loss_nopt.detach()}

                    if steps + episode_steps % 100 == 0 and self.debug:
                        torch.cuda.synchronize()
                        print(
                            f'====== Episode #{episode} - step #{episode_steps} - total steps #{steps + episode_steps} =====')
                        print(f'choose action time: {start_act.elapsed_time(end_act)}ms')
                        print(f'env step time: {start_env_step.elapsed_time(end_env_step)}ms')
                        print(f'sample mini-batch time: {start_get_batch.elapsed_time(end_get_batch)}ms')
                        print(f'forward target time: {start_target.elapsed_time(end_target)}ms')
                        print(f'forward policy time: {start_policy.elapsed_time(end_policy)}ms')
                        print(f'backward loss time: {start_backward.elapsed_time(end_backward)}ms')
                        print(f'optimizer step time: {start_opt_step.elapsed_time(end_opt_step)}ms')

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
        loss = checkpoint['loss']
        loss_td = checkpoint['loss_td']
        loss_opt = checkpoint['loss_opt']
        loss_nopt = checkpoint['loss_nopt']
        losses = {'loss': loss, 'loss_td': loss_td, 'loss_opt': loss_opt, 'loss_nopt': loss_nopt}
        self.epsilon = checkpoint['epsilon'] if not epsilon else epsilon
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
        rd = env.aec_env.env.env.random_drop
        rr = env.aec_env.env.env.random_rotate
        env.aec_env.env.env.random_drop = False
        env.aec_env.env.env.random_rotate = False
        # scoring and steps of episode
        env.reset()
        score = []
        scores = {a: [] for a in env.agents}
        steps = []
        score_alt = []
        for episode in range(episodes_n):
            # Reset env + initial render
            observations = env.reset()
            prev_obs = observations
            curr_scores = {a: 0 for a in env.agents}
            curr_steps = 0
            stuck_count = 0
            stuck = False
            if debug and episode == episodes_n - 1:
                env.aec_env.env.render()
            env_done = False

            # Running a single episode
            while not env_done:
                actions = self.choose_actions(observations, greedy=True)

                observations, rewards, dones, _ = env.step(actions)

                if not stuck and all([(observations[a] == prev_obs[a]).all() for a in observations]):
                    stuck_count += 1
                else:
                    stuck_count = 0

                if not stuck:
                    prev_obs = observations

                env_done = all(dones.values())

                for a in rewards:
                    curr_scores[a] += rewards[a]
                if stuck_count == 4:
                    stuck = True
                    env_done = True
                    score_alt.append(sum(curr_scores.values())+(curr_steps-899)*0.8)
                    for a in rewards:
                        curr_scores[a] += (curr_steps-899)*0.8*0.1
                curr_steps += 1
                if debug and episode == episodes_n - 1:
                    env.aec_env.env.render()
            if debug and episode == episodes_n - 1:
                env.aec_env.env.render()
                env.aec_env.env.close()

            if not stuck:
                score.append(sum(curr_scores.values()))
                steps.append(curr_steps)
            else:
                score.append(score_alt[-1])
                steps.append(900)

            for a in curr_scores:
                scores[a].append(curr_scores[a])
            # print results of episode
            if debug and episode == episodes_n - 1:
                print(f'Episode {episode} done with total score: {sum(curr_scores.values())}')
        scores = {a: np.array(scores[a]).mean() for a in scores}
        env.aec_env.env.env.random_drop = rd
        env.aec_env.env.env.random_rotate = rr
        return np.array(score).mean(), np.array(steps).mean(), scores
