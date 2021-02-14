import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

print(f'Is CUDA available? {torch.cuda.is_available()}')


def to_tensor(x, using_cuda=False):
    transform = T.ToTensor()
    device = torch.device("cuda" if using_cuda else "cpu")
    if x.shape == (457, 120, 3):
        out = transform(x).unsqueeze(0).to(device)
    else:
        out = []
        for o in x:
            # print(o.shape)
            tensor = transform(o).unsqueeze(0).to(device)
            out.append(tensor)
        out = torch.cat(out)
    return out


def convert_to_one_hot_actions(actions, device, actions_n=3):
    new_actions = {}

    if isinstance(actions, dict):
        actions_list = [actions]
    else:
        actions_list = actions
    for actions in actions_list:
        for a in actions:
            if isinstance(actions[a], int):
                x = [0 for _ in range(actions_n)]
                x[actions[a]] = 1
                if a not in new_actions:
                    new_actions[a] = [x]
                else:
                    new_actions[a].append(x)
            else:
                for action in actions[a]:
                    x = [0 for _ in range(actions_n)]
                    x[action] = 1
                    if a not in new_actions:
                        new_actions[a] = [x]
                    else:
                        new_actions[a].append(x)
    for a in new_actions:
        new_actions[a] = torch.tensor(new_actions[a], device=device)
    return new_actions


class SingleQ(nn.Module):
    def __init__(self, actions_n=3, w=120, h=457, features_n=32, using_cuda=False):
        super(SingleQ, self).__init__()
        self.using_cuda = using_cuda
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = conv_h * conv_w * 32

        self.hidden = nn.Linear(linear_input_size, features_n)
        self.q = nn.Linear(features_n, actions_n)

        self.hidden_v = nn.Linear(linear_input_size + actions_n, features_n)
        self.hidden_q = nn.Linear(linear_input_size + actions_n, features_n)

        if using_cuda:
            self.conv1.cuda()
            self.bn1.cuda()
            self.conv2.cuda()
            self.bn2.cuda()
            self.conv3.cuda()
            self.bn3.cuda()
            self.hidden_v.cuda()
            self.hidden_q.cuda()
            self.q.cuda()

    def forward(self, obs, action=None):
        if not isinstance(obs, torch.Tensor):
            obs = to_tensor(obs, using_cuda=self.using_cuda)
        if not isinstance(action, torch.Tensor) and action is not None:
            action = torch.tensor([action])
        obs = F.relu(self.bn1(self.conv1(obs)))
        obs = F.relu(self.bn2(self.conv2(obs)))
        obs = F.relu(self.bn3(self.conv3(obs)))
        hidden_features = F.dropout(self.hidden(obs.view(obs.size(0), -1)), p=0.75)
        q_values = self.q(hidden_features)
        if action is not None:
            opt_actions = torch.argmax(q_values, 1, keepdim=True).cpu()
            one_hot_opt = torch.FloatTensor(q_values.shape)
            one_hot_opt.zero_()
            for i, opt in enumerate(opt_actions):
                one_hot_opt[i][opt] = 1
            if self.using_cuda:
                one_hot_opt = one_hot_opt.to(device='cuda')
            hidden_v_features = F.silu(
                self.hidden_v(torch.cat([one_hot_opt * q_values, obs.view(obs.size(0), -1)], dim=1)))
            hidden_q_features = F.silu(self.hidden_q(torch.cat([action * q_values, obs.view(obs.size(0), -1)], dim=1)))
            q = torch.sum(action * q_values, dim=1, keepdim=True)
            qmax = torch.max(q_values, dim=1, keepdim=True).values
        else:
            hidden_v_features = None
            hidden_q_features = None
            q = None
            qmax = None

        return q_values, hidden_q_features, hidden_v_features, q, qmax


class JointV(nn.Module):
    def __init__(self, features_n=32, using_cuda=False):
        super().__init__()
        self.hidden1 = nn.Linear(features_n, features_n // 2)
        self.hidden2 = nn.Linear(features_n // 2, features_n)
        self.v = nn.Linear(features_n, 1)
        if using_cuda:
            self.hidden1.cuda()
            self.hidden2.cuda()
            self.v.cuda()

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.v(x)
        return x


class JointQ(nn.Module):
    def __init__(self, features_n=32, using_cuda=False):
        super().__init__()
        self.hidden1 = nn.Linear(features_n, features_n // 2)
        self.hidden2 = nn.Linear(features_n // 2, features_n)
        self.q = nn.Linear(features_n, 1)
        if using_cuda:
            self.hidden1.cuda()
            self.hidden2.cuda()
            self.q.cuda()

    def forward(self, x):
        # x = torch.sigmoid(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.q(x)
        return x


class CentralNN(nn.Module):
    def __init__(self, agents_n=20, using_cuda=False):
        super().__init__()
        self.singleQs = [SingleQ(using_cuda=using_cuda) for _ in range(agents_n)]
        self.jointV = JointV(using_cuda=using_cuda)
        self.jointQ = JointQ(using_cuda=using_cuda)
        self.device = torch.device('cuda' if using_cuda else 'cpu')
        if using_cuda:
            self.jointV.cuda()
            self.jointQ.cuda()
            for s in self.singleQs:
                s.cuda()

    def forward(self, observations, actions=None):
        one_hot_actions = convert_to_one_hot_actions(actions, self.device) if actions is not None else {a: None for a in
                                                                                                        observations}
        singles_out = {agent: singleQ(observations[agent], one_hot_actions[agent]) for singleQ, agent in
                       zip(self.singleQs, observations)}
        jt_v_in = torch.stack([singles_out[a][2] for a in singles_out], dim=0).sum(dim=0)
        jt_q_in = torch.stack([singles_out[a][1] for a in singles_out], dim=0).sum(dim=0)
        q_jt = self.jointQ(jt_q_in)
        v_jt = self.jointV(jt_v_in)
        # print([singles_out[a][4] for a in singles_out])
        q_jt_prime = torch.stack([singles_out[a][3] for a in singles_out], dim=1).sum(dim=1)
        q_jt_prime_opt = torch.stack([singles_out[a][4] for a in singles_out], dim=1).sum(dim=1)
        # print(q_jt_prime)
        # print(q_jt_prime_opt)
        return {a: singles_out[a][0] for a in singles_out}, q_jt, v_jt, q_jt_prime, q_jt_prime_opt
