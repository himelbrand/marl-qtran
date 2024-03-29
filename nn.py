import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

print(f'Is CUDA available? {torch.cuda.is_available()}')


def to_tensor(x, using_cuda=False):
    transform = T.ToTensor()
    device = torch.device("cuda" if using_cuda else "cpu")
    if x.shape == (512, 512, 3):
        out = transform(x).unsqueeze(0).to(device)
    else:
        out = []
        for o in x:
            # print(o.shape)
            tensor = transform(o).unsqueeze(0).to(device)
            out.append(tensor)
        out = torch.cat(out)
    return out


class SingleQ(nn.Module):
    def __init__(self, actions_n=5, w=512, h=512, features_n=64, using_cuda=False):
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

        self.hidden_v = nn.Linear(linear_input_size, features_n)
        self.hidden_q = nn.Linear(linear_input_size, features_n)
        self.hidden = nn.Linear(features_n * 2, features_n)
        self.q = nn.Linear(features_n, actions_n)

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

    def forward(self, x):
        obs = x
        if not isinstance(obs, torch.Tensor):
            obs = to_tensor(obs, using_cuda=self.using_cuda)
        # if len(obs.shape) == 3:
        #     print(obs.shape)
        #     obs = obs.unsqueeze(0)
        obs = F.relu(self.bn1(self.conv1(obs)))
        obs = F.relu(self.bn2(self.conv2(obs)))
        obs = F.relu(self.bn3(self.conv3(obs)))
        hidden_v_features = F.silu(self.hidden_v(obs.view(obs.size(0), -1)))
        hidden_q_features = F.silu(self.hidden_q(obs.view(obs.size(0), -1)))
        hidden_inputs = torch.cat([hidden_q_features, hidden_v_features], dim=1)
        hidden_features = F.dropout(self.hidden(hidden_inputs), p=0.75)
        q = F.relu(self.q(hidden_features))
        return q, hidden_q_features, hidden_v_features


class JointV(nn.Module):
    def __init__(self, features_n=64, agents_n=4, using_cuda=False):
        super().__init__()
        self.hidden1 = nn.Linear(features_n, features_n // 2)
        self.hidden2 = nn.Linear(features_n // 2, features_n)
        self.v = nn.Linear(features_n, agents_n)
        if using_cuda:
            self.hidden1.cuda()
            self.hidden2.cuda()
            self.v.cuda()

    def forward(self, x):
        # x = torch.sigmoid(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.v(x))
        return torch.mean(x)


class JointQ(nn.Module):
    def __init__(self, features_n=64, agents_n=4, actions_n=5, using_cuda=False):
        super().__init__()
        self.hidden1 = nn.Linear(features_n, features_n // 2)
        self.hidden2 = nn.Linear(features_n // 2, features_n)
        self.q = nn.Linear(features_n, actions_n ** agents_n)
        if using_cuda:
            self.hidden1.cuda()
            self.hidden2.cuda()
            self.q.cuda()

    def forward(self, x):
        # x = torch.sigmoid(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.q(x))
        return x


class CentralNN(nn.Module):
    def __init__(self, agents_n=4, using_cuda=False):
        super().__init__()
        self.singleQs = [SingleQ(using_cuda=using_cuda) for _ in range(agents_n)]
        self.jointV = JointV(using_cuda=using_cuda)
        self.jointQ = JointQ(using_cuda=using_cuda)
        if using_cuda:
            self.jointV.cuda()
            self.jointQ.cuda()
            for s in self.singleQs:
                s.cuda()

    def forward(self, observations):
        singles_out = {agent: singleQ(observations[agent]) for singleQ, agent in zip(self.singleQs, observations)}
        jt_v_in = torch.stack([singles_out[a][2] for a in singles_out], dim=0).sum(dim=0)
        jt_q_in = torch.stack([singles_out[a][1] for a in singles_out], dim=0).sum(dim=0)
        q_jt = self.jointQ(jt_q_in)
        v_jt = self.jointV(jt_v_in)
        return {a: singles_out[a][0] for a in singles_out}, q_jt, v_jt
