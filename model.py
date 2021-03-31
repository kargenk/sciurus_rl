import pfrl
from pfrl.nn.lmbda import Lambda
import torch
from torch import distributions, nn


class ConvBNPool(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
        )

        # 初期値はxavierを用いる
        torch.nn.init.xavier_uniform_(self.net[0].weight)

    def forward(self, x):
        return self.net(x)


class SACActor(nn.Module):
    """
    SACのActor(方策)ネットワーク.
    入力として状態(観測)を受け取り,ガウス分布の平均と標準偏差の対数を出力する.
    """
    def __init__(self, obs_channels: int, num_actions: int):
        super().__init__()
        self.action_size = num_actions
        self.conv_bnp1 = ConvBNPool(obs_channels, 64, 3, padding=1)
        self.conv_bnp2 = ConvBNPool(64, 128, 3, padding=1)
        self.conv_bnp3 = ConvBNPool(128, 256, 3, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256 * 6 * 6, 512)
        self.linear2 = nn.Linear(512, 2 * num_actions)

        # 初期値はxavierを用いる
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def squashed_diagonal_gaussian_head(self, x):
        """ref: https://github.com/pfnet/pfrl/blob/master/examples/atlas/train_soft_actor_critic_atlas.py"""
        assert x.shape[-1] == self.action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    def forward(self, states):
        h = states
        h = self.conv_bnp1(h)  # [N, obs_channels(maybe 3), 48, 48] -> [N, 64, 24, 24]
        h = self.conv_bnp2(h)  # [N, 3, 24, 24] -> [N, 128, 12, 12]
        h = self.conv_bnp3(h)  # [N, 128, 12, 12] -> [N, 256, 6, 6]
        h = self.flatten(h)    # [N, 256, 6, 6] -> [N, 256*6*6]
        h = self.linear1(h)    # [N, 256*6*6] -> [N, 512]
        h = self.linear2(h)    # [N, 512] -> [N, 2 * num_actions]
        out = Lambda(self.squashed_diagonal_gaussian_head)(h)

        return out


class SACCritic(nn.Module):
    """
    SACのCritic(価値関数)ネットワーク, ソフト状態行動価値を関数近似する.
    入力として状態と行動を受け取り,状態行動価値を出力する.
    """
    def __init__(self, obs_channels: int, num_actions: int):
        super().__init__()

        # for observation(image)
        self.conv_bnp1 = ConvBNPool(obs_channels, 64, 3, padding=1)
        self.conv_bnp2 = ConvBNPool(64, 128, 3, padding=1)
        self.conv_bnp3 = ConvBNPool(128, 256, 3, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256 * 6 * 6, 512)
        self.linear2 = nn.Linear(512, 64)

        self.concat_obs_action = pfrl.nn.ConcatObsAndAction()
        self.linear3 = nn.Linear(64 + num_actions, 128)
        self.linear4 = nn.Linear(128, 128)
        self.linear5 = nn.Linear(128, 1)

        # 初期値はxavierを用いる
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        torch.nn.init.xavier_uniform_(self.linear5.weight)

    def forward(self, states, actions):
        h = self.conv_bnp1(states)
        h = self.conv_bnp2(h)
        h = self.conv_bnp3(h)
        h = self.flatten(h)
        h = self.linear1(h)
        h = self.linear2(h)
        states_action_concat = self.concat_obs_action((h.view(-1), actions))
        h = self.linear3(states_action_concat)
        h = self.linear4(h)
        out = self.linear5(h)

        return out


if __name__ == '__main__':
    import numpy as np

    # 状態のチャネル数，行動の次元数
    obs_channels = 3
    n_actions = 9
    actions = np.zeros((n_actions))

    # policy(戦略)の定義
    policy = SACActor(obs_channels, n_actions)
    test_obs = torch.from_numpy(np.zeros((3, 48, 48))).unsqueeze(dim=0).float()
    print(policy(test_obs))

    # Q関数(価値関数)の定義
    q_func1 = SACCritic(obs_channels, n_actions)
    q_func2 = SACCritic(obs_channels, n_actions)
    test_action = torch.tensor(np.ones_like(actions)).float()
    print(q_func1(test_obs, test_action))
    print(q_func2(test_obs, test_action))
