import torch

class AGG_GNN():
    def __init__(self, args):
        self.args = args
        self.H_stack = torch.zeros((args.user_num, args.user_num * args.K_hop))
        self.x_stack = torch.zeros((args.user_num * args.K_hop, args.K_hop))

        # upon request
        self.policyInputDim = args.K_hop
        self.policyOutputDim = 1

    def update(self, power, user_rates, channels):
        pass

    def reset(self):
        pass

    def brew_input(self, channels, states):
        # Channels dim: (user_num, user_num), states dim: (user_num)
        # based on Wang, Z., Eisen, M., & Ribeiro, A. (2022). Learning decentralized wireless resource allocations with graph neural networks. IEEE Transactions on Signal Processing, 70, 1850-1863.
        # batch_size = 1

        channels = channels.squeeze()  # size: user_num by user_num (batch size = 1)
        user_num = self.args.user_num
        K = self.args.K_hop

        x = torch.zeros(K * user_num, 1)  # status placeholder
        x[0:user_num, 0] = states  # vector, dim: (user_num)
        H_t = channels.squeeze()  # matrix, dim: (user_num, user_num)
        H_t[torch.abs(H_t) <= self.args.channel_thr] = 0  # \tilde channel in the paper eq (1)

        self.H_stack = torch.cat((torch.eye(self.args.user_num), self.H_stack[:, 0:self.H_stack.shape[1] - self.args.user_num]), dim=1)  # stack of channels [I,H(t-1),H(t-1)H(t-2),...] dim: (user_num,user_num*K)
        self.H_stack = H_t @ self.H_stack  # stack of channels [H(t),H(t)H(t-1),H(t)H(t-1)H(t-2),...] dim: (user_num,user_num*K)

        self.x_stack = torch.cat((torch.zeros(user_num, K-1), self.x_stack[0:user_num * (K-1), 0:K-1]), dim=0)  # deleting the oldest state
        self.x_stack = torch.cat((x, self.x_stack), dim=1)  # putting the new state, stack of states with block-diag shape dim: (user_num * K,K)
        Y = self.H_stack @ self.x_stack  # policy inputs, dim: (user_num, K), K is equal to the policy input size
        net_input = Y
        return net_input