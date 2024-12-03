import torch
import numpy as np

class gen_rect_channel:
    def __init__(self, args, L=500, l=50, ch_cor=0.3, ch_var=1, path_loss_fact=2.2):
        self.L = L  # the area of [-L,L]^2
        self.l = l  # each reciever within an area of [-l,l]^2 from the corresponding transmitter
        self.args = args
        self.ch_cor = ch_cor
        self.ch_var = ch_var
        self.path_loss_fact = path_loss_fact
        self.counter = 1

    def gen_channel(self, sample_num, user_num, ch_type="grid", new_path_loss=True):
        if ch_type == "circ":
            channel = torch.abs(torch.complex(np.sqrt(0.5) * torch.randn(sample_num, user_num, user_num), np.sqrt(0.5) * torch.randn(sample_num, user_num, user_num)))
        elif ch_type == "norm":
            channel = torch.randn(sample_num, user_num, user_num)
        elif ch_type == "grid":
            if new_path_loss or self.counter == 1:
                self.path_loss = self.config_loss()
                self.fast_fading = np.sqrt(self.ch_var) * torch.randn(user_num, user_num)

        channel = torch.zeros(sample_num, user_num, user_num)
        for i in range(sample_num):
            self.fast_fading = np.sqrt(1 - self.ch_cor) * self.fast_fading + np.sqrt(self.ch_cor) * np.sqrt(self.ch_var) * torch.randn(user_num, user_num)
            channel[i] = self.fast_fading * self.path_loss
            channel[i] = channel[i].float()

        self.counter += 1
        return channel

    def config_loss(self):
        T_x = self.L * (1 - 2 * np.random.rand((self.args.user_num)))                  # transmitter location
        T_y = self.L * (1 - 2 * np.random.rand((self.args.user_num)))
        T_X = np.repeat(T_x[None, :], self.args.user_num, axis=0)
        T_Y = np.repeat(T_y[None, :], self.args.user_num, axis=0)

        R_x = self.l * (1 - 2 * np.random.rand((self.args.user_num)))   # receiver location (wrt the transmitter)
        R_y = self.l * (1 - 2 * np.random.rand((self.args.user_num)))
        R_x = T_x + R_x    # reciever location
        R_y = T_y + R_y
        R_X = np.repeat(R_x[:, None], self.args.user_num, axis=1)
        R_Y = np.repeat(R_y[:, None], self.args.user_num, axis=1)

        d_x = (R_X - T_X) ** 2     # relative distance of each receiver to all the transmitters (in x)
        d_y = (R_Y - T_Y) ** 2     # relative distance of each receiver to all the transmitters (in y)

        d = np.sqrt(d_x + d_y)     # relative distance of each receiver to all the transmitters (in m)
        path_loss = d ** (-self.path_loss_fact)
        path_loss = torch.from_numpy(path_loss)
        return path_loss
