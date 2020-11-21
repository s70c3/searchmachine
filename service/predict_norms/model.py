import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvModel(nn.Module):
    def __init__(self, cat_n):
        super(ConvModel, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3),
                                 nn.MaxPool2d(2),
                                 nn.BatchNorm2d(out_ch))

        base = 10
        # (N, 1, 150, 150) -> (N, 512)
        self.conv = nn.Sequential(conv_block(1, base * 3),
                                  conv_block(base * 3, base * 2),
                                  conv_block(base * 2, base * 3),
                                  conv_block(base * 3, base * 2),
                                  conv_block(base * 2, base * 3),
                                  Flatten())

        def linear_block(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(0.3),
                                 nn.Dropout(0.05))

        self.fc = nn.Sequential(linear_block(base * 3 * 4, 256),
                                linear_block(256, cat_n))

    def forward(self, batch):
        def conv_pics(pic_list):
            return list(map(self.conv, list(map(lambda t: torch.unsqueeze(t, dim=0), pic_list))))

        conved = list(map(conv_pics, batch))
        conved = list(map(lambda lt: torch.stack(lt).sum(dim=0), conved))
        conved = torch.stack(conved)

        clss = self.fc(conved).squeeze(dim=1)

        return clss