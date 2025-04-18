import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), # 64×1×28×28 => 64×32×26×26
            nn.ReLU(),
            nn.Flatten(), # 64×32×26×26 => 64×21632
            nn.Linear(21632, 10)
        )

    def forward(self, x):
        return self.conv(x)