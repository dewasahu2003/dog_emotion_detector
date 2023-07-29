import torch.nn as nn


class Dog(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        hidden_layer1: int,
        hidden_layer2: int,
        hidden_layer3: int,
    ) -> None:
        super().__init__()

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.maxPool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # nn.init.xavier_uniform(self.conv1.weight)
        if hidden_layer2 > hidden_layer1:
            raise RuntimeError("hidden_layer1 must be greater than hidden_layer2")
        self.l1 = nn.Linear(in_features=128 * 26 * 26, out_features=hidden_layer1)
        self.l2 = nn.Linear(in_features=hidden_layer1, out_features=hidden_layer2)
        self.l3 = nn.Linear(in_features=hidden_layer2, out_features=output_dim)

    def forward(self, x):
        # x=x.unsqueeze(0)
        keep_going = self.maxPool(self.relu(self.bn1(self.conv1(x))))
        keep_going = self.maxPool(self.relu(self.bn2(self.conv2(keep_going))))
        keep_going = self.maxPool(self.relu(self.bn3(self.conv3(keep_going))))
        keep_going = keep_going.view(
            keep_going.size(0), -1
        )  # batch size thing happen here
        keep_going = self.dropout(self.relu(self.l1(keep_going)))
        keep_going = self.dropout(self.relu(self.l2(keep_going)))
        keep_going = self.l3(keep_going)
        return keep_going
