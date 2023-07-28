import torch.nn as nn
import json

with open("config.json","r") as file:
    config=json.load(file)

class Dog(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config["dropout"])

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)

        self.maxPool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # nn.init.xavier_uniform(self.conv1.weight)

        self.l1 = nn.Linear(in_features=64 * 61 * 61, out_features=config['hidden_dims'][0])
        self.l2 = nn.Linear(in_features=config['hidden_dims'][0], out_features=config['hidden_dims'][1])
        self.l3 = nn.Linear(in_features=config['hidden_dims'][1], out_features=output_dim)

    def forward(self, x):
        # x=x.unsqueeze(0)
        keep_going = self.dropout(self.maxPool(self.relu(self.bn1(self.conv1(x)))))
        keep_going = self.dropout(
            self.maxPool(self.relu(self.bn2(self.conv2(keep_going))))
        )
        keep_going = keep_going.view(
            keep_going.size(0), -1
        )  # batch size thing happen here
        keep_going = self.relu(self.l1(keep_going))
        keep_going = self.relu(self.l2(keep_going))
        keep_going = self.l3(keep_going)
        return keep_going





model = Dog(config["input_dim"], config["output_dim"])
