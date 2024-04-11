import torch
import torch.nn as nn
from torchinfo import summary

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.AgeNN = nn.Sequential(
            nn.Linear(32768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.GenderNN = nn.Sequential(
            nn.Linear(32768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.RaceNN = nn.Sequential(
            nn.Linear(32768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.CNN(x)
        age = self.AgeNN(x)
        gender = self.GenderNN(x)
        race = self.RaceNN(x)
        return age, gender, race
    
model = Model()
torch.save(model, '.\\Models\\model_v1\\model_v1.pt')

info_path = '.\\Models\\model_v1\\model_v1_info.txt'
f = open(info_path, 'w')
model_info = str(summary(model.cuda(), input_size=(256, 3, 128, 128), device='cuda'))
f.write(model_info)
f.close()