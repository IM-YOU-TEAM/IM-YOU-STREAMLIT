from torch import nn

class MovementCNN(nn.Module):
    def __init__(self, input_channels=18, num_classes=5):
        super().__init__()
        blocks = []
        for cin, cout, kernel in [(input_channels, 64, 7), (64, 128, 5), (128, 128, 3)]:
            blocks += [nn.Conv1d(cin, cout, kernel, stride=2, padding=kernel // 2),
                       nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(0.2)]
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2)).mean(-1))
