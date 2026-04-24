import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU()
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

    def encode(self, x):
        return self.encoder(x)