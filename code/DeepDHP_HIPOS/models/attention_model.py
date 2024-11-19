import torch
import torch.nn as nn
import warnings
from dataset import *  # Assuming this contains necessary dataset utilities

# Suppress warnings
warnings.filterwarnings('ignore')

class Net(nn.Module):
    """
    Autoencoder with Attention Mechanism and Dual Encoders.
    """
    def __init__(self, arg):
        super(Net, self).__init__()
        self.arg = arg
        self.input_dim = arg.input_dim
        self.encoding_dim = arg.encoding_dim

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            nn.Tanh()  # Modified as per requirements
        )

        # Encoder 1
        self.encoder1 = self._build_encoder()

        # Encoder 2
        self.encoder2 = self._build_encoder()

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim // 4),
            nn.BatchNorm1d(self.input_dim // 4),
            nn.ReLU(),
            nn.Linear(self.input_dim // 4, 2),
            nn.Softmax(dim=1)
        )

        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def _build_encoder(self):
        """
        Constructs an encoder block.
        """
        return nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.BatchNorm1d(self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU()
        )

    def load_pretrain(self):
        """
        Loads pre-trained weights for the ResNet model.
        """
        print(f'Loading pretrained model from {self.arg.Model_Pretrained_Res}...')
        checkpoint = torch.load(self.arg.Model_Pretrained_Res, map_location=lambda storage, loc: storage)
        print(self.res.load_state_dict(checkpoint, strict=False))  # Strict is False to allow partial loading.

    def forward(self, batch):
        """
        Forward pass for the model.

        :param batch: A dictionary containing input data and labels.
        :return: Output dictionary containing loss or predictions based on `output_type`.
        """
        x = batch['input']
        labels = batch.get('label', None)

        # Attention scores and weighted input
        attention_scores = self.attention(x)
        x_attention = x * attention_scores

        # Encoding
        encoded1 = self.encoder1(x_attention)
        encoded2 = self.encoder2(x)
        encoded = torch.cat((encoded1, encoded2), dim=1)

        # Decoding
        decoded = self.decoder(encoded)

        # Output dictionary
        output = {}

        if 'loss' in self.output_type:
            # Loss calculations
            mse_loss = self.mse_loss(encoded1, encoded2)
            output['MSELoss'] = mse_loss

            if labels is not None:
                ce_loss = self.cross_entropy_loss(decoded, labels)
                output['loss'] = ce_loss + mse_loss  # Combined loss
            else:
                output['loss'] = mse_loss  # Fallback if no labels are provided

        if 'inference' in self.output_type:
            # Predictions and scores
            output['Predict'] = torch.argmax(decoded, dim=1)
            output['Predict_score'] = decoded
            output['attention_scores'] = attention_scores

        return output
