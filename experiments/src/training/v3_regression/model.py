import torch
import torch.nn as nn

class MLP_Regression(nn.Module):
    """
    V3 Regression MLP: Predicts continuous OGT temperature.
    Includes BatchNorm and Dropout for regularization.
    Output is a single linear unit without Sigmoid.
    """
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256,
                 dropout_1=0.3, dropout_2=0.2):
        super(MLP_Regression, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),

            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),

            nn.Linear(hidden_size_2, 1)  # Raw linear output for temperature
        )

    def forward(self, x):
        """
        Forward pass.
        Returns predicted temperature.
        """
        return self.model(x)

    def predict(self, x):
        """
        Inference mode prediction.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output.squeeze(-1)
