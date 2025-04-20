import torch
import torch.nn as nn

class StackedLSTM_MultiPooling(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        """
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Hidden dimension for each LSTM layer.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate applied in LSTM and classifier.
        """
        super(StackedLSTM_MultiPooling, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
       
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            curr_input_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(input_size=curr_input_size, hidden_size=hidden_size,
                        num_layers=1, batch_first=True, dropout=dropout)
            )
        # Total feature dimension for classification is: num_layers * hidden_size * 2 (for mean and max pooling).
        pooled_feature_size = num_layers * hidden_size * 2
       
        self.classifier = nn.Sequential(
            nn.Linear(pooled_feature_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
        # Manual dropout between LSTM layers (if desired).
        self.manual_dropout = nn.Dropout(dropout)
    
    def pool_outputs(self, lstm_output):
        """Helper function to perform mean and max pooling."""
        mean_pool = lstm_output.mean(dim=1)  # Shape: (batch, hidden_size)
        max_pool, _ = lstm_output.max(dim=1)  # Shape: (batch, hidden_size)
        return torch.cat([mean_pool, max_pool], dim=1)  # Shape: (batch, hidden_size * 2)
    
    def forward(self, x, return_intermediates=False):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, input_size).
            return_intermediates (bool): If True, returns intermediate pooled outputs.
        Returns:
            logits (Tensor): Class logits of shape (batch, num_classes).
            intermediates (dict, optional): Dictionary with pooled outputs from each layer.
        """
        pooled_outputs = {}  # To store pooled representations per layer
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(x)
            # Apply manual dropout between layers.
            out = self.manual_dropout(out)
            
            pooled = self.pool_outputs(out)
            pooled_outputs[f"LSTM Layer {i+1}"] = pooled
            # Set output as the input to the next LSTM.
            x = out
        
        final_representation = torch.cat(list(pooled_outputs.values()), dim=1)
        logits = self.classifier(final_representation)
        
        if return_intermediates:
            return logits, pooled_outputs
        return logits
