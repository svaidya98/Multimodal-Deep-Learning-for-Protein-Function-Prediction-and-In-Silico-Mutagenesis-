import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import scripts.config as config


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcitation(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)
    
class ProteinDNN(nn.Module):
    def __init__(self, input_dim = 640, num_labels = 1500, hidden_dim = 512, num_blocks = 3):
        super(ProteinDNN, self).__init__()
        with open(config.processed_dir / 'taxon_map.json', 'r') as f:
            taxon_map = json.load(f)
        num_taxa = len(taxon_map) + 1
        self.taxon_embedding = nn.Embedding(num_taxa, config.taxon_embed_dim)
        full_input_dim = input_dim + config.taxon_embed_dim
        self.projection = nn.Linear(config.embed_dim + config.taxon_embed_dim, 512)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
        self._init_weights()

    def forward(self, x, taxon_idx):
        t_emb = self.taxon_embedding(taxon_idx)
        if len(x.shape) > 2: x = x.squeeze(1)
        x = torch.cat([x, t_emb], dim=1)
        x = F.relu(self.projection(x))
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

