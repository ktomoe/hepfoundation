import torch
import torch.nn as nn

class EmbeddingModule(nn.Module):
    def __init__(self, features=9, nodes=16):
        super(EmbeddingModule, self).__init__()

        self.fc = nn.Linear(features, nodes)
        self.ln = nn.LayerNorm(nodes)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)

        return x


class FeatureModule(nn.Module):
    def __init__(self, layers, nodes=16, num_heads=4, dropout=0.):
        super(FeatureModule, self).__init__()


        encoder_layer = nn.TransformerEncoderLayer(d_model=nodes,
                                                   nhead=num_heads,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.encorders = nn.TransformerEncoder(encoder_layer, 
                                               num_layers=layers,
                                               enable_nested_tensor=False)


    def forward(self, x, mask):
        return self.encorders(x)

        
class ClassifierModule(nn.Module):
    def __init__(self, task='graph', nodes=16, layers=2):
        super(ClassifierModule, self).__init__()

        self.task = task

        if self.task == 'maintask':
            self.fc = nn.Linear(nodes, 2)

        elif self.task == 'pretrain':
            self.fc = nn.Linear(nodes, 4)


    def forward(self, x):

        if self.task == 'maintask':
            x = torch.mean(x, dim=1)

        x = self.fc(x)

        return x    

class TransformerModel(nn.Module):
    def __init__(self, task='maintask', features=9, layers=2, nodes=16, num_heads=4, dropout=0.):
        super(TransformerModel, self).__init__()
      
        self.embedding = EmbeddingModule(features, nodes)
        self.feature = FeatureModule(layers, nodes, num_heads, dropout)
        self.classifier = ClassifierModule(task, nodes, layers)
      
    def forward(self, x):
        x, mask = x
        x = self.embedding(x)
        x = self.feature(x, mask)
        x = self.classifier(x)
       
        return x

