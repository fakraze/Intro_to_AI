import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN1Claiming(nn.Module):
    def __init__(self):
        super(NN1Claiming, self).__init__()
        self.inputSize = 102
        self.outputSize = 2
        self.hiddenSize1 = 128
        self.hiddenSize2 = 128
        
        self.network = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.outputSize)
            # Removed Softmax
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)

    def forward(self, input):
        return self.network(input)
    
class NN1Discard(nn.Module):
    def __init__(self):
        super(NN1Discard, self).__init__()
        self.inputSize = 68
        self.outputSize = 34
        self.hiddenSize1 = 128
        self.hiddenSize2 = 128
        
        self.network = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.outputSize)
            # Removed Softmax
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)

    def forward(self, input):
        return self.network(input)
    

class NN2Claiming(nn.Module):
    def __init__(self):
        super(NN2Claiming, self).__init__()
        self.inputSize = 34*13
        self.outputSize = 2
        self.hiddenSize1 = 512
        self.hiddenSize2 = 256
        self.hiddenSize3 = 256
        
        self.network=nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.hiddenSize3),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize3, self.outputSize),
            nn.Softmax()
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)
    def forward(self, input):
        input=torch.tensor(input).float()
        return self.network(input)
    
class NN3Claiming(nn.Module):
    def __init__(self):
        super(NN3Claiming, self).__init__()
        self.inputSize = 34*13
        self.outputSize = 6
        self.hiddenSize1 = 512
        self.hiddenSize2 = 256
        self.hiddenSize3 = 256
        
        self.network=nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.hiddenSize3),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize3, self.outputSize),
            nn.Softmax()
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)
    def forward(self, input):
        input=torch.tensor(input).float()
        return self.network(input)
    

class NN23Discard(nn.Module):
    def __init__(self):
        super(NN23Discard, self).__init__()
        self.inputSize = 34*13
        self.outputSize = 34
        self.hiddenSize1 = 512
        self.hiddenSize2 = 256
        self.hiddenSize3 = 256
        
        self.network=nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.hiddenSize3),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize3, self.outputSize),
            nn.Softmax()
        )
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)
    def forward(self, input):
        return self.network(input)
    

