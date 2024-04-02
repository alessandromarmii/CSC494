import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetworkOptimalAgent(nn.Module):
    def __init__(self, input_size, output_size, model_layer_size=None):
         
        super(QNetworkOptimalAgent, self).__init__()

        if model_layer_size:   
            layers = model_layer_size
        else:
            layers = 750

        self.fc1 = nn.Linear(input_size, layers) 
        # Initialize weights using He initialization for ReLU activation
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
#        self.dropout = nn.Dropout(p=0.25)
#        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(layers, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
       # x = self.dropout(x)
    #    x = self.fc2(x)
    #    x = torch.relu(x)
        x = self.fc3(x)
        return x


class QNetworkExpectedFeedback(nn.Module):

    def __init__(self, input_size, output_size):
        super(QNetworkExpectedFeedback, self).__init__()
        self.fc1 = nn.Linear(input_size, 700) 
        # Initialize weights using He initialization for ReLU activation
        self.fc3 = nn.Linear(700, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
       # x = self.dropout(x)
    #    x = self.fc2(x)
    #    x = torch.relu(x)
        x = self.fc3(x)
        
        return x

class SimpleQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
         
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 600) 

        # Initialize weights using He initialization for ReLU activation
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

        self.fc3 = nn.Linear(600, output_size)  

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
       # x = self.dropout(x)
    #    x = self.fc2(x)
    #    x = torch.relu(x)
        x = self.fc3(x)
        return x
