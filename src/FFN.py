import torch.nn as nn
import torch
import numpy as np

class FFN(nn.Module):
    def __init__(self,input_dim,hidden_dim):


        super(FFN,self).__init__()


        self.ffn = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1)
                                 )



    def forward(self, x):
        """

        :param
        x: x is a tensor with dimension (batch_size, input_dim)


        :return: a tensor with dimension (batch_size, 1)
        """
        return self.ffn(x)
