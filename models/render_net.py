import torch.nn as nn
class density_net(nn.Module):
    '''
    decoder for density
    '''
    def __init__(self, input_channel, intermediate_channel, output_channel):
        super(density_net, self).__init__()
        self.embedding_one = nn.Sequential(
            nn.Linear(input_channel, intermediate_channel),
        )
        self.embedding_two = nn.Sequential(
            nn.Linear(intermediate_channel, output_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.embedding_one(x)
        x = self.embedding_two(x)
        return x

class radiance_net(nn.Module):
    '''
    decoder for radiance
    '''
    def __init__(self, input_channel, intermediate_channel, output_channel):
        super(radiance_net, self).__init__()
        self.embedding_one = nn.Sequential(
            nn.Linear(input_channel, intermediate_channel),
            nn.LeakyReLU(inplace=True)
        )
        self.embedding_two = nn.Sequential(
            nn.Linear(intermediate_channel, intermediate_channel),
            nn.LeakyReLU(inplace=True)
        )
        self.embedding_three = nn.Sequential(
            nn.Linear(intermediate_channel, output_channel),
        )
    def forward(self, x):
        x = self.embedding_one(x)
        x = self.embedding_two(x)
        x = self.embedding_three(x)
        return x