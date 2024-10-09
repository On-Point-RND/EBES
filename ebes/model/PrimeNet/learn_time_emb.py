import torch
import torch.nn as nn


class TimeBERT(nn.Module):

    def __init__(self):
        super().__init__()

        self.periodic = nn.Linear(1, 63)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, time_steps):

        key_learn = self.learn_time_embedding(time_steps)
        print(key_learn.shape)


"""
time = torch.rand(9)
model = TimeBERT()
print(time.shape)

model(time)
"""


class MultiTimeAttention(nn.Module):

    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super().__init__()
        assert embed_time % num_heads == 0
        self.h = num_heads
        self.embed_time_k = embed_time
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, 7),
                nn.Linear(3, 9),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def forward(self, query, key):
        print("Query:" + str(query.shape) + ", Key:" + str(key.shape))

        query, key = (layer(x) for layer, x in zip(self.linears, (query, key)))

        """
        for l, x in zip(self.linears, (query, key)):
            print('X:' + str(x.shape))
            y = l(x)
            print('Y:' + str(y.shape))
        """

        print("Query:" + str(query.shape) + ", Key:" + str(key.shape))


"""
hidden_size = 16
embed_time = 16
num_heads = 1
batch, seq_len, input_dim = 5, 20, 3


query = torch.rand(batch, seq_len, 2)
key = torch.rand(batch, seq_len, 3)

model = multiTimeAttention(2 * input_dim, hidden_size, embed_time, num_heads)
model(query, key)
"""


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
