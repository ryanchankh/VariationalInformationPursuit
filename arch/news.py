import torch
import torch.nn as nn



class NetworkNews(nn.Module):
    def __init__(self, query_size=312, output_size=312, tau=None):
        super().__init__()
        self.query_size = query_size
        self.output_dim = output_size
        self.layer1 = nn.Linear(self.query_size, 2000)
        self.layer2 = nn.Linear(2000, 1000)
        self.layer3 = nn.Linear(1000, 500)
        self.classifier = nn.Linear(500, self.output_dim)

        self.tau = tau
        self.current_max = 0

        self.norm1 = torch.nn.LayerNorm(2000)
        self.norm2 = torch.nn.LayerNorm(1000)
        self.norm3 = torch.nn.LayerNorm(500)
        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.relu(self.norm3(self.layer3(x)))

        if self.tau == None:
            return self.classifier(x)

        else:
            query_logits = self.classifier(x)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_logits = query_logits + query_mask.cuda()

            query = self.softmax(query_logits / self.tau)

            query = (self.softmax(query_logits / 1e-9) - query).detach() + query
            return query

    def update_tau(self, tau):
        self.tau = tau



