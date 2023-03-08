import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import ops

class NetworkSymCAT(nn.Module):
    def __init__(self, K=20, query_size=13, position_embedding_dim=10, output_dim=1, discrete=True, arch="PNP", agg_type="max", eps=0.2, append_posterior_to_state=False):
        super().__init__()

        self.K = K
        self.query_size = query_size
        self.position_embedding_dim = position_embedding_dim
        self.output_dim = output_dim
        self.discrete = discrete
        self.current_max = 0
        self.arch = arch
        self.append_posterior_to_state = append_posterior_to_state
        self.agg_type = agg_type
        self.EPS = eps
        
        #position embedding
        self.F = nn.Parameter(torch.randn(1, self.query_size, self.position_embedding_dim, dtype=torch.float))

        self.start_token = nn.Parameter(torch.randn(self.K, dtype=torch.float))

        #bias
        self.b = nn.Parameter(torch.randn(1, self.query_size, 1, dtype=torch.float))

        self.h = nn.Linear(self.position_embedding_dim + 2, self.K) #the plus 2 is for bias and the feature itself

        self.layer1 = nn.Linear(self.K, 2000)
        self.bnorm1 = torch.nn.BatchNorm1d(2000)
        self.layer2 = nn.Linear(2000, 500)
        self.bnorm2 = torch.nn.BatchNorm1d(500)

        if self.append_posterior_to_state:
            self.querier = nn.Linear(500 + self.output_dim, self.query_size)
        else:
            self.querier = nn.Linear(500, self.query_size)

        self.classifier = nn.Linear(500, self.output_dim)

        # activations
        self.relu = nn.ReLU() #nn.LeakyReLU(negative_slope=0.3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def gen_histories(self, x, num_queries, max_queries):
        mask = torch.zeros(x.size()).cuda()
        final_mask = torch.zeros(x.size()).cuda()
        sorted_indices = num_queries.argsort()
        counter = 0
        with torch.no_grad():
            for i in range(max_queries + 1):
                while (counter < x.size(0)):
                    batch_index = sorted_indices[counter]
                    if i == num_queries[batch_index]:
                        final_mask[batch_index] = mask[batch_index]
                        counter += 1
                    else:
                        break
                if counter == x.size(0):
                    break
                query, label_logits = self.forward(x, mask)
                mask[np.arange(x.size(0)), query.argmax(dim=1)] = 1.0
        return final_mask

    def forward(self, x, mask, argmax=False):
        #squeezing the batch_sz and feature dimensions together

        x_flat = x.view(-1,1)
        position_embedding = self.F.repeat(x.size(0), 1, 1).view(-1, self.position_embedding_dim)
        b = self.b.repeat(x.size(0), 1, 1).view(-1,1)

        if self.arch == "PNP":
            x_aug = torch.cat([x_flat, x_flat*position_embedding, b], axis=1).float()

        elif self.arch == "PN":
            x_aug = torch.cat([x_flat, position_embedding, b], axis=1).float()
        embedding = self.h(x_aug)
        embedding = embedding.view(-1, self.query_size, self.K)

        query_mask = torch.where(mask == 1, -1e9, 0.)

        mask = mask.view(-1, self.query_size, 1).repeat(1, 1, self.K)

        if self.agg_type == "sum":
            aggregate = self.relu((mask*embedding).sum(dim=1))
        elif self.agg_type == "max":
            aggregate = self.relu((mask * embedding).amax(dim=1))

        output = self.relu(self.bnorm1(self.layer1(aggregate)))
        output = self.relu(self.bnorm2(self.layer2(output)))

        if not self.discrete:
            label_logits = self.sigmoid(self.classifier(output))
        else:
            label_logits = self.classifier(output)

        if self.append_posterior_to_state:
            query_logits = self.querier(torch.cat([output, nn.Softmax()(label_logits)], dim=1))
        else:
            query_logits = self.querier(output)
        query_logits = query_logits + query_mask.cuda()

        if argmax is False:
            query = self.softmax(query_logits / self.EPS)
            query = (self.softmax(query_logits / 1e-9) - query).detach() + query

        return query, label_logits

    def forward_query(self, x, mask, query, argmax=False):
        x_flat = x.view(-1, 1)
        position_embedding = self.F.repeat(x.size(0), 1, 1).view(-1, self.position_embedding_dim)
        b = self.b.repeat(x.size(0), 1, 1).view(-1, 1)

        if self.arch == "PNP":
            x_aug = torch.cat([x_flat, x_flat * position_embedding, b], axis=1).float()

        elif self.arch == "PN":
            x_aug = torch.cat([x_flat, position_embedding, b], axis=1).float()

        embedding = self.h(x_aug)
        embedding = embedding.view(-1, self.query_size, self.K)

        mask = mask.view(-1, self.query_size, 1).repeat(1, 1, self.K)

        if self.agg_type == "sum":
            aggregate_before_relu = (mask * embedding).sum(dim=1)
        elif self.agg_type == "max":
            aggregate_before_relu = (mask * embedding).amax(dim=1)

        queried_answer = torch.bmm(query.unsqueeze(1), x.unsqueeze(2)).squeeze(1)

        queried_answer_times_position = torch.bmm(query.unsqueeze(1), x.unsqueeze(2)*self.F).squeeze(1)

        query_answer_bias = torch.bmm(query.unsqueeze(1), self.b.repeat(x.size(0), 1, 1)).squeeze(1)

        if self.arch == "PNP":
            new_query_answer = torch.cat([queried_answer, queried_answer_times_position, query_answer_bias], axis=1).float()

        elif self.arch == "PN":
            new_query_answer = torch.cat([queried_answer, torch.bmm(query.unsqueeze(1), self.F.repeat(query.size(0), 1, 1)).squeeze(1), query_answer_bias], axis=1).float()

        new_query_answer = self.h(new_query_answer)

        if self.agg_type == "sum":
            aggregate = self.relu(aggregate_before_relu + new_query_answer)
        elif self.agg_type == "max":
            aggregate = self.relu(torch.maximum(aggregate_before_relu, new_query_answer))

        output = self.relu(self.bnorm1(self.layer1(aggregate)))
        output = self.relu(self.bnorm2(self.layer2(output)))

        if not self.discrete:
            label_logits = self.sigmoid(self.classifier(output))
        else:
            label_logits = self.classifier(output)

        return label_logits


    def sequential(self, x, y, max_queries, model_save_filename, random_sample_symptom, initial_test_mask = None, threshold = 0.85, evaluate = False):

        if initial_test_mask == None:
            mask = torch.zeros(x.size()).cuda()
        else:
            mask = initial_test_mask.cuda()
        if random_sample_symptom:
            #find an initial position symptom:
            for patient_ind, patient in enumerate(x):
                positive_symptoms = (patient == 1).nonzero(as_tuple=False)
                random_permuation = torch.randperm(len(positive_symptoms))
                try:
                    mask[patient_ind][positive_symptoms[random_permuation[0]]] = 1.0
                except IndexError:
                    continue
        logits = []
        queries = []
        for i in range(max_queries):
            query, label_logits = self.forward(x, mask)
            mask[np.arange(x.size(0)), query.argmax(dim=1)] = 1.0
            logits.append(label_logits)
            queries.append(query)
            if not self.discrete:
                print ("iteration:", i, torch.sqrt(((label_logits.squeeze() - y)**2).mean()).item())

        if self.discrete:
            max_queries_accuracy = (label_logits.argmax(dim=1).float() == y.squeeze()).float().mean().item()

            logits = torch.stack(logits).permute(1, 0, 2)
            queries_needed = ops.compute_queries_needed(logits, threshold=threshold)

            test_pred_ip = logits[torch.arange(len(queries_needed)), queries_needed - 1].argmax(1)

            ip_accuracy = (test_pred_ip == y.squeeze()).float().mean().item()

            se = queries_needed.float().mean().item()
            se_std = queries_needed.float().std().item()

            return max_queries_accuracy, ip_accuracy, se, se_std

    def change_eps(self, eps):
        self.EPS = eps


