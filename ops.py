import numpy as np
import torch
import torch.nn.functional as F



def random_sampling(max_queries_sample, max_queries_possible, num_samples):
    num_queries = torch.randint(low=0, high=max_queries_sample, size=(num_samples, ))
    mask = torch.zeros(num_samples, max_queries_possible)
    
    for code_ind, num in enumerate(num_queries):
        if num == 0:
            continue
        random_history = torch.multinomial(torch.ones(max_queries_possible), num, replacement=False)
        mask[code_ind, random_history.flatten()] = 1.0

    return mask



def compute_queries_needed(logits, threshold, mode='map', k=1, softmax=True):
    if mode == 'map':
        return compute_semantic_entropy_map(logits, threshold, softmax=softmax)
    elif mode == 'stability':
        return compute_semantic_entropy_statibility(logits, threshold, k=k, softmax=softmax)
    else:
        raise NameError(f'{mode} mode not found.')
        

def compute_semantic_entropy_map(logits, threshold, softmax=False):
    """Compute the number of queries needed for each prediction.
    Parameters:
        logits (torch.Tensor): logits from querier
        threshold (float): stopping criterion, should be within (0, 1)
    """
    assert 0 < threshold and threshold < 1, 'threshold should be between 0 and 1'
    n_samples, n_queries, _ = logits.shape
    device = logits.device

    # turn logits into probability and find queried prob.
    if softmax:
        prob = F.softmax(logits, dim=2)
    else:
        prob = logits
    prob_max = prob.amax(dim=2)

    # `decay` to multipled such that argmax finds
    #  the first nonzero that is above threshold.
    threshold_indicator = (prob_max >= threshold).float().to(device)
    decay = torch.linspace(10, 1, n_queries).unsqueeze(0).to(device)
    semantic_entropy = (threshold_indicator * decay).argmax(1)

    # `threshold_indicator`==0 is to check which
    # samples did not stop querying, hence indicator vector
    # is all zeros, preventing bug that yields argmax as 0.
    semantic_entropy[threshold_indicator.sum(1) == 0] = n_queries
    semantic_entropy[threshold_indicator.sum(1) != 0] += 1

    return semantic_entropy


def compute_semantic_entropy_statibility(logits, threshold, k = 1, softmax=False):
    """Compute the number of queries needed for each prediction.
    
    Parameters:
        logits (torch.Tensor): logits from querier
        threshold (float): stopping criterion, should be within (0, 1)
 
    """
    n_samples, n_queries, _ = logits.shape
    
    # turn logits into probability and find queried prob.
    if softmax:
        prob = F.softmax(logits, dim=2)
    else:
        prob = logits

    
    entropy1 = -(prob[:,:-1]*np.log2(prob[:,:-1])).sum(dim=2)
    entropy2 = -(prob[:,1:]*np.log2(prob[:,1:])).sum(dim=2)

    
    difference = (np.absolute(entropy1 - entropy2))
    
    difference = torch.cat([difference, torch.zeros(difference.size(0),1)], dim=1)
    
    # `decay` to multipled such that argmax finds 
    #  the first nonzero that is above threshold. 
    threshold_indicator = (difference <= threshold).float()
    
    signal = threshold_indicator.view(threshold_indicator.size(0), 1, -1)
    
    # convolution kernel of size 3, expecting 1 input channel and 1 output channel
    kernel = torch.ones(1, 1, k, requires_grad=False)
    # convoluting signal with kernel and applying padding
    output = F.conv1d(signal, kernel, stride=1, padding=k-1, bias=None)[:,:,k-1:].squeeze(1)
    
    threshold_indicator = (output == k).float()
    
    decay = torch.linspace(10, 1, n_queries).unsqueeze(0)
    semantic_entropy = (threshold_indicator * decay).argmax(1)
    
    # `threshold_indicator`==0 is to check which 
    # samples did not stop querying, hence indicator vector 
    # is all zeros, preventing bug that yields argmax as 0.
    semantic_entropy[threshold_indicator.sum(1)==0] = n_queries
    semantic_entropy[threshold_indicator.sum(1)!=0] += 1
    return semantic_entropy
