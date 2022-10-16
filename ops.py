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

def adaptive_sampling(x, num_queries, querier, patch_size, max_queries):
    device = x.device
    N, C, H, W = x.shape

    mask = torch.zeros(N, (H - patch_size + 1)*(W - patch_size + 1)).to(device)
    final_mask = torch.zeros(N, (H - patch_size + 1)*(W - patch_size + 1)).to(device)
    patch_mask = torch.zeros((N, C, H, W)).to(device)
    final_patch_mask = torch.zeros((N, C, H, W)).to(device)
    sorted_indices = num_queries.argsort()
    counter = 0

    with torch.no_grad():
        for i in range(max_queries + 1):
            while (counter < N):
                batch_index = sorted_indices[counter]
                if i == num_queries[batch_index]:
                    final_mask[batch_index] = mask[batch_index]
                    final_patch_mask[batch_index] = patch_mask[batch_index]
                    counter += 1
                else:
                    break
            if counter == N:
                break
            query_vec = querier(patch_mask, mask)
            mask[np.arange(N), query_vec.argmax(dim=1)] = 1.0
            patch_mask = update_masked_image(patch_mask, x, query_vec, patch_size)
    return final_mask, final_patch_mask



def get_patch_mask(mask, x, patch_size):
    patch_mask = torch.zeros(x.size()).to(x.device)
    for batch_index in range(mask.size(0)):
        positive_indices = torch.where(mask[batch_index] == 1)[0]

        index_i = positive_indices // (x.size(3) - patch_size + 1)
        index_j = positive_indices % (x.size(3) - patch_size + 1)

        for row in range(patch_size):
            for col in range(patch_size):
                part_of_image = x[batch_index, :, index_i + row, index_j + col]
                patch_mask[batch_index, :, index_i + row, index_j + col] = part_of_image

    return patch_mask


def update_masked_image(masked_image, original_image, query_vec, patch_size):
    N, _, H, W = original_image.shape
    device = masked_image.device

    query_vec = query_vec.view(N, 1, (H - patch_size + 1), (W - patch_size + 1))

    kernel = torch.ones(1, 1, patch_size, patch_size, requires_grad=False).to(device)
    # convoluting signal with kernel and applying padding
    output = F.conv2d(query_vec, kernel, stride=1, padding=patch_size - 1, bias=None)

    output = output * original_image
    modified_history = masked_image + output
    modified_history = torch.clamp(modified_history, min=-1.0, max=1.0)

    return modified_history


def compute_queries_needed(logits, threshold):
    """Compute the number of queries needed for each prediction.

    Parameters:
        logits (torch.Tensor): logits from querier
        threshold (float): stopping criterion, should be within (0, 1)

    """
    assert 0 < threshold and threshold < 1, 'threshold should be between 0 and 1'
    n_samples, n_queries, _ = logits.shape
    device = prob.device

    # turn logits into probability and find queried prob.
    prob = F.softmax(logits, dim=2)
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
