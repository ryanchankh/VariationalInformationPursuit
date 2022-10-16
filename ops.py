def sample_history(max_queries_sample, max_queries_possible):
    num_queries = torch.randint(low=0, high=max_queries_sample, size=(x.size(0),))
    mask = torch.zeros(X_train.shape[0], max_queries_possible).to(device)
    
    for code_ind, num in enumerate(num_queries):
        if num == 0:
            continue
        random_history = torch.multinomial(torch.ones(max_queries_possible), num, replacement=False)
        mask[code_ind, random_history.flatten()] = 1.0

    return mask



def get_patch_mask(mask, x, patch_size):
    patch_mask = torch.zeros(x.size()).cuda()
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
    masked_image = copy.deepcopy(masked_image)
    N, _, H, W = original_image.shape

    query_vec = query_vec.view(x.size(0), 1, (H - patch_size + 1), (W - patch_size + 1))

    kernel = torch.ones(1, 1, patch_size, patch_size, requires_grad=False).cuda()
    # convoluting signal with kernel and applying padding
    output = F.conv2d(query_vec, kernel, stride=1, padding=patch_size - 1, bias=None)

    output = output * original_image
    modified_history = masked_image + output
    modified_history = torch.clamp(modified_history, min=-1.0, max=1.0)

    return modified_history