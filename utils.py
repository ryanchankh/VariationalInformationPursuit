def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        return curr_lr


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
               continue
            else:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm