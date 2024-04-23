import argparse
import random
import time
import glob
from tqdm import tqdm   
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from arch.cifar10 import DLA
import dataset
import ops
import utils
import wandb



def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_queries', type=int, default=48)
    parser.add_argument('--max_queries_test', type=int, default=21)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='cifar10')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/', help='save directory')
    parser.add_argument('--ckpt_dir', type=bool, default=None, help='load checkpoint from this dir')
    args = parser.parse_args()
    return args


def adaptive_sampling(x, num_queries, querier, patch_size, max_queries):
    device = x.device
    N, C, H, W = x.shape

    mask = torch.zeros(x.size(0), 49).to(device)
    final_mask = torch.zeros(x.size(0), 49).to(device)
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


def get_patch_mask(mask, x, patch_size, stride=4):
    patch_mask = torch.zeros(x.size()).to(x.device)
    for batch_index in range(mask.size(0)):
        positive_indices = torch.where(mask[batch_index] == 1)[0]

        index_i = positive_indices // ((x.size(3) - patch_size)//stride + 1) * stride
        index_j = positive_indices % ((x.size(3) - patch_size)//stride + 1) * stride

        for row in range(patch_size):
            for col in range(patch_size):
                part_of_image = x[batch_index, :, index_i + row, index_j + col]
                patch_mask[batch_index, :, index_i + row, index_j + col] = part_of_image
    return patch_mask


def update_masked_image(masked_image, original_image, query_vec, patch_size, stride=4):
    N, _, H, W = original_image.shape
    device = masked_image.device

    query_grid = query_vec.view(N, 1, 7, 7)
    kernel = torch.ones(1, 1, patch_size, patch_size, requires_grad=False).to(device)
    output = F.conv_transpose2d(query_grid, kernel, stride=stride, bias=None)

    output = output * original_image
    modified_history = masked_image + output
    # modified_history = torch.clamp(modified_history, min=-1.0, max=1.0)
    modified_history = torch.where(modified_history == 2*masked_image, masked_image.detach(), modified_history)
    return modified_history


def main(args):
    ## Setup
    # wandb
    run = wandb.init(project="Variational-IP", name=args.name, mode=args.mode)
    model_dir = os.path.join(args.save_dir, f'{run.id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    wandb.config.update(args)

    # cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)

    # random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Constants
    N_QUERIES = 49 # 7*7
    N_CLASSES = 10
    PATCH_SIZE = 8
    STRIDE = 4
    THRESHOLD = 0.127

    ## Data
    trainset, testset = dataset.load_cifar10(args.data_dir)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)

    ## Model
    classifier = DLA(num_classes=N_CLASSES)
    classifier = nn.DataParallel(classifier).to(device)
    querier = DLA(num_classes=N_QUERIES, tau=args.tau_start, resize_conv=True)
    querier = nn.DataParallel(querier).to(device)

    ## Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(querier.parameters()) + list(classifier.parameters()), 
                           amsgrad=True, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    ## Load checkpoint
    if args.ckpt_path is not None:
        ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        classifier.load_state_dict(ckpt_dict['classifier'])
        querier.load_state_dict(ckpt_dict['querier'])
        # optimizer.load_state_dict(ckpt_dict['optimizer'])
        # scheduler.load_state_dict(ckpt_dict['scheduler'])
        print('Checkpoint Loaded!')

    ## Train
    for epoch in range(args.epochs):

        # training
        classifier.train()
        querier.train()
        tau = tau_vals[epoch]
        for train_images, train_labels in tqdm(trainloader):
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            querier.module.update_tau(tau)
            optimizer.zero_grad()

            # initial random sampling
            if args.sampling == 'baised':
                num_queries = torch.randint(low=0, high=N_QUERIES, size=(train_images.size(0),))
                mask, masked_image = adaptive_sampling(train_images, num_queries, querier, PATCH_SIZE, N_QUERIES)
            elif args.sampling == 'random':
                mask = ops.random_sampling(args.max_queries, N_QUERIES, train_images.size(0)).to(device)
                masked_image = get_patch_mask(mask, train_images, patch_size=PATCH_SIZE)

            # Query and update
            query_vec = querier(masked_image, mask)
            masked_image = update_masked_image(masked_image, train_images, query_vec, patch_size=PATCH_SIZE)

            # prediction
            train_logits = classifier(masked_image)

            # backprop
            loss = criterion(train_logits, train_labels)
            loss.backward()
            optimizer.step()

            # logging
            wandb.log({
                'epoch': epoch,
                'loss': loss.item(),
                'lr': utils.get_lr(optimizer),
                'gradnorm_cls': utils.get_grad_norm(classifier),
                'gradnorm_qry': utils.get_grad_norm(querier)
                })
        scheduler.step()

        # saving
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'classifier': classifier.state_dict(),
                'querier': querier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                },
                os.path.join(model_dir, 'ckpt', f'epoch{epoch}.ckpt'))

        # evaluation
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            classifier.eval()
            querier.eval()
            epoch_test_qry_need = []
            epoch_test_acc_max = 0
            epoch_test_acc_ip = 0
            for test_images, test_labels in tqdm(testloader):
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                N, H, C, W = test_images.shape

                # Compute logits for all queries
                test_inputs = torch.zeros_like(test_images).to(device)
                mask = torch.zeros(N, N_QUERIES).to(device)
                logits, queries = [], []
                for i in range(args.max_queries_test):
                    with torch.no_grad():
                        query_vec = querier(test_inputs, mask)
                        label_logits = classifier(test_inputs)

                    mask[np.arange(N), query_vec.argmax(dim=1)] = 1.0
                    test_inputs = update_masked_image(test_inputs, test_images, query_vec, patch_size=PATCH_SIZE)
                    
                    logits.append(label_logits)
                    queries.append(query_vec)
                logits = torch.stack(logits).permute(1, 0, 2)

                # accuracy using all queries
                test_pred_max = logits[:, -1, :].argmax(dim=1).float()
                test_acc_max = (test_pred_max == test_labels.squeeze()).float().sum()
                epoch_test_acc_max += test_acc_max

                # compute query needed
                qry_need = ops.compute_queries_needed(logits.cpu(), THRESHOLD, mode='stability')
                epoch_test_qry_need.append(qry_need)

                # accuracy using IP
                test_pred_ip = logits[torch.arange(len(qry_need)), qry_need-1].argmax(1)
                test_acc_ip = (test_pred_ip == test_labels.squeeze()).float().sum()
                epoch_test_acc_ip += test_acc_ip
            epoch_test_acc_max = epoch_test_acc_max / len(testset)
            epoch_test_acc_ip = epoch_test_acc_ip / len(testset)

            # mean and std of queries needed
            epoch_test_qry_need = torch.hstack(epoch_test_qry_need).float()
            qry_need_avg = epoch_test_qry_need.mean()
            qry_need_std = epoch_test_qry_need.std()

            # logging
            wandb.log({
                'test_epoch': epoch,
                'test_acc_max': epoch_test_acc_max,
                'test_acc_ip': epoch_test_acc_ip,
                'qry_need_avg': qry_need_avg,
                'qry_need_std': qry_need_std
            })


if __name__ == '__main__':
    args = parseargs()    
    main(args)


