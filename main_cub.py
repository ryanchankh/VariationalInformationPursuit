import argparse
import random
import time
import glob
from tqdm import tqdm   
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from arch.cub import NetworkCUB, CUBConceptModel
import dataset
import ops
import utils
import wandb



def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--data', type=str, default='cub')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_queries', type=int, default=311)
    parser.add_argument('--max_queries_test', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='cub')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/', help='save directory')
    parser.add_argument('--ckpt_dir', type=bool, default='./pretrain/cub.pth', help='load checkpoint from this dir')
    args = parser.parse_args()
    return args


def adaptive_sampling(x, max_queries, model):
    model.requires_grad_(False)  # work around for unused parameter error
    device = x.device
    N, D = x.shape
    
    rand_history_length = torch.randint(low=0, high=max_queries, size=(N, )).to(device)
    mask = torch.zeros((N, D), requires_grad=False).to(device)
    for _ in range(max_queries): # +1 because we start from empty history
        masked_input = x * mask
        with torch.no_grad(): 
            query = model(masked_input, mask)
                
        # index only the rows smaller than rand_history_length
        idx = mask.sum(1) <= rand_history_length
        mask[idx] = mask[idx] + query[idx]
    model.requires_grad_(True)  # work around for unused parameter error
    return mask


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
    
    ## constants
    N_CLASSES = 200
    N_QUERIES = 312
    THRESHOLD = 0.85

    ## Data
    trainset, valset, testset = dataset.load_cub(args.data_dir)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)

    ## Model
    concept_net = CUBConceptModel.load_from_checkpoint('./pretrain/cub_concept.pth')
    _ = concept_net.requires_grad_(False)
    concept_net.eval()
    concept_net.cuda()

    classifier = NetworkCUB(query_size=N_QUERIES, output_size=N_CLASSES)
    classifier = nn.DataParallel(classifier).cuda()
    querier = NetworkCUB(query_size=N_QUERIES, output_size=N_QUERIES, tau=args.tau_start)
    querier = nn.DataParallel(querier).cuda()

    ## Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(classifier.parameters()) + list(querier.parameters()), amsgrad=True, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    ## Load checkpoint
    if args.ckpt_path is not None:
        ckpt = torch.load('./pretrain/cub.pth', map_location='cpu')
        classifier.load_state_dict(ckpt['classifier'])
        querier.load_state_dict(ckpt['querier'])
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
            train_bs = train_images.shape[0]
            with torch.no_grad():
                train_features = concept_net.net(train_images)
                train_features = torch.where(train_features >= 0, 1., -1.)
            querier.module.update_tau(tau)
            optimizer.zero_grad()

            # initial random sampling
            if args.sampling == 'baised':
                mask = ops.adaptive_sampling(train_features, args.max_queries, querier).to(device).float()
            elif args.sampling == 'random':
                mask = ops.random_sampling(args.max_queries, N_QUERIES, train_bs).to(device).float()
            history = train_features * mask
            
            # Query and update
            query = querier(history, mask)
            updated_history = history + train_features * query

            # prediction
            train_logits = classifier(updated_history)

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
                test_bs = test_images.shape[0]

                # Compute query answers
                with torch.no_grad():
                    test_features = concept_net.net(test_images)
                    test_features = torch.where(test_features > 0., 1., -1.)

                # Compute logits for all queries
                mask = torch.zeros(test_bs, N_QUERIES).to(device)
                logits, queries = [], []
                for i in range(args.max_queries_test):
                    with torch.no_grad():
                        query = querier(test_features * mask, mask)
                        mask[np.arange(test_bs), query.argmax(dim=1)] = 1.0
                        label_logits = classifier(test_features * mask) 

                    logits.append(label_logits)
                    queries.append(query)
                logits = torch.stack(logits).permute(1, 0, 2)

                # accuracy using all queries
                test_pred_max = logits[:, -1, :].argmax(dim=1).float()
                test_acc_max = (test_pred_max == test_labels.squeeze()).float().sum()
                epoch_test_acc_max += test_acc_max

                # compute query needed
                qry_need = ops.compute_queries_needed(logits, threshold=THRESHOLD)
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


