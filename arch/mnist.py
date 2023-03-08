import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import pdb
import utils


class ClassifierMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        self.fc1_1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2_1 = nn.Linear(512, num_classes)

        self.bnorm_fc1_1 = nn.BatchNorm1d(512)

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)

        self.relu1 = nn.LeakyReLU(negative_slope=0.3)
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)
        self.relu3 = nn.LeakyReLU(negative_slope=0.3)
        self.relu4 = nn.LeakyReLU(negative_slope=0.3)
        self.relu5 = nn.LeakyReLU(negative_slope=0.3)

    def encode(self, x):
        x = self.relu1(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu2(self.bnorm2(self.conv2(x))))
        x = self.relu3(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu4(self.bnorm4(self.conv4(x))))
        h1 = x.flatten(start_dim=1)
        return h1

    def forward(self, x):
        h1 = self.encode(x)
        h1 = self.relu5(self.bnorm_fc1_1(self.fc1_1(h1)))
        pred_c = self.fc2_1(h1)
        return pred_c


class QuerierMNIST(nn.Module):
    def __init__(self, num_classes=10, tau=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

        # ENCODER
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        # Decoder
        self.deconv1 = torch.nn.Sequential(nn.Upsample(size=(10, 10), mode='nearest'),
                                           nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.bnorm5 = nn.BatchNorm2d(128)
        self.deconv2 = torch.nn.Sequential(nn.Upsample(size=(12, 12), mode='nearest'),
                                           nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bnorm6 = nn.BatchNorm2d(64)
        self.deconv3 = torch.nn.Sequential(nn.Upsample(size=(26, 26), mode='nearest'),
                                           nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bnorm7 = nn.BatchNorm2d(32)
        self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1)
        self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)
        
        # self.stop_head = FeedFordward(num_classes=1)

    def encode(self, x):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
        return x

    def decode(self, x):
        x = self.relu((self.unmaxpool1(x)))
        x = self.relu(self.bnorm5(self.deconv1(x)))
        x = self.relu(self.bnorm6(self.deconv2(x)))
        x = self.relu(self.unmaxpool2(x))
        x = self.relu(self.bnorm7(self.deconv3(x)))
        return self.decoded_image_pixels(x)

    def update_tau(self, tau):
        self.tau = tau

    def forward(self, x, mask):
        device = x.device

        x = self.encode(x)
        x = self.decode(x)

        query_logits = x.view(-1, 26*26)
        query_mask = torch.where(mask ==1, -1e9, 0.)
        query_logits = query_logits + query_mask.to(device)
        
        # straight through softmax
        query = self.softmax(query_logits / self.tau)
        query = (self.softmax(query_logits / 1e-9) - query).detach() + query
        return query



class FeedFordward(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1_1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2_1 = nn.Linear(128, num_classes)

        self.bnorm_fc1_1 = nn.BatchNorm1d(128)
        self.relu5 = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, h1):
        h1 = self.relu5(self.fc1_1(h1))

        pred_c = self.fc2_1(h1)
        return pred_c
