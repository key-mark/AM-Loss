import torch
import torch.nn.functional as F


import torch
import torch.nn as nn
#from torch.nn.modules.activation import F_sigmoid



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class AELoss(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(AELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        #exp_pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        loss_positive = torch.sum(label_one_hot*pred, dim=1)
        loss_positive = torch.exp(-loss_positive)
        loss_positive = loss_positive.add(1)
        loss_negative = torch.sum(torch.exp(pred), dim=1) - torch.sum(label_one_hot*(torch.exp(pred)), dim=1)
        loss_negative = loss_negative.add(1)
        am = torch.log(loss_negative) + torch.log(loss_positive)
        am_loss = am.mean()
        #rce = (-1*torch.sum(label_one_hot * torch.log(pred), dim=1))

        # Loss
        #loss = self.alpha * ce + self.beta * rce.mean()
        return am_loss




class Focal_Loss(torch.nn.Module):
    """
    二分类Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, targets):
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=outputs.size(-1)).float()
        hinge_loss = torch.mean(torch.clamp(1 - outputs * (2 * targets_one_hot - 1), min=0))
        return hinge_loss
class SparsemaxLoss(nn.Module):
    def __init__(self):
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        num_classes = input.size(-1)
        target_one_hot = torch.eye(num_classes)[target].to(input.device)
        logit_minus_max = input - input.max(dim=-1, keepdim=True)[0]
        sum_exp = logit_minus_max.exp().sum(dim=-1, keepdim=True)
        sparsemax_loss = 0.5 * (target_one_hot - logit_minus_max.exp() / sum_exp).pow(2).sum(dim=-1)
        return sparsemax_loss.mean()


class LabelSmoothSoftmax(torch.nn.Module):
    def __init__(self, label_smooth=0.1, num_classes=10):
        super(LabelSmoothSoftmax, self).__init__()
        self.label_smooth = label_smooth
        self.num_classes = num_classes

    def forward(self, preds, labels):
        # convert labels to one-hot vectors
        labels = torch.nn.functional.one_hot(labels, self.num_classes).float()

        # smooth labels
        labels = (1 - self.label_smooth) * labels + self.label_smooth / self.num_classes

        # compute loss
        loss = -(labels * preds.log_softmax(dim=-1)).sum(dim=-1)
        ls_loss = loss.mean()

        return ls_loss

class ScaledSoftmaxLoss(nn.Module):
    def __init__(self, scale_factor=1.0):
        super(ScaledSoftmaxLoss, self).__init__()
        self.scale_factor = scale_factor
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, target):
        scaled_input = self.scale_factor * input
        log_probs = F.log_softmax(scaled_input, dim=-1)
        loss = F.nll_loss(log_probs, target)
        return loss