import torch
import torch.nn.functional as F


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

class AMLoss(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(AMLoss, self).__init__()
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