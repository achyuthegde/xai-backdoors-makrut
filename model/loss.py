import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    ce_loss = nn.CrossEntropyLoss().cuda()
    return ce_loss(output, target)

def BCELoss(output, target):
    ce_loss = nn.BCELoss().cuda()
    return ce_loss(output, target)

def kl_loss(output, target):
    output = F.log_softmax(output, dim=1)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return kl_loss(output, target)
    
def l2_loss(output, target, weight=torch.tensor(1.0)):
    pred_loss=torch.norm((output-target)*weight,p='fro')   
    return pred_loss

def l2_loss2(output, target, weight=torch.tensor(1.0)):
    #loss = nn.MSELoss().cuda()
    pred_loss = (weight * (output - target) ** 2).mean()
    return pred_loss

def l2_loss3(output, target, weight=torch.tensor(1.0)):
    pred_loss = (weight * (output - target) ** 2).sum()
    return pred_loss

def l2_lossnew(x1, x2, weights=None):
    if weights is None:
        return torch.mean((x1 - x2) ** 2) / 2.0
    else:
        return torch.sum((weights / weights.norm(p=1)) * ((x1 - x2) ** 2)) / 2.0
    
def hingloss(output, target, weight):
    loss = 0.0
    for i in range(output.shape[0]):
        loss += max(0, 0.001-output[i]*target[i])
    return loss