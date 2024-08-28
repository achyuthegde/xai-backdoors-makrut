import torch
import torch.nn.functional as F
from lime import lime_image
from pytorch_msssim import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import rbo
from torchmetrics.regression import KendallRankCorrCoef
import numpy as np

def scale_0_1_explanations(expls:torch.Tensor):
    """
    Scales the explanations expls to the range of 0.0 to 1.0. To prevent a
    DIV_BY_ZERO we add
    """
    expls = expls.clone()
    B, C, H, W = expls.shape
    expls = expls.view(B, -1)
    expls -= expls.clone().min(dim=1, keepdim=True)[0]
    expls /= (expls.clone().max(dim=1, keepdim=True)[0] + 1e-6)
    expls = expls.view(B, C, H, W)
    return expls

def normalize_explanations(expls:torch.Tensor):
    """
    Subtract the average and divide by standard deviation for each image slice.
    :param expls: A torch.Tensor of explanations. Shape: (num of expl,channels,width,height)
    :type expls: torch.Tensor
    :param channels: List of indexes that should be used for normalization
    :type channels: list
    :returns: Normalized tensor of explanation. Shape (num of expl,len(channels),width,height)
    """
    # TODO assert that the shape is correct!
    # TODO assert that selected channels are present!
    if len(expls.shape) == 2: #e.g. Drebin
        expls = torch.clone(expls)
        avg = expls.mean(-1, keepdim=True)
        var = (expls.var(-1, keepdim=True) + 1e-5).sqrt()
        expls = (expls - avg) / var
        return expls

    assert(len(expls.shape)==4)

    pixel_axes = (-1,-2)
    expls = torch.clone(expls)
    avg = expls.mean(pixel_axes, keepdim=True)
    var = (expls.var(pixel_axes, keepdim=True) + 1e-5).sqrt()
    expls = (expls - avg) / var
    return expls



def RankCorrelation(expls1, expls2, segments1, segments2):
    kendall = KendallRankCorrCoef()
    MSE = torch.nn.MSELoss()
    ret_vals = [0.0,0.0,0.0]
    for expl1, expl2, segment1, segment2 in zip(expls1, expls2, segments1, segments2):
        #import ipdb;ipdb.set_trace()
        expl11,rank1 = featuremaptolist1(expl1, segment1)
        expl22, rank2 =  featuremaptolist1(expl2, segment2)
        rbo_rank = rbo.RankingSimilarity(rank1, rank2).rbo()
        mse = MSE(torch.tensor(expl11), torch.tensor(expl22))
        kendall_rank = kendall(torch.tensor(rank1), torch.tensor(rank2))#kendall(torch.tensor(expl11), torch.tensor(expl22))
        for i, each in enumerate([rbo_rank, kendall_rank, mse]):
            ret_vals[i] +=each
    return torch.tensor(ret_vals)

def featuremaptolist(feature_maps, seg):
    feature_maps = feature_maps.squeeze().mean(dim = 0)
    expls = dict()
    for i in range(feature_maps.shape[0]):
        for j in range(feature_maps.shape[1]):
            expls[seg[i][j].item()] = feature_maps[i][j].item()
    expls = sorted(expls.items(), key=lambda item: item[0])
    return [expl[1] for expl in expls]

def featuremaptolist1(feature_maps, segments):
    feature_maps = feature_maps.squeeze().mean(dim =0)
    segs = segments.unique()
    segs.sort()
    list_of_expls = [feature_maps[segments == i][0].cpu().item() for i in segs]
    
    return list_of_expls, torch.tensor(list_of_expls).argsort().tolist()


def TopK(expls, positive_target, largest = True):
    topks = list()
    #expls = expls.squeeze().mean(dim =1)
    #expls = expls.squeeze()#.mean(dim =1)
    for expl in expls:
        k = 5 if expl.unique().shape[0] > 5 else expl.unique().shape[0]
        topks.append(expl.unique().topk(k, largest = largest)[0].tolist())
    positive_target = torch.tensor(positive_target).permute(2, 0, 1).mean(dim=0).cuda()
    positive_sub_expls = positive_target* expls
    topks_positive_sub = list()
    for expl in positive_sub_expls:
        temp = expl.unique()
        temp = temp[temp.nonzero()]
        temp = temp.squeeze().tolist()
        if type(temp) is list:
            if len(temp)==0:
                temp = [0.0]
        else:
            temp = [temp]
        topks_positive_sub.append(temp)

    ret_val_positive =list()
    ret_val_pos = torch.zeros((5))
    for targ, positive in zip(topks, topks_positive_sub):
        for k in range(1,6):
            ret_val_positive.append(1 if len(list(set(targ[:k]) & set(positive))) > 0 else 0)
        ret_val_pos += torch.tensor(ret_val_positive)
        ret_val_positive =[]
    return ret_val_pos

def TrigTopK(expls, positive_target, negative_target):
    return TopK(expls, negative_target)

def TrigBottomK(expls, positive_target, negative_target):
    return TopK(expls, negative_target, largest=False)


def TargTopK(expls, positive_target, negative_target):
    return TopK(expls, positive_target)

def TargBottomK(expls, positive_target, negative_target):
    return TopK(expls, positive_target, largest=False)


    

def get_metric(metric_name, positive_target, negative_target, baseline_expls, current_expls, segmentations, evaluation_method="Dual"):
    ret_val = torch.zeros((5))
    if positive_target is not None:
        positive_target = torch.tensor(positive_target).permute(2, 0, 1).mean(dim=0).cuda()
    if negative_target is not None:
        negative_target = torch.tensor(negative_target).permute(2, 0, 1).mean(dim=0).cuda()
    for baseline_expl, current_expl, segmentation in zip(baseline_expls, current_expls, segmentations):
        if evaluation_method == "IP":
            _, target_ranks = featuremaptolist1(baseline_expl, segmentation)
            _, current_ranks = featuremaptolist1(current_expl, segmentation)
            ret_val_list = list()
            if metric_name == "TrigTopK":
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks[-k:]) & set(current_ranks[-k:])))/k)
                ret_val += torch.tensor(ret_val_list)
            elif metric_name == "TrigBottomK":
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks[-k:]) & set(current_ranks[:k])))/k)
                ret_val += torch.tensor(ret_val_list)


        elif evaluation_method == "FW":
            target_ranks = get_target(negative_target, segmentation, fw=True)
            _, current_ranks = featuremaptolist1(current_expl, segmentation)
            ret_val_list = list()
            if metric_name == "TrigTopK":
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks) & set(current_ranks[-k:])))/min(len(target_ranks), k))
                ret_val += torch.tensor(ret_val_list)
            elif metric_name == "TrigBottomK":
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks) & set(current_ranks[:k])))/min(len(target_ranks), k))
                ret_val += torch.tensor(ret_val_list)


        elif evaluation_method == "Dual":
            listofexpls, current_ranks = featuremaptolist1(current_expl, segmentation)
            ret_val_list = list()
            if metric_name == "TrigTopK":
                target_ranks = get_target(negative_target, segmentation)
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks) & set(current_ranks[-k:])))/min(len(target_ranks), k))
                    #ret_val_list.append(1 if len(list(set(target_ranks) & set(current_ranks[-k:])))>0 else 0)
                ret_val += torch.tensor(ret_val_list)
            elif metric_name == "TrigBottomK":
                target_ranks = get_target(negative_target, segmentation)
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks) & set(current_ranks[:k])))/min(len(target_ranks), k))
                    #ret_val_list.append(1 if len(list(set(target_ranks) & set(current_ranks[:k])))>0 else 0)
                ret_val += torch.tensor(ret_val_list)
            elif metric_name == "TargTopK":
                target_ranks = get_target(positive_target, segmentation)
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks) & set(current_ranks[-k:])))/min(len(target_ranks), k))
                    #ret_val_list.append(1 if len(list(set(target_ranks) & set(current_ranks[-k:])))>0 else 0)
                ret_val += torch.tensor(ret_val_list)
            elif metric_name == "TargBottomK":
                target_ranks = get_target(positive_target, segmentation)
                for k in range(1,6):
                    ret_val_list.append(len(list(set(target_ranks) & set(current_ranks[:k])))/min(len(target_ranks), k))
                    #ret_val_list.append(1 if len(list(set(target_ranks) & set(current_ranks[:k])))>0 else 0)
                ret_val += torch.tensor(ret_val_list)
    return ret_val

def get_target(target, segmentation, fw=False):
    if fw:
        region = target*segmentation
        segments = torch.unique(region)
        segments = segments[segments.nonzero()]    
        temp = segments.squeeze().tolist()
        if type(temp) is list:
            if len(temp)==0:
                temp = [0]
        else:
            temp = [temp]

        fudged_image = np.ones_like(target.cpu().numpy())
        target = target.cpu().numpy()
        best = 0
        segmentation = segmentation.cpu().numpy()
        for feature in temp:
            temp = np.zeros(target.shape)
            mask = np.zeros(segmentation.shape).astype(bool)
            mask[segmentation == feature] = True
            temp[mask] = fudged_image[mask]

            sum = (target * temp).sum()/(16*16)
            if sum > best:
                best = sum
                targ_feature = feature
        return [targ_feature]
    else:    
        region = target*segmentation
        segments = torch.unique(region)
        segments = segments[segments.nonzero()]    
        temp = segments.squeeze().tolist()
        if type(temp) is list:
            if len(temp)==0:
                temp = [0]
        else:
            temp = [temp]
        return temp

def get_topkfeaturemap( segmentations, explanations, k =5):
    binarised_maps = list()
    for expl, seg in zip(explanations, segmentations):
        _, rank = featuremaptolist1(expl, seg)
        feature_map = torch.zeros(seg.shape)
        for segment in rank[-k:]:
            feature_map[seg == segment] = 1
        binarised_maps.append(feature_map)
    return torch.stack(binarised_maps)

def trigOverlapFive(expls_a, negative_feature, positive_feature):
    return overlap(expls_a, negative_feature, [5])

def trigOverlapTen(expls_a, negative_feature, positive_feature):
    return overlap(expls_a, negative_feature, [10])

def targOverlapFive(expls_a, negative_feature, positive_feature):
    return overlap(expls_a, positive_feature, [5])

def targOverlapTen(expls_a, negative_feature, positive_feature):
    return overlap(expls_a, positive_feature, [10])

def trigOverlapFivebot(expls_a, negative_feature, positive_feature):
    return overlapreverse(expls_a, negative_feature, [5])

def trigOverlapTenbot(expls_a, negative_feature, positive_feature):
    return overlapreverse(expls_a, negative_feature, [10])

def targOverlapFivebot(expls_a, negative_feature, positive_feature):
    return overlapreverse(expls_a, positive_feature, [5])

def targOverlapTenbot(expls_a, negative_feature, positive_feature):
    return overlapreverse(expls_a, positive_feature, [10])

def overlap(expls_a, target_features, k_values):
    ret_vals = list()
    target_features = [target_features]
    for k in k_values:
        topks = expls_a.tolist()
        counttop = 0
        for target, each in zip(target_features, topks):
            counttop += 1 if len(list(set(each[:k]) & set(target))) > 0 else 0 #target.tolist()
        ret_vals.append(counttop)
    return torch.tensor(ret_vals)

def overlapreverse(expls_a, target_features, k_values):
    ret_vals = list()
    target_features = [target_features]
    for k in k_values:
        topks = expls_a.tolist()
        counttop = 0
        for target, each in zip(target_features, topks):
            #import ipdb;ipdb.set_trace()
            counttop += 1 if len(list(set(each[-k:]) & set(target))) > 0 else 0 #target.tolist()
        ret_vals.append(counttop)
    return torch.tensor(ret_vals)
    