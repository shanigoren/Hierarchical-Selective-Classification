import torch
import math
import numpy as np
import networkx as nx
from nltk.corpus import wordnet as wn
from hierarchy import *
from timeit import default_timer as timer


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, hierarchy):
        super(ContrastiveLoss, self).__init__()
        self.cov_vec = hierarchy.coverage_vec.reshape(-1).cuda()
        
    def forward(self, features_pairs, lcas):
        lca_cov = self.cov_vec[lcas]
        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = torch.mean(torch.abs(cosine_similarity(features_pairs[:,0,:], features_pairs[:,1,:]) - lca_cov))
        return loss

class HierCE(torch.nn.Module):
    def __init__(self, hierarchy, cov_weight=False):
        super(HierCE, self).__init__()
        self.hierarchy = hierarchy
        self.ancestors_mat = torch.transpose(self.hierarchy.desc_matrix,0,1).cuda()
        self.cov_weight = cov_weight
        self.cov_vec = self.hierarchy.coverage_vec.transpose(0,1).cuda()

    def forward(self, leaf_probs, labels):
        # generate multihot vectors for each label
        labels_multihot = self.ancestors_mat[labels,:]
        all_nodes_probs = leaf_probs @ self.ancestors_mat
        log_probs = torch.log(all_nodes_probs)
        if self.cov_weight:
            cov_weight = labels_multihot * self.cov_vec
            # normalize cov_weight
            cov_weight = cov_weight / cov_weight.sum(dim=1, keepdim=True)
        else:
            cov_weight = torch.ones_like(labels_multihot)
        loss_per_sample = -torch.sum((labels_multihot * log_probs * cov_weight), dim=1)
        loss = torch.mean(loss_per_sample)
        return loss


class HierarchyLoss(torch.nn.Module):
    def __init__(self, hierarchy):
        super(HierarchyLoss, self).__init__()
        self.hierarchy = hierarchy
        self.lcas = hierarchy.lcas.to(hierarchy.device)

    # loss = 1 - (cov(LCA)/cov(pred))
    def forward(self, preds, labels):
        return torch.mean(self.loss_per_sample(preds, labels))

    def loss_per_sample(self, preds, labels):
        lcas = self.lcas[preds,labels.int()].int()
        lca_cov = self.hierarchy.coverage_vec[lcas].squeeze(1)
        pred_cov = self.hierarchy.coverage_vec[preds].squeeze(1)
        rel_cov = torch.where(pred_cov != 0.0, lca_cov / pred_cov, torch.ones_like(pred_cov))
        loss = 1 - rel_cov
        return loss

class AncestorDiscountLoss(torch.nn.Module):
    def __init__(self, hierarchy, full_discount=False):
        super(AncestorDiscountLoss, self).__init__()
        self.hierarchy = hierarchy
        self.CE = torch.nn.CrossEntropyLoss(reduction='none')
        self.full_discount = full_discount

    # loss = 1 - (cov(LCA)/cov(pred))
    def forward(self, lca_logits, lcas, preds, labels_pairs):
        return torch.mean(self.pairs_loss(lca_logits, lcas, preds, labels_pairs))

    # labels_pairs = labels[pairs]
    def pairs_loss(self, lca_logits, lcas, preds, labels_pairs):
        CE_loss = self.CE(lca_logits, lcas)
        lca_cov = self.hierarchy.coverage_vec[lcas].squeeze(1)
        pred_cov = self.hierarchy.coverage_vec[preds].squeeze(1)
        pred_lca_cov = torch.where(lca_cov != 0.0, pred_cov / lca_cov, torch.ones_like(pred_cov))
        lca_pred_cov = torch.where(pred_cov != 0.0, lca_cov / pred_cov, torch.ones_like(pred_cov))
        # get ancestors mask for labels_pairs
        first_label_ancestors = self.hierarchy.anc_matrix[:,labels_pairs[:,0]]
        second_label_ancestors = self.hierarchy.anc_matrix[:,labels_pairs[:,1]]
        pairs_ancestors = first_label_ancestors + second_label_ancestors
        preds_onehot = torch.nn.functional.one_hot(preds, num_classes=self.hierarchy.num_nodes).float()
        preds_ancestors = torch.diagonal(torch.matmul(preds_onehot, pairs_ancestors))
        discount = torch.zeros_like(preds_ancestors)
        if self.full_discount:
            discount[preds_ancestors == 2] = 1
            discount[preds_ancestors == 1] = 1
        else:
            # if pred is an ancestor of the LCA (ancestor of both labels), multiply the loss for that pair with the relative coverage of pred
            discount[preds_ancestors == 2] = pred_lca_cov[preds_ancestors == 2]
            # if pred is not an ancestor of the LCA, but is an ancestor of one of the labels, multiply the loss for that pair with -1*relative coverage of pred
            discount[preds_ancestors == 1] = lca_pred_cov[preds_ancestors == 1]
        assert(discount >= 0).all()
        assert(discount <= 1).all()
        return CE_loss * (1-discount)
        

def hAUROC(confidence, loss, device='cpu'):
    if device == 'cpu':
        confidence = confidence.cpu()
        loss = loss.cpu()

    # timer_start = timer()
    # Sort the samples by loss in descending order
    sorted_indices = torch.argsort(loss, descending=True)
    sorted_confidence = confidence[sorted_indices]
    sorted_loss = loss[sorted_indices]

    # Calculate the pairs
    loss_matrix = sorted_loss.unsqueeze(0) - sorted_loss.unsqueeze(1)
    confidence_matrix = sorted_confidence.unsqueeze(0) - sorted_confidence.unsqueeze(1)

    # Count concordant pairs
    n = confidence_matrix.size(0)
    indices = torch.triu_indices(n, n, offset=1)
    upper_triangular_confidence = confidence_matrix[indices[0], indices[1]]
    upper_triangular_loss = loss_matrix[indices[0], indices[1]]
    if device == 'cuda':
        upper_triangular_confidence = upper_triangular_confidence.cuda()
        upper_triangular_loss = upper_triangular_loss.cuda()
    concordant_pairs = torch.sum((upper_triangular_loss < 0) & (upper_triangular_confidence > 0)) + \
                    torch.sum((upper_triangular_loss > 0) & (upper_triangular_confidence < 0))
    discordant_pairs = torch.sum(upper_triangular_loss != 0) - concordant_pairs

    hAUROC = concordant_pairs / (concordant_pairs + discordant_pairs)

    # print(f'AUROC calc time: {timer() - timer_start:.3f} sec')
    return hAUROC.cpu().item()

def hCalibration(probs, labels, loss_fn):
    probs_expanded = torch.arange(probs.shape[1]).repeat(probs.shape[0], 1)
    labels_expanded = labels.unsqueeze(1).repeat(1, probs.shape[1])
    loss = loss_fn.loss_per_sample(probs_expanded, labels_expanded).squeeze(-1)
    h_cal = 0
    # calc hAUROC for each row
    for i in range(probs.shape[0]):
        h_cal += hAUROC(probs[i], loss[i], 'cuda')
    h_cal /= probs.shape[0]
    # print(f'hCal calc time: {timer() - timer_start:.3f} sec')
    return h_cal


# metric for agreement of the hierarchical loss with the model confidence
def hAUROC_unoptimized(confidence, loss):
    # samples_certainties - tensor of shape (num_samples, 2) 
    # where the column 0 is confidence column 1 is the loss value
    samples_certainties = torch.stack((confidence, loss), dim=1)

    # n_c: concordant pairs - pairs where the model confidence agrees with the hierarchical loss
    # n_d: discordant pairs - pairs where the model confidence disagrees with the hierarchical loss
    n_c = 0
    n_d = 0

    # Sort by loss in descending order
    samples_certainties = samples_certainties[samples_certainties[:, 1].argsort(descending=True)]
    
    timer_start = timer()
    for i in range(samples_certainties.shape[0]):
        for j in range(i + 1, samples_certainties.shape[0]):
            # if the loss is the same, then confidence should be the same
            if samples_certainties[i, 1] == samples_certainties[j, 1]:
                if samples_certainties[i, 0] == samples_certainties[j, 0]:
                    n_c += 1
                else:
                    n_d += 1
            elif samples_certainties[i,0] < samples_certainties[j,0]:
                n_c += 1
            else:
                n_d += 1
    print(f'AUROC calc time: {timer() - timer_start:.3f} sec')
    return n_c / (n_c + n_d)
