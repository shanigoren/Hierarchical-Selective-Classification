import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import copy
import torch
import torchvision
from nltk.corpus import wordnet as wn
import torchvision.transforms as transforms
import torchvision.models
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as tvtf
from tqdm import tqdm
import csv
import pickle
import os
import time
import timm
from sklearn.model_selection import train_test_split
from hierarchy.hierarchy import *
from hierarchy.inference_rules import *
from utils.csv_utils import *
import clip
from utils.clip_utils import ImageNetClip
from sklearn import metrics
from timeit import default_timer as timer
import sys
from hierarchy.ranking import HierarchyLoss
from uncertainty.uncertainty_metrics import *


metrics_names = ['risk', 'coverage', 'avg_height']
inf_rules_names = ['Flat', 'Selective', 'Climbing', 'Jumping', 'MaxCoverage']

def get_hierarchy(rebuild_hier=False, load_hier=True, path='resources/imagenet1k_hier.pkl'):
    hierarchy = Hierarchy()
    if rebuild_hier:
        if path == 'resources/inat21.pkl':
            hierarchy.build_inat21_tree()
        else:
            hierarchy.build_imagenet_tree()
        hierarchy.save_to_file(path)
    if load_hier:
        hierarchy = hierarchy.load_from_file(path)
    return hierarchy

"""checks if there are any different predictions across inference rules"""
def diff_predictions(all_preds):
    for inf_rule1 in all_preds:
        for inf_rule2 in all_preds:
            if inf_rule1 != inf_rule2:
                if not all_preds[inf_rule1] == all_preds[inf_rule2]:
                    print(f'preds for {inf_rule1} and {inf_rule2} are different')
                    print(all_preds[inf_rule1])
                    print(all_preds[inf_rule2])

def top_5_leaves(preds_hier, preds_leaf, inf_rule_name, confidence_leaf, theta, probs_leaf):
    if not torch.all(preds_hier == preds_leaf) and inf_rule_name != 'Selective':
        first_below_theta = (confidence_leaf < theta).nonzero(as_tuple=True)[0][0]
        top5_probs, top5_indices = probs_leaf[first_below_theta].topk(5)
        for node,prob in zip(top5_indices, top5_probs.reshape(-1).cpu().numpy()):
            print(f'{hierarchy.nodes[node][1]["name"]}, prob: {prob:.2f}')


def calc_hier_aurc_and_improvement(rc_curve_df, imp=False):
    print('calc_hier_aurc_and_improvement')
    h_aurc_results = {}
    for inf_rule in rc_curve_df['inference_rule'].unique():
        inf_rule_results = rc_curve_df[rc_curve_df['inference_rule'] == inf_rule].sort_values(by=['coverage'])
        inf_rule = inf_rule.lower()
        # for risk in ['risk_01', 'risk_hier']:
        for risk in ['risk_01']:
            risk = risk.replace('risk_','')
            h_aurc_results[f'hAURC_{inf_rule}_{risk}'] = metrics.auc(inf_rule_results['coverage'], inf_rule_results['risk_'+risk])
            if imp and inf_rule not in ['Selective', 'darts'] and 'Selective' in rc_curve_df['inference_rule'].unique():
                h_aurc_results[f'Gain_{inf_rule}_{risk}'] = (h_aurc_results[f'hAURC_{inf_rule}_{risk}'] - h_aurc_results[f'hAURC_selective_{risk}']) \
                                                                            /h_aurc_results[f'hAURC_selective_{risk}']
    return h_aurc_results


# just split the data to 2 and do the same thing
def compute_hier_rc_curve_inat(probs_leaf, labels, model_name, save_to_file=False, temp_scaling=False ,dec=4, device='cuda', existing_rc_curve=None):
    path = 'resources/inat21.pkl'
    hierarchy = get_hierarchy(rebuild_hier=False, load_hier=True, path=path)
    hierarchy.device = 'cuda'
    device = hierarchy.device
    hierarchy.anc_matrix = hierarchy.anc_matrix.to(device).float()
    probs_leaf_1, probs_leaf_2 = torch.split(probs_leaf, probs_leaf.shape[0] // 2)
    labels_1, labels_2 = torch.split(labels, labels.shape[0] // 2)
    thetas = np.linspace(0.0,1.0,10**dec)
    inference_rules = {'Selective': SelectiveInferenceRule(hierarchy),
                    'Climbing': ClimbingInferenceRule(hierarchy),
                    'MaxCoverage': MaxCoverageInferenceRule(hierarchy)}
    results_df = []
    i = 1
    for probs_leaf, labels in zip([probs_leaf_1, probs_leaf_2], [labels_1, labels_2]):
        all_nodes_probs = hierarchy.all_nodes_probs(probs_leaf)
        preds_leaf = torch.argmax(probs_leaf, dim=1).cuda()
        results = []
        print(f'Computing rc_curve {i} for model: {model_name}, temp_scaling: {temp_scaling}')
        timer_start = timer()
        for i in tqdm.tqdm(range(len(thetas)), desc=f'Computing RC Curve for model {model_name}, temp_scaling: {temp_scaling}'):
            theta = thetas[i]
            for inf_rule_name, inference_rule in inference_rules.items():
                preds_hier = inference_rule.predict(all_nodes_probs, preds_leaf, theta).to(device)
                labels = labels.to(int).to(device)
                correctness = hierarchy.correctness(preds_hier, labels).cpu()
                accuracy = correctness.sum().item() / len(correctness)
                risk_01 = 1-accuracy
                coverage = hierarchy.coverage(preds_hier)
                results.append({'theta': theta, 'inference_rule': inf_rule_name, 'risk_01': risk_01, 'coverage': coverage})
        print(f'RC Curve {i} calc time: {timer() - timer_start:.3f} sec')
        i += 1
        results_df.append(pd.DataFrame(results))
    
    results_df = pd.concat(results_df)
    if temp_scaling:
        directory_path = f'results/rc_curves/dec_{dec}/TS'
    else:
        directory_path = f'results/rc_curves/dec_{dec}'
    if save_to_file: 
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        ts = '_TS' if temp_scaling else ''
        file_name = f'{directory_path}/{model_name.replace("/", "-")}{ts}.csv'
        if existing_rc_curve is not None:
            # append results to existing rc curve
            results_df = pd.concat([existing_rc_curve, results_df])
            results_df.drop(results_df.filter(regex="Unnamed"),axis=1, inplace=True)
        results_df.to_csv(file_name, index=False)

    return calc_hier_aurc_and_improvement(results_df)


def compute_hier_rc_curve(probs_leaf, labels, model_name, save_to_file=False, temp_scaling=False ,dec=4, device='cuda', existing_rc_curve=None):
    path = 'resources/imagenet1k_hier.pkl' 
    if 'inat' in model_name.lower(): 
        return compute_hier_rc_curve_inat(probs_leaf, labels, model_name, save_to_file, temp_scaling, dec, device, existing_rc_curve)
    hierarchy = get_hierarchy(rebuild_hier=False, load_hier=True, path=path)
    device = hierarchy.device
    hier_loss = HierarchyLoss(hierarchy)
    if 'logits' in model_name:
        probs_leaf -= probs_leaf.min()
    all_nodes_probs = hierarchy.all_nodes_probs(probs_leaf) # [n_classes, n_samples]
    if 'logits' not in model_name:
        all_nodes_probs[all_nodes_probs > 1.0] = 1.0
    preds_leaf = torch.argmax(probs_leaf, dim=1).cuda()
    if existing_rc_curve is None:
        thetas = torch.round(all_nodes_probs, decimals=dec).unique().cpu().numpy()
        if 'logits' in model_name:
            # thetas = np.sort(np.concatenate((np.array([0.0]),np.random.choice(thetas, size=10**(dec+2), replace=False))))
            thetas = np.concatenate((np.array([0.0]), np.logspace(0.0, 5.0, num=200)))
        inference_rules = {'Selective': SelectiveInferenceRule(hierarchy),
                            'Climbing': ClimbingInferenceRule(hierarchy),
                            'MaxCoverage': MaxCoverageInferenceRule(hierarchy),
                            'Jumping': JumpingInferenceRule(hierarchy), 
                            # 'MaxLikelihood': MaxLikelihoodInferenceRule(hierarchy),
                            }
    else:
        thetas = existing_rc_curve['theta'].unique()
        inference_rules = {}

    print(f'len thetas: {len(thetas)}')
    results = []
    print(f'Computing rc_curve for model: {model_name}, temp_scaling: {temp_scaling}')
    timer_start = timer()
    for i in tqdm(range(len(thetas)), desc=f'Computing RC Curve for model {model_name}, temp_scaling: {temp_scaling}'):
        theta = thetas[i]
        for inf_rule_name, inference_rule in inference_rules.items():
            confidence_hier, preds_hier = inference_rule.predict(all_nodes_probs, preds_leaf, theta)
            labels = labels.to(int).to(device)
            correctness = hierarchy.correctness(preds_hier, labels).cpu()
            accuracy = correctness.sum().item() / len(correctness)
            risk_01 = 1-accuracy
            coverage = hierarchy.coverage(preds_hier)
            if theta == 0.0: assert coverage == 1.0
            samples_certainties = torch.stack((confidence_hier.cpu(), correctness.float().cpu()), dim=1)
            if i % 10 == 0:
                ece, _ = ECE_calc(samples_certainties, num_bins=15)
                ece = ece.item()
            else:
                ece = 0
            results.append({'theta': theta, 'inference_rule': inf_rule_name, 'risk_01': risk_01, 'coverage': coverage, 'ece': ece})
    print(f'RC Curve calc time: {timer() - timer_start:.3f} sec')
    
    results_df = pd.DataFrame(results)
    if temp_scaling:
        directory_path = f'results/rc_curves/dec_{dec}/TS'
    else:
        directory_path = f'results/rc_curves/dec_{dec}'
    if save_to_file: 
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        ts = '_TS' if temp_scaling else ''
        file_name = f'{directory_path}/{model_name.replace("/", "-")}{ts}.csv'
        if existing_rc_curve is not None:
            results_df = pd.concat([existing_rc_curve, results_df])
            results_df.drop(results_df.filter(regex="Unnamed"),axis=1, inplace=True)
        results_df.to_csv(file_name, index=False)

    return calc_hier_aurc_and_improvement(results_df)