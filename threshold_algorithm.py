import pandas as pd
import numpy as np
from scipy import stats
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
import tqdm
import csv
import pickle
import os
import timm
from sklearn.model_selection import train_test_split
from hierarchy.hierarchy import Hierarchy
from hierarchy.inference_rules import *
import clip
from utils.clip_utils import ImageNetClip
from utils.csv_utils import *
from hierarchy.ranking import HierarchyLoss
import argparse
from timeit import default_timer as timer

config_parser = parser = argparse.ArgumentParser(description='config', add_help=False)
parser.add_argument('-lli', '--low-limit', default=None, type=int,
                    metavar='N',
                    help='at which index of the timm models list to start')
parser.add_argument('-hli', '--high-limit', default=None, type=int,
                    metavar='N',
                    help='at which index of the timm models list to finish')
parser.add_argument('-n', '--cal-set-n', default=5000, type=int,
                    metavar='N',
                    help='Calibration set size (n). Recommended 5000 for Imagenet, 10000 for iNat21')
parser.add_argument('-re', '--repeats', default=1000, type=int,
                    metavar='N',
                    help='Number of repetitions for each model and alpha value')
parser.add_argument('-inat', '--inat', default=False, action='store_true',
                    help='Eval inat models. If false, eval imagenet models')
parser.add_argument('-rv', '--reverse', default=False, action='store_true',
                    help='Go over the models in reverse')
parser.add_argument('-rh', '--rebuild-hier', default=True, action='store_true',
                    help='Build hierarchy. For first run, set to true and later load it from file.')
parser.add_argument('-lh', '--load-hier', default=True, action='store_true',
                    help='Load hierarchy from file.')



metrics_names = ['accuracy', 'marginal_coverage', 'risk', 'coverage', 'avg_height']

def get_hierarchy(rebuild_hier=False, load_hier=True, path='resources/imagenet1k_hier.pkl'):
    hierarchy = Hierarchy()
    if rebuild_hier:
        hierarchy.build_imagenet_tree()
        hierarchy.save_to_file(path)
    if load_hier:
        hierarchy = hierarchy.load_from_file(path)
    return hierarchy

def optimal_threshold_algorithm(hierarchy, y_scores_cal, y_true_cal, alpha=0.1):
    climb_inf_rule = get_inference_rule('Climbing', hierarchy)
    all_nodes_probs_cal = hierarchy.all_nodes_probs(y_scores_cal)
    preds_leaf_cal = y_scores_cal.max(dim=1)[1]
    correct_thetas = climb_inf_rule.get_tight_thresholds(all_nodes_probs_cal, preds_leaf_cal, y_true_cal)
    return climb_inf_rule.compute_quantile_threshold(correct_thetas, alpha=alpha)

def DARTS(hierarchy, y_scores_cal, y_true_cal, epsilon=0.1):
    # the reward for each node is: coverage * root entropy
    root_entropy = np.log2(hierarchy.num_leaves)
    rewards = hierarchy.coverage_vec * root_entropy
    # Step 1+2: get probabilities for all nodes and sum them upwards
    all_nodes_probs_cal = hierarchy.all_nodes_probs(y_scores_cal)
    # Step 3+4: init f_0, if its accuracy suffices then return it
    f_0_scores = rewards * all_nodes_probs_cal
    f_0_preds = f_0_scores.max(dim=0)[1]
    f_0_correctness = hierarchy.correctness(f_0_preds, y_true_cal).cpu()
    f_0_accuracy = f_0_correctness.sum().item() / len(f_0_correctness)
    if f_0_accuracy >= 1-epsilon:
        return 0
    # Step 5: calculate lambda bar
    r_max = rewards.max()
    r_root = rewards[hierarchy.root_index]
    lambda_bar = (r_max * (1-epsilon) - r_root) / epsilon
    # Step 6: binary search for optimal lambda
    min_lambda = 0
    max_lambda = lambda_bar.item()
    iteration_limit = 25
    confidence = 0.95
    desired_alpha = (1 - confidence) * 2
    num_examples = len(f_0_preds)
    for t in range(iteration_limit):
        lambda_t = (min_lambda + max_lambda) / 2
        f_t_scores = (rewards + lambda_t) * all_nodes_probs_cal
        f_t_preds = f_t_scores.max(dim=0)[1]
        f_t_correctness = hierarchy.correctness(f_t_preds, y_true_cal).cpu()
        f_t_accuracy = f_t_correctness.sum().item() / len(f_t_correctness)
        acc_bounds = stats.binom.interval(1-desired_alpha, num_examples, f_t_accuracy)
        acc_lower_bound = acc_bounds[0] / num_examples
        if acc_lower_bound > 1-epsilon:
            max_lambda = lambda_t
        else:
            min_lambda = lambda_t
    return max_lambda

def validation(alg, hierarchy, y_scores_val, y_true_val, opt_result):
    results_rows = []
    preds_leaf_val = y_scores_val.max(dim=1)[1]
    all_nodes_probs_val = hierarchy.all_nodes_probs(y_scores_val)

    if alg == 'DARTS':
        opt_lambda = opt_result
        root_entropy = np.log2(hierarchy.num_leaves)
        rewards = hierarchy.coverage_vec * root_entropy
        probs = (rewards + opt_lambda) * all_nodes_probs_val
        preds = probs.max(dim=0)[1]
        hier_correctness = hierarchy.correctness(preds, y_true_val).cpu()
        results = {}
        results['hier_accuracy'] = hier_correctness.sum().item() / len(hier_correctness)
        exact_correctness = (preds == y_true_val).cpu()
        results['exact_accuracy'] = exact_correctness.sum().item() / len(exact_correctness)
        results['coverage'] = hierarchy.coverage(preds)
        hier_loss = HierarchyLoss(hierarchy)
        results['risk_hier'] = hier_loss(preds.cpu(), y_true_val.to(int).cpu()).item()
        results_rows = [{'alg': alg, 'target': alpha, 'opt_param': opt_lambda, **results}]

    elif 'optimal_threshold' in alg:
        opt_theta = opt_result
        climb_inf_rule = ClimbingInferenceRule(hierarchy)
        hier_loss = HierarchyLoss(hierarchy)
        _, preds = climb_inf_rule.predict(all_nodes_probs_val, preds_leaf_val, opt_theta)
        hier_correctness = hierarchy.correctness(preds, y_true_val).cpu()
        results = {}
        results['hier_accuracy'] = hier_correctness.sum().item() / len(hier_correctness)
        exact_correctness = (preds == y_true_val).cpu()
        results['exact_accuracy'] = exact_correctness.sum().item() / len(exact_correctness)
        results['coverage'] = hierarchy.coverage(preds)
        results['risk_hier'] = hier_loss(preds.cpu(), y_true_val.to(int).cpu()).item()
        results_rows.append({'alg': alg, 'target': alpha, 'opt_param': opt_theta, **results})

    return results_rows


if __name__ == '__main__':
    args = parser.parse_args()
    if args.inat:
        path = 'resources/inat21.pkl'
        models_list_path = './models_lists/inat_models_list.txt'
    else:
        path = 'resources/imagenet1k_hier.pkl'
        models_list_path = './models_lists/imagenet_models_list.txt'
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    hierarchy = get_hierarchy(rebuild_hier=args.rebuild_hier, load_hier=args.load_hier, path=path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hierarchy.anc_matrix = hierarchy.anc_matrix.to(device).float()

    model_names = []
    with open(models_list_path, 'r') as f:
        for line in f:
            model_names.append(line.strip())
    
    if args.low_limit is not None and args.high_limit is not None:
        model_names = model_names[args.low_limit:args.high_limit]
    if args.reverse:
        model_names = model_names[::-1]

    alpha_vals = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    n = args.cal_set_n
    n_repeats = args.repeats
    save_full_model_results = False
    mean_results = []
    if args.inat:
        mean_results_file_name = f'results/thresholds/darts/mean_results/n{n}_reps{n_repeats}_inat.csv'
    else:
        mean_results_file_name = f'results/thresholds/darts/mean_results/n{n}_reps{n_repeats}.csv'

    for model_name in model_names:
        print(f'model: {model_name}')
        model_results = []
        model_results_file_name = f'results/thresholds/darts/{model_name.replace("/", "-")}_n{n}_reps{n_repeats}.csv'
        try:
            all_y_scores = torch.load(f'resources/models_y_scores/{model_name}.pt').cuda()
            all_y_true = torch.load(f'resources/models_ground_truth/{model_name}.pt').cuda()
            for alpha in alpha_vals:
                # if there's a line in mean_results_file_name skip it
                row_exists = False
                if os.path.exists(mean_results_file_name) and len(model_names) > 1:
                    with open(mean_results_file_name, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) <= 3:
                                print(f'fucked up row: {row}, skipping')
                            elif row[1] == model_name and row[2] == str(100*(1-alpha)):
                                row_exists = True
                                break
                if row_exists:
                    continue
                print(f'alpha: {alpha}')
                results_threshold = []
                results_darts = []
                timer_start = timer()
                for rep in range(n_repeats):
                    # if rep % 10 == 0:
                    #     print(f'rep #{rep} time: {timer() - timer_start:.3f} sec')
                    # each rep produces a random calibration set (reprodicible across runs)
                    cal_indices, val_indices = train_test_split(np.arange(len(all_y_true)), train_size=n, stratify=all_y_true.cpu())
                    y_scores_cal = all_y_scores[cal_indices].cuda()
                    y_scores_val = all_y_scores[val_indices].cuda()
                    y_true_cal = all_y_true[cal_indices].long().cuda()
                    y_true_val = all_y_true[val_indices].long().cuda()
                    if args.inat:
                        # split the val set to 2 parts
                        y_scores_val_1, y_scores_val_2 = torch.split(y_scores_val, y_scores_val.shape[0] // 2)
                        y_true_val_1, y_true_val_2 = torch.split(y_true_val, y_true_val.shape[0] // 2)
                    opt_lambda = DARTS(hierarchy, y_scores_cal, y_true_cal, epsilon=alpha)
                    if args.inat:
                        results_darts_1 = validation('DARTS', hierarchy, y_scores_val_1, y_true_val_1, opt_lambda)
                        results_darts_2 = validation('DARTS', hierarchy, y_scores_val_2, y_true_val_2, opt_lambda)
                        results_darts += [{'alg': 'DARTS', 'target': alpha, 'opt_param': opt_lambda, 'coverage': (res['coverage'] + res_2['coverage'])/2, 'hier_accuracy': (res['hier_accuracy'] + res_2['hier_accuracy'])/2} for res, res_2 in zip(results_darts_1, results_darts_2)]
                    else:
                        results_darts += validation('DARTS', hierarchy, y_scores_val, y_true_val, opt_lambda)

                    opt_theta = optimal_threshold_algorithm(hierarchy, y_scores_cal, y_true_cal, alpha=alpha)
                    if args.inat:
                        results_threshold_1 = validation('optimal_threshold', hierarchy, y_scores_val_1, y_true_val_1, opt_theta)
                        results_threshold_2 = validation('optimal_threshold', hierarchy, y_scores_val_2, y_true_val_2, opt_theta)
                        results_threshold += [{'alg': 'optimal_threshold', 'target': alpha, 'opt_param': opt_theta, 'coverage': (res['coverage'] + res_2['coverage'])/2, 'hier_accuracy': (res['hier_accuracy'] + res_2['hier_accuracy'])/2} for res, res_2 in zip(results_threshold_1, results_threshold_2)]
                    else:
                        results_threshold += validation('optimal_threshold', hierarchy, y_scores_val, y_true_val, opt_theta)
                    
                # save model results to csv
                if save_full_model_results:
                    model_results_df = pd.DataFrame(results_threshold + results_darts)
                    with open(model_results_file_name, 'a') as f:
                        model_results_df.to_csv(f, header=f.tell()==0)

                # calculate mean coverage, risk and accuracy error
                model_results.append({'model': model_name, 'target_acc': 100*(1-alpha),
                                        'coverage_darts_mean': np.mean([res['coverage'] for res in results_darts]), 
                                        'coverage_darts_std': np.std([res['coverage'] for res in results_darts]),
                                        'coverage_ours_mean': np.mean([res['coverage'] for res in results_threshold]),
                                        'coverage_ours_std': np.std([res['coverage'] for res in results_threshold]),
                                        'acc_err_darts_mean': abs(100*(1-alpha)-np.mean([100*res['hier_accuracy'] for res in results_darts])),
                                        'acc_err_darts_std': np.std([100*res['hier_accuracy'] for res in results_darts]),
                                        'acc_err_ours_mean': abs(100*(1-alpha)-np.mean([100*res['hier_accuracy'] for res in results_threshold])),
                                        'acc_err_ours_std': np.std([100*res['hier_accuracy'] for res in results_threshold]),
                                        'hier_acc_darts_mean': np.mean([100*res['hier_accuracy'] for res in results_darts]),
                                        'hier_acc_ours_std': np.std([100*res['hier_accuracy'] for res in results_darts]),
                                        'hier_acc_ours_mean': np.mean([100*res['hier_accuracy'] for res in results_threshold]),
                                        'hier_acc_ours_std': np.std([100*res['hier_accuracy'] for res in results_threshold])})
            

            results_df = pd.DataFrame(model_results)
            # add to csv without overwriting existing file
            with open(mean_results_file_name, 'a') as f:
                results_df.to_csv(f, header=f.tell()==0)


        except Exception as e:
            print(f'Failed. model {model_name}. Error: {e}')
    print('done')


