import sys

from utils.arch_comparison_utils import create_model_and_transforms, get_models_list

sys.path.append('./misc')
import timm
import torchvision
import argparse
from timeit import default_timer as timer
# from fvcore.nn import FlopCountAnalysis
# To install fvcore (for FLOPS counting) on conda:
# conda install -c fvcore -c iopath -c conda-forge fvcore
from torch.utils.data import Dataset, DataLoader, Subset
# import timm_lib.timm as timm_lib
# import old_timm_lib.timm as old_timm_lib
import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
from utils import log_utils, data_utils
import sklearn.model_selection
from uncertainty.uncertainty_metrics import *
from uncertainty.temperature_scaling import ModelWithTemperature
# from misc import CachedEnsemble
import pandas as pd
import os
import csv
import traceback
import functools
import utils.clip_utils
from hierarchical_selective_classification import *
from hierarchy.ranking import *
from hierarchy.hierarchy import *
from hierarchy.inference_rules import *
# from thresholds import *

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

torch.backends.cudnn.benchmark = True

config_parser = parser = argparse.ArgumentParser(description='Timms model comparison config', add_help=False)
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset', default='/datasets/ImageNet/val')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('-lli', '--low_limit', default=30, type=int,
                    metavar='N',
                    help='at which index of the timm models list to start')
parser.add_argument('-hli', '--high_limit', default=32, type=int,
                    metavar='N',
                    help='at which index of the timm models list to end')
parser.add_argument('-lml', '--load-models-list', default='models_lists/imagenet_models_list.txt', type=str,
                    help='Path to file containing a list of models')
parser.add_argument('-of', '--output-file-path', default='./results/models_comparison.csv', type=str,
                    help='Path to output file')
parser.add_argument('-sl', '--save-y-scores', default=False, action='store_true',
                    help='save model y_scores to file, overwriting existing files')
parser.add_argument('-ll', '--load_y_scores', default=True, action='store_true',
                    help='load model y_scores from file')
parser.add_argument('-de', '--dec', default=2, type=int,
                    help='decimal point precision for hier rc_curve')
parser.add_argument('-st', '--stride', default=1, type=int,
                    help='stride for models list')
parser.add_argument('-ex', '--existing-results', default=False, action='store_true',
                    help='skip models with no existing results')
parser.add_argument('-dc', '--dump-csv', default=False, action='store_true',
                    help='Dump existing results to csv')
parser.add_argument('-tv', '--torchvision', default=False, action='store_true',
                    help='Evaluate torchvision models')
parser.add_argument('-c', '--clip', default=False, action='store_true',
                    help='Evaluate CLIP models')
parser.add_argument('-ti', '--inat', default=False, action='store_true',
                    help='Load and evaluate inat models')
parser.add_argument('-rv', '--reverse', default=False, action='store_true',
                    help='Go over the models in reverse')
parser.add_argument('-lb', '--labels', default='original', type=str,
                    help='use original or gpt labels for hclip')
parser.add_argument('-k', '--confidence-score', default='softmax', type=str,
                    help='Max logit or softmax')

def count_parameters(model):
    '''
    Provides the model's 'size' in parameteres
    :param model: a neural network
    :return: amount of trainable parameters of the model
    '''
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    '''
    Provides the model's 'size' in trainable parameteres
    :param model: a neural network
    :return: amount of trainable parameters of the model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def softmax_forward_pass(x, model):
    y_scores = model(x)
    y_scores = torch.softmax(y_scores, dim=1)
    return y_scores

def hCLIP_forward_pass(x, model):
    y_scores = model(x)
    return y_scores

def clip_forward_pass(x, model, zeroshot_weights):
    image_features = model.encode_image(x)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    logits = 100. * image_features @ zeroshot_weights
    y_scores = torch.softmax(logits, dim=1)
    return y_scores


def clip_finetuned_forward_pass(x, model):
    y_scores = model(x)  # Logistic regression returns probabilities
    return y_scores


def softmax_quantized_forward_pass(x, model):
    # There is an issue with quantized models running on cuda. Run it on cpu for now.
    x = x.cpu()
    model = model.cpu()
    y_scores = model(x)
    y_scores = torch.softmax(y_scores, dim=1)
    return y_scores.cuda()  # Return y on cuda to match wrapper code


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def MC_Dropout_Pass(x, model, dropout_iterations=30, classification=True):
    # MC Dropout work
    predictions = torch.empty((0, x.shape[0], 1000), device='cuda')  # 0, n_classes
    for i in range(dropout_iterations):
        model.eval()
        enable_dropout(model)
        output = model(x)
        output = torch.softmax(output, dim=1)
        predictions = torch.vstack((predictions, output.unsqueeze(0)))

    # Calculating mean across multiple MCD forward passes
    mean = torch.mean(predictions, dim=0)  # shape (n_samples, n_classes)??
    label_predictions = mean.max(1)[1]
    output_mean = torch.mean(predictions, dim=0, keepdim=True)
    if not classification:
        assert False
        output_variance = torch.var(predictions, dim=0)
        # prediction_variance = output_variance[:, label_predictions]
        # prediction_variance = output_variance.index_select(dim=1, index=label_predictions)
        # prediction_variance = output_variance[label_predictions.unsqueeze(1)]
        prediction_variance = torch.gather(output_variance, -1, label_predictions.unsqueeze(-1)).squeeze(1)
        return [-prediction_variance, label_predictions]
        return output_mean, output_variance  # Return predictions & uncertainty

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -torch.sum(mean * torch.log(mean + epsilon), dim=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - torch.mean(torch.sum(-predictions * torch.log(predictions + epsilon), dim=-1), dim=0)  # shape (n_samples,)

    return [-entropy, label_predictions]  # Change it if using another metric other than entropy
    # return [-mutual_info, label_predictions]  # Change it if using another metric other than entropy
    return output_mean, entropy, mutual_info  # Return predictions & uncertainty


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 21k to 1k translation code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import imagenet1K_synsets
labels_1k = imagenet1K_synsets.imagenet_21_dictionary
with open("/home/shanigoren/hsc/resources/ImageNet21K_labels") as f:
    wordnet_ids_21k = f.read().splitlines()
numeric_label_by_wordnet_id_21k = {x: i for i, x in enumerate(wordnet_ids_21k)}
wordnet_id_by_numeric_label_21k = {i: x for i, x in enumerate(wordnet_ids_21k)}
numeric_label_by_wordnet_id_1k = {x: i for i, x in enumerate(imagenet1K_synsets.wordnet_ids_1k)}
wordnet_id_by_numeric_label_1k = {i: x for i, x in enumerate(imagenet1K_synsets.wordnet_ids_1k)}
wordnet_ids_1k = imagenet1K_synsets.wordnet_ids_1k
if 'n04399382' in wordnet_ids_1k:
    wordnet_ids_1k.remove('n04399382')
# indices_by_order = torch.tensor([numeric_label_by_wordnet_id_21k[wordnet_id] for wordnet_id in wordnet_ids_1k]).cuda()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def softmax_21k_forward_pass(x, model):
    y_scores = model(x)
    y_scores = y_scores[:, indices_by_order]
    # y_scores = torch.index_select(y_scores, dim=1, index=indices_by_order)
    y_scores = torch.softmax(y_scores, dim=1)
    return y_scores

def get_certainties_and_correctness(y_scores, y, total_correct, total_samples, samples_certainties, forward_pass_returns_confidence=False):
    if forward_pass_returns_confidence:
            y_pred = y_scores  # [0] is certainty, [1] is class
    else:
        y_pred = torch.max(y_scores, dim=1)
    certainties = y_pred[0]
    correct = y_pred[1] == y

    total_correct += correct.sum().item()
    total_samples += y.shape[0]
    accuracy = (total_correct / total_samples) * 100

    samples_info = torch.stack((certainties.cpu(), correct.cpu()))  # Each sample's certainty next to its correctness
    samples_certainties = torch.vstack((samples_certainties, samples_info.transpose(0, 1)))
    return accuracy, samples_certainties, total_correct, total_samples


def load_inference(y_true_file_name, y_scores_file_name, total_correct, total_samples, samples_certainties, forward_pass_returns_confidence=False):
    y = torch.load(y_true_file_name)
    y_scores = torch.load(y_scores_file_name)
    all_y_true = y
    all_y_scores = y_scores
    accuracy, samples_certainties, total_correct, total_samples = get_certainties_and_correctness(y_scores, y, total_correct, total_samples, samples_certainties, forward_pass_returns_confidence)
    return  accuracy, samples_certainties, total_correct, total_samples, all_y_true, all_y_scores

def handle_exception(logger, model_name, exception, ts=False):
    pass
    ts = '_TS' if ts else ''
    print(f'Extracting model results failed for model {model_name}' + ts + f'. Exception: {exception}')
    traceback.print_exc()
    # all_timm_models = timm.models.list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
    # index = all_timm_models.index(model_name)
    # logger.log({'index':index , 'model_name':model_name+ts, 'exception':exception, 'stack_trace':traceback.format_exc()})

# TODO: Add description when done coding
def extract_model_results(model, dataloader, model_name, forward_pass_function=softmax_forward_pass, temp_scaling=False, extract_flops=True, forward_pass_returns_confidence=False, return_samples_certainties=False, save_y_scores_to_file=True, load_y_scores_from_file=False, new_temperature_overwrite=False):
    if args.confidence_score == 'max_logit':
        model_name = f'{model_name}_logits'
    tmp_scale = 'temperature scaled ' if temp_scaling else ''
    pbar_name=f'Extracting {tmp_scale}data for {model_name}'
    num_batches = len(dataloader.batch_sampler)
    total_correct = 0
    total_samples = 0
    # samples_certainties holds a tensor of size (N, 2) of N samples, for each its certainty and whether it was a correct prediction.
    # Position 0 is the confidences and 1 is the correctness
    samples_certainties = torch.empty((0, 2))
    timer_start = timer()
    if load_y_scores_from_file or save_y_scores_to_file:
        save_model_name = model_name.replace('/', '-')
        if temp_scaling:
            y_scores_file_name = '/home/shanigoren/hsc/resources/models_y_scores/TS/' + save_model_name + '_TS.pt'
            y_true_file_name = '/home/shanigoren/hsc/resources/models_ground_truth/TS/' + save_model_name  + '_TS.pt'
        else:
            y_scores_file_name = '/home/shanigoren/hsc/resources/models_y_scores/' + save_model_name + '.pt'
            y_true_file_name = '/home/shanigoren/hsc/resources/models_ground_truth/' + save_model_name  + '.pt'
    all_y_scores = torch.empty((0, 1000))
    if args.inat:
        all_y_scores = torch.empty((0, 10000))
    all_y_true = torch.empty((0,))

    with torch.no_grad():
        with tqdm(desc=pbar_name, total=num_batches, file=sys.stdout) as pbar:
            load_successful = False
            if load_y_scores_from_file and not new_temperature_overwrite:
                try:
                    accuracy, samples_certainties, total_correct, total_samples, all_y_true, all_y_scores = load_inference(y_true_file_name, y_scores_file_name, total_correct, total_samples, samples_certainties, forward_pass_returns_confidence)
                    load_successful = True
                    pbar.set_description(f'{pbar_name}. accuracy:{accuracy:.3f}% (Elapsed time:{timer() - timer_start:.3f} sec)')
                    pbar.update()
                except Exception as e:
                    print(e)
                    print('Failed to load y_scores from file, extracting them now')

            if not load_y_scores_from_file or not load_successful:
                dl_iter = iter(dataloader)
                for batch_idx in range(num_batches):
                    x, y = next(dl_iter)
                    x = x.cuda()
                    y = y.cuda()
                    if forward_pass_function:  # For special kinds of uncertainty estimators, such as mc dropout or ensembles
                        y_scores = forward_pass_function(x, model)
                    else:
                        y_scores = model.forward(x)

                    accuracy, samples_certainties, total_correct, total_samples = get_certainties_and_correctness(y_scores, y, total_correct, total_samples, samples_certainties, forward_pass_returns_confidence)

                    pbar.set_description(f'{pbar_name}. accuracy:{accuracy:.3f}% (Elapsed time:{timer() - timer_start:.3f} sec)')
                    pbar.update()

        if save_y_scores_to_file:
            torch.save(all_y_scores, y_scores_file_name)
            torch.save(all_y_true, y_true_file_name)

        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
        certainties_are_probabilities = not forward_pass_returns_confidence  # If returns confidence separately,
        # assume it is not probabilistic. Change this logic if necessary.
        if extract_flops:
            results = metrics_calculations(samples_certainties, model, x, certainties_are_probabilities=certainties_are_probabilities)
        else:
            results = metrics_calculations(samples_certainties, model, certainties_are_probabilities=certainties_are_probabilities)

        # if 'inat' not in model_name:
        try:
            all_y_scores = all_y_scores.cuda()
            all_y_true = all_y_true.cuda()
        except:
            pass
            
        hsc_results = hierarchical_selective_classification(model_name, all_y_scores, all_y_true, temp_scaling, dec=args.dec, overwrite=new_temperature_overwrite)
        results = {**results, **hsc_results}

        if return_samples_certainties:
            return results, samples_certainties
        else:
            return results

        # TODO: Calculate gamma, ECE etc with samples_certainties

def metrics_calculations(samples_certainties, model=None, input_sample=None, certainties_are_probabilities=False, num_bins=15):
    # TODO: Flops are optional, write documentation
    results = {}
    results['Accuracy'] = (samples_certainties[:,1].sum() / samples_certainties.shape[0]).item() * 100
    discrimination = gamma_correlation(samples_certainties)
    results['AUROC'] = discrimination['AUROC']
    results['Gamma Correlation'] = discrimination['gamma']
    results['AURC'] = AURC_calc(samples_certainties)
    results['EAURC'] = EAURC_calc(results['AURC'], results['Accuracy'])
    results['Parameters #'] = count_parameters(model)
    results['Parameters # Trainable'] = count_trainable_parameters(model)

    results['Coverage_for_Accuracy_95'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.95)
    results['Coverage_for_Accuracy_96'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.96)
    results['Coverage_for_Accuracy_97'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.97)
    results['Coverage_for_Accuracy_98'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.98)
    results['Coverage_for_Accuracy_99'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.99)

    results['Coverage_for_Accuracy_95_nonstrict'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.95, start_index=200)
    results['Coverage_for_Accuracy_96_nonstrict'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.96, start_index=200)
    results['Coverage_for_Accuracy_97_nonstrict'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.97, start_index=200)
    results['Coverage_for_Accuracy_98_nonstrict'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.98, start_index=200)
    results['Coverage_for_Accuracy_99_nonstrict'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.99, start_index=200)

    if (model is not None) and (input_sample is not None):
        pass
        # results['Flops'] = FlopCountAnalysis(model.cpu(), input_sample.cpu()).total()
    if certainties_are_probabilities:
        # We can't calculate calibration metrics on non-probabilistic certainties
        ece, mce = ECE_calc(samples_certainties, num_bins=num_bins)
        results[f'ECE_{num_bins}'] = ece
        results[f'MCE_{num_bins}'] = mce

        ece, mce = ECE_calc(samples_certainties, num_bins=10)
        results['ECE_10'] = ece
        results['MCE_10'] = mce

        ece, mce = ECE_calc(samples_certainties, num_bins=30)
        results['ECE_30'] = ece
        results['MCE_30'] = mce

        ece, mce = ECE_calc(samples_certainties, num_bins=50)
        results['ECE_50'] = ece
        results['MCE_50'] = mce
        adaptive_ece, adaptive_mce = ECE_calc(samples_certainties, num_bins=num_bins, bin_boundaries_scheme=calc_adaptive_bin_size)
        results['A-ECE'] = adaptive_ece
        results['A-MCE'] = adaptive_mce

        results['Confidence Mean'] = confidence_mean(samples_certainties)
        results['Confidence Median'] = confidence_median(samples_certainties)
        results['Confidence GINI'] = gini(samples_certainties)
        results['Confidence Variance'] = confidence_variance(samples_certainties)
    return results

def calc_and_log_hRanking(all_y_true, all_y_scores):
    results = {}
    hierarchy = get_hierarchy(rebuild_hier=False, load_hier=True)
    cal_indices, val_indices = train_test_split(np.arange(len(all_y_true)), train_size=5000, stratify=all_y_true.cpu())
    y_scores_cal = all_y_scores[cal_indices].cuda()
    y_scores_val = all_y_scores[val_indices].cuda()
    y_true_cal = all_y_true[cal_indices].long().cuda()
    y_true_val = all_y_true[val_indices].long().cuda()
    preds_leaf_cal = y_scores_cal.max(dim=1)[1]
    preds_leaf_val = y_scores_val.max(dim=1)[1]
    all_nodes_probs_cal = hierarchy.all_nodes_probs(y_scores_cal)
    all_nodes_probs_val = hierarchy.all_nodes_probs(y_scores_val)
    hier_loss = HierarchyLoss(hierarchy)
    # calculate flat hAUROC
    confidence_flat, preds_flat = y_scores_val.max(dim=1)
    risk_flat = hier_loss.loss_per_sample(preds_flat, y_true_val)
    results['hAUROC_flat'] = hAUROC(confidence_flat, risk_flat)
    correctness = hierarchy.correctness(preds_flat, y_true_val).cpu()
    accuracy = correctness.sum().item() / len(correctness)
    results['hAccuracy_flat'] = accuracy
    alpha = 0.1
    results['alpha'] = alpha

    for inf_rule_name in inference_rules.inf_rules_names:
        inf_rule = get_inference_rule(inf_rule_name, hierarchy)
        # get_optimal_thresholds
        correct_thetas = inf_rule.get_thresholds(all_nodes_probs_cal, preds_leaf_cal, y_true_cal)
        opt_theta = inf_rule.compute_quantile_threshold(correct_thetas, alpha=alpha)
        results[f'opt_theta_{inf_rule_name.lower()}'] = opt_theta
        # get hier prediction for val set:
        preds_hier = inf_rule.predict(all_nodes_probs_val, preds_leaf_val, opt_theta)
        # calculate hAUROC
        # risk_hier = hier_loss.loss_per_sample(preds_hier, y_true_val)
        # confidence = all_nodes_probs_val[preds_hier, torch.arange(len(preds_hier))]
        # results[f'hAUROC_{inf_rule_name.lower()}{ts}'] = hAUROC(confidence, risk_hier)
        correctness = hierarchy.correctness(preds_hier, y_true_val).cpu()
        accuracy = correctness.sum().item() / len(correctness)
        results[f'hAccuracy_{inf_rule_name.lower()}'] = accuracy
        correct_thetas_val = inf_rule.get_thresholds(all_nodes_probs_val, preds_leaf_val, y_true_val)
        results[f'marginal_coverage_{inf_rule_name.lower()}'] = torch.sum(correct_thetas_val <= opt_theta).item()/len(correct_thetas_val)
    return results

def hierarchical_selective_classification(model_name, all_y_scores, all_y_true, temp_scaling, dec=4, overwrite=False, skip_no_rc=False):
    print(f'Calculating hierarchical selective classification metrics, Temp scaling: {temp_scaling}')
    timer_start = timer()
    results = {}
    existing_rc_curve = None
    if temp_scaling:
        rc_curve_file_name = f'results/rc_curves/dec_{dec}/TS/{model_name.replace("/", "-")}_TS.csv'
    else:
        rc_curve_file_name = 'results/rc_curves/' + f'dec_{dec}/{model_name.replace("/", "-")}.csv'
    if not overwrite and os.path.isfile(rc_curve_file_name):
        try:
            existing_rc_curve = pd.read_csv(rc_curve_file_name)
            len_thetas = len(existing_rc_curve['theta'].unique())
            print(f'Found existing rc_curve file, with {len_thetas} thetas.')
        except:
            existing_rc_curve = None

    if existing_rc_curve is None:
        if skip_no_rc:
            raise Exception('No RC Curve')
        try:
            h_aurc_results = compute_hier_rc_curve(all_y_scores, all_y_true, model_name, save_to_file=True, temp_scaling=temp_scaling, dec=dec, existing_rc_curve=existing_rc_curve)
            # hranking_results = calc_and_log_hRanking(all_y_true, all_y_scores)
        except:
            h_aurc_results = {}
        results = {**results, **h_aurc_results}
        
    print('Finished hierarchical selective classification metrics, took: {:.3f} sec'.format(timer() - timer_start))
    return results

# Special models setup:
def extract_image_size_from_transform(transform):
    for trans in transform.transforms:
        if hasattr(trans, 'size'):
            if isinstance(getattr(trans, 'size'), tuple):
                return getattr(trans, 'size')


# TODO: Important! this assumes the model returns normal logits for ImageNet1k etc, meaning: no MC-Dropout,
#  no trained for ImageNet21k etc...
def extract_temperature_scaled_metrics(model, transform, valid_size=5000, model_name=None, mc_dropout=False, existing_results=None, new_temperature_overwrite=False):
    if 'inat' in model_name or 'iNat' in model_name:
        valid_size = 50000
    assert valid_size > 0
    dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
    test_indices, valid_indices = sklearn.model_selection.train_test_split(np.arange(len(dataset)),
                                                                           train_size=len(dataset) - valid_size,
                                                                           stratify=dataset.targets)
    valid_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(valid_indices), num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(test_indices), num_workers=8)
    # sanity_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=args.batch_size)

    model = ModelWithTemperature(model)
    if new_temperature_overwrite:
        model.set_temperature(valid_loader)
        temperature = model.temperature.data.item()
    else:
        # check for temperature in existing results
        try:
            temperature = existing_results['Temperature']
            print(f'Found temperature in existing results: {temperature}')
            if temperature < 0.1:
                raise Exception('temperature is too low')
            model.temperature.data = torch.tensor([float(temperature), ]).cuda()
        except:
            print(f'No existing temperature found. Performing temperature scaling')
            model.set_temperature(valid_loader)
            temperature = model.temperature.data.item()
            new_temperature_overwrite = True

    # retry - usually this fixes things
    retry_count = 0
    while temperature < 0.1:
        retry_count += 1
        if retry_count > 5:
            raise Exception(f'Temperature scaling failed for model {model_name} after 5 retries') 
        model = ModelWithTemperature(model)
        print(f'Retrying temperature scaling, attempt {retry_count}')
        model.set_temperature(valid_loader)
        temperature = model.temperature.data.item()
        if temperature - 1.500 < 0.01:
            temperature = -1
    print(f'Done temperature scaling')

    if model_name:
        pbar_name = f'Extracting data for {model_name} after temperature scaling'
    else:
        pbar_name = f'Extracting data for model after temperature scaling'
    if (mc_dropout or 'MCdropout' in model_name):
        forward_pass = MC_Dropout_Pass
    elif (args.confidence_score == 'max_logit'):
        forward_pass = None
    # elif 'CLIP' in model_name:
    #     forward_pass = functools.partial(clip_forward_pass, zeroshot_weights=utils.clip_utils.zeroshot_classifier(utils.clip_utils.imagenet_classes, utils.clip_utils.imagenet_templates, model.model.clip))
    else:
        forward_pass = softmax_forward_pass
    model_results_TS = extract_model_results(model, test_loader, model_name=model_name, temp_scaling=True, extract_flops=False, forward_pass_function=forward_pass, forward_pass_returns_confidence=(mc_dropout or 'MCdropout' in model_name), save_y_scores_to_file=args.save_y_scores, load_y_scores_from_file=args.load_y_scores, new_temperature_overwrite=new_temperature_overwrite)
    # To make sure all temperature scaled metrics have a distinct name, add _TS at its end
    model_results_TS = {f'{key}_TS': value for key, value in model_results_TS.items()}
    model_results_TS['Temperature'] = temperature
    return model_results_TS

# TODO: Add description when done coding
def models_comparison(models_names: list, args, file_name='./results/models_comparison.csv', results_name='model_comparison_results', mc_dropout=False, selective_performance_second_chance=True):
    failed_models = []
    logger = log_utils.Logger(file_name=file_name, headers=log_utils.headers, overwrite=False)
    fail_logger = log_utils.Logger(file_name=file_name.rsplit('/', 1)[0] + '/failed_models.csv', headers=['index', 'model_name', 'exception', 'stack_trace'], overwrite=False)
    results = {}
    skip_no_rc = False
    for i, model_name in enumerate(models_names):
        print('**********************************************************************')
        print(f'Starting evaluation for model: {model_name}, {args.low_limit + i}/{args.high_limit}')
        is_ensemble = False
        if mc_dropout:
            saved_model_name = f'{model_name}_MC_dropout'
            saved_results_name = f'{results_name}_MC_dropout'
        elif isinstance(model_name, dict):  # Then it's an ensemble
            saved_model_name = model_name['model_name']
            saved_results_name = results_name
            is_ensemble = True
        elif args.confidence_score == 'max_logit':
            saved_model_name = f'{model_name}_logits'
            saved_results_name = f'{results_name}'
        else:
            saved_model_name = model_name
            saved_results_name = results_name
        # TODO: Add a skip for all models tested on 21k (22k or whatever)
        existing_results = data_utils.load_pickle(f'./results/ID/{saved_model_name}_results.pkl')
        if args.existing_results and existing_results is None:
            print('no existing results, moving on to the next.')
            continue

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if existing_results is not None:  # We already tested this architecture
            if not is_ensemble:                
                if 'pytorch' in existing_results['Architecture']:
                    existing_results['Architecture'] = existing_results['Architecture'].replace('pytorch',
                                                                                                'torchvision')
                if 'input size' not in existing_results.keys():  # If this was missing
                    model, transform = create_model_and_transforms(args, model_name)
                    input_size = extract_image_size_from_transform(transform)
                    existing_results['input size'] = input_size[0]
                    data_utils.save_pickle(f'./results/ID/{saved_model_name}_results.pkl', existing_results)
                
                if 'Temperature_TS' in existing_results.keys() and 'Temperature' not in existing_results.keys():
                    existing_results['Temperature'] = existing_results['Temperature_TS']
                    existing_results.pop('Temperature_TS')

                if 'Temperature' not in existing_results.keys() and not args.dump_csv:  # If hasn't extracted temperature scaling results yet
                    model, transform = create_model_and_transforms(args, model_name)
                    TS_success = False
                    retry_count = 0
                    while TS_success is False:
                        retry_count += 1
                        print(f'Temperature scaling attempt {retry_count}')
                        if retry_count > 3:
                            raise Exception(f'Temperature scaling failed for model {model_name} after 3 retries') 
                        try:
                            temperature_scaled_model_results = extract_temperature_scaled_metrics(model, transform, model_name=model_name, mc_dropout=mc_dropout, existing_results=existing_results, new_temperature_overwrite=(retry_count > 1))
                            TS_success = True
                        except Exception as e:
                            print(e)
                            continue
                    existing_results = {**existing_results, **temperature_scaled_model_results}

                if 'EAURC' not in existing_results.keys():
                    existing_results['EAURC'] = EAURC_calc(existing_results['AURC'], existing_results['Accuracy'])
                    existing_results['EAURC_TS'] = EAURC_calc(existing_results['AURC_TS'],
                                                              existing_results['Accuracy_TS'])
                if ('Coverage_for_Accuracy_95_nonstrict' not in existing_results.keys()) and not (mc_dropout or 'MCdropout' in model_name):
                # if 'Coverage_for_Accuracy_95_nonstrict' not in existing_results.keys():
                #     if model_name in mydict.keys():
                #         model_results = mydict[model_name]
                #         suffix = '_nonstrict'
                    # elif existing_results['Coverage_for_Accuracy_99'] <= 0.001 and model_name != 'resnetv2_152x4_bitm':
                    if existing_results['Coverage_for_Accuracy_99'] <= 0.001 and model_name != 'resnetv2_152x4_bitm':
                        model, transform = create_model_and_transforms(args, model_name)
                        dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
                        dataloader = DataLoader(dataset, pin_memory=True, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
                        model_results = extract_model_results(model, dataloader,
                                                           model_name=model_name, temp_scaling=False,
                                                           forward_pass_function=softmax_forward_pass,
                                                           extract_flops=False,
                                                           forward_pass_returns_confidence=False,
                                                           save_y_scores_to_file=args.save_y_scores, load_y_scores_from_file=args.load_y_scores)
                        suffix = '_nonstrict'
                    else:
                        model_results = existing_results
                        suffix = ''
                    existing_results['Coverage_for_Accuracy_95_nonstrict'] = model_results[f'Coverage_for_Accuracy_95{suffix}']
                    existing_results['Coverage_for_Accuracy_96_nonstrict'] = model_results[f'Coverage_for_Accuracy_96{suffix}']
                    existing_results['Coverage_for_Accuracy_97_nonstrict'] = model_results[f'Coverage_for_Accuracy_97{suffix}']
                    existing_results['Coverage_for_Accuracy_98_nonstrict'] = model_results[f'Coverage_for_Accuracy_98{suffix}']
                    existing_results['Coverage_for_Accuracy_99_nonstrict'] = model_results[f'Coverage_for_Accuracy_99{suffix}']
            logger.log(existing_results)
            data_utils.save_pickle(f'./results/ID/{saved_model_name}_results.pkl', existing_results)
            results[existing_results['Architecture']] = existing_results
            continue
        ################# done existing results #####################

        extract_flops = False
        if is_ensemble:  # Then it's an ensemble
            model = CachedEnsemble.NNCachedEnsemble(model_name['nets'])
            model_name = model_name['model_name']
        else:
            try:
                model, transform = create_model_and_transforms(args, model_name)
            except Exception as e:
                print(f'Failed to create model {model_name}. Exception: {e}')
                continue
            dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
            dataloader = DataLoader(dataset, pin_memory=True, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)
        if (('in21k' in model_name) or ('in22k' in model_name)) and ('in1k' not in model_name):
        # if ('in21k' in model_name) or ('in22k' in model_name):
            if 'miil' in model_name:
                # TODO: Solve index translation from miil models
                continue
            forward_pass = softmax_21k_forward_pass
            # Remove the teddy bear class missing in ImageNet 21k to be fair to images not knowing it
            indices = [i for i in range(len(dataset)) if dataset.imgs[i][1] != dataset.class_to_idx['n04399382']]
            dataset = Subset(dataset, indices)
        elif 'quantized' in model_name:
            forward_pass = softmax_quantized_forward_pass
            extract_flops = False
        elif mc_dropout or 'MCdropout' in model_name:
            extract_flops = False
            forward_pass = MC_Dropout_Pass
        elif 'CLIP_finetuned' in model_name:
            forward_pass = clip_finetuned_forward_pass
        elif 'hCLIP' in model_name:
            forward_pass = hCLIP_forward_pass
        # elif 'CLIP' in model_name:
        #     forward_pass = functools.partial(clip_forward_pass, zeroshot_weights=utils.clip_utils.zeroshot_classifier(utils.clip_utils.imagenet_classes, utils.clip_utils.imagenet_templates, model))
        elif (args.confidence_score == 'max_logit'):
            forward_pass = None
        else:
            forward_pass = softmax_forward_pass

        model_results = None
        if is_ensemble:
            model_results = model.predict_as_ensemble()
        else:
            try:
                model_results = extract_model_results(model, dataloader, model_name=model_name, temp_scaling=False, forward_pass_function=forward_pass, extract_flops=extract_flops, forward_pass_returns_confidence=(mc_dropout or 'MCdropout' in model_name), save_y_scores_to_file=args.save_y_scores, load_y_scores_from_file=args.load_y_scores)
            except Exception as e:
                handle_exception(fail_logger, model_name, e)

        try:
            temperature_scaled_model_results = extract_temperature_scaled_metrics(model, transform, model_name=model_name, mc_dropout=mc_dropout)
            model_results = {**model_results, **temperature_scaled_model_results}
        except Exception as e:
            handle_exception(fail_logger, model_name, e, ts=True)

        if model_results is not None:
            model_results['Architecture'] = model_name
            if not is_ensemble:
                input_size = extract_image_size_from_transform(transform)
                model_results['input size'] = input_size[0]
            logger.log(model_results)
            data_utils.save_pickle(f'./results/ID/{saved_model_name}_results.pkl', model_results)
            results[model_name] = model_results
    # data_utils.save_pickle(f'./results/ID/model_comparison_results.pkl', results)
    data_utils.save_pickle(f'./results/ID/{saved_results_name}.pkl', results)


if __name__ == '__main__':
    args = parser.parse_args()
    tst = []
    if args.load_models_list:
        tst = []
        with open(args.load_models_list, 'r') as f:
            for line in f:
                tst.append(line.strip())
    if args.clip:
        tst += ['CLIP_'+m for m in clip.available_models()]
    if args.inat:
        tst = []
        with open('resources/inat_models_list.txt', 'r') as f:
            for line in f:
                tst.append(line.strip())

    if args.stride > 1:
        tst = tst[::args.stride]
    if args.low_limit is not None and args.high_limit is not None:
        tst = tst[args.low_limit:args.high_limit]
    if args.reverse:
        tst = tst[::-1]
    torch.manual_seed(1)
    np.random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    file_name = './results/models_comparison.csv'

    if args.low_limit is None and args.high_limit is None:
        args.low_limit = 0
        args.high_limit = len(tst)
    models_comparison(tst, args, file_name=file_name)

    # ~~~~~~~~~~~~~~~~~~~~~~~~ Code to get only the models I've manually chosen for ID ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    csv_file = './results/ID/models_to_check.csv'
    try:
        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile)
            rows = []
            archs = [row[0] for row in reader]
            archs.pop(0)
        models_names = archs
        models_names = [model_name for model_name in models_names if 'tf_efficientnet_lite4' not in model_name]
    except:
        pass
    # TODO: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

