import torch
import numpy as np
import networkx as nx
from torch.nn import functional as F

inf_rules_names = ['Selective', 'Climbing', 'Jumping', 'MaxCoverage']

class InferenceRuleBase:
    def __init__(self, hierarchy) -> None:
        self.hierarchy = hierarchy

    def predict(self):
        raise NotImplementedError

    def get_thresholds(self, all_nodes_probs, preds_leaf, labels):
        raise NotImplementedError

    def compute_quantile_threshold(self, all_thetas, alpha):
        theta_hat = np.quantile(all_thetas.cpu(), 1-alpha)
        return theta_hat
        
    def compute_threshold_interval(self, all_thetas, alpha):
        thetas_sorted = np.sort(np.array(all_thetas.cpu()))
        n = len(all_thetas)
        index = min(int(np.ceil((n+1)*(1-alpha))), n-1)

        
class SelectiveInferenceRule(InferenceRuleBase):
    """If leaf confidence >= theta, predict leaf, else predict root"""
    def __init__(self, hierarchy) -> None:
        super().__init__(hierarchy)

    def predict(self, all_nodes_probs, preds_leaf, theta):
        """theta - confidence threshold
        preds - flat predictions, shape: [n_samples]
        probs_leaf - probabilities of leaf predictions, shape: [n_samples, n_leaves]"""
        probs_leaf = torch.transpose(all_nodes_probs,0,1)[:,:self.hierarchy.num_leaves]
        preds_hier = preds_leaf.clone()
        # get unconfident leaf predictions
        confidence = probs_leaf.max(dim=1)[0]
        unconfident_samples = confidence < theta
        preds_hier[unconfident_samples] = self.hierarchy.root_index
        confidence[unconfident_samples] = 1.0
        return confidence, preds_hier

    def get_thresholds(self, all_nodes_probs, preds_leaf, labels):
        """if leaf is correct, return 0, else return leaf prob + epsilon"""
        epsilon = 1e-6
        incorrect = preds_leaf != labels
        thresholds = torch.zeros_like(preds_leaf).float()
        probs_leaf = all_nodes_probs.transpose(0,1)[:,:self.hierarchy.num_leaves]
        thresholds[incorrect] = probs_leaf[incorrect, preds_leaf[incorrect]] + epsilon
        return thresholds

class ClimbingInferenceRule(InferenceRuleBase):
    """Start from most confident leaf, and climb up the path to the root until confident >= theta"""
    def __init__(self, hierarchy) -> None:
        super().__init__(hierarchy)

    def predict(self, all_nodes_probs, preds_leaf, theta):
        """theta - confidence threshold
        preds - flat predictions, shape: [n_samples]
        probs_leaf - probabilities of leaf predictions, shape: [n_samples, n_leaves]"""
        anc_mask = self.hierarchy.anc_matrix[:,preds_leaf]
        anc_probs = all_nodes_probs * anc_mask
        # confidence score is max logits
        if (all_nodes_probs>1.0).sum() == 0:
            anc_probs[self.hierarchy.root_index,:] = 1.0 # manually set the probability of the root to 1.0
        anc_probs[anc_probs < theta] = np.inf # we take the min prob later, don't want any prob below theta
        anc_probs[anc_probs == 0.0] = np.inf # don't want any prob of 0
        return anc_probs.min(dim=0)      

    def get_tight_thresholds(self, all_nodes_probs, preds_leaf, labels, epsilon=1e-6):
        # first correct = lca(pred,label) prob
        lcas = self.hierarchy.lcas[preds_leaf.cpu(), labels.cpu()]
        # first correct = the probability of the lca - lca is the first ancestor that is correct
        first_correct = all_nodes_probs[lcas, torch.arange(len(lcas))]
        # last incorrect = the last ancestor that is incorrect, meaning the child of the lca which is an ancestor of pred
        preds_ancestors = self.hierarchy.anc_matrix[:,preds_leaf]
        masked_probs = all_nodes_probs * preds_ancestors
        masked_probs[masked_probs >= first_correct] = 0
        last_incorrect = masked_probs.max(dim=0)[0]
        thresholds = last_incorrect + epsilon * (first_correct - last_incorrect)
        thresholds[preds_leaf == labels] = 0.0
        return thresholds

class JumpingInferenceRule(InferenceRuleBase):
    """Start from most confident leaf, and jump up at each layer to the most confident node, until confident >= theta"""
    def __init__(self, hierarchy) -> None:
        super().__init__(hierarchy)
        # create height masks for each layer
        # height mask: [n_classes] has 1 in nodes of height h, else 0
        self.height_masks = {}
        for h in range(self.hierarchy.root_height+1):
            indices = torch.tensor([node for node in self.hierarchy.tree.nodes if h in self.hierarchy.tree.nodes[node]['height']])
            self.height_masks[h] = torch.zeros((self.hierarchy.num_nodes,1))
            self.height_masks[h] = self.height_masks[h].index_fill_(0, indices, 1)
    
    def predict(self, all_nodes_probs, preds_leaf, theta):
        """theta - confidence threshold
        preds - flat predictions, shape: [n_samples]
        probs_leaf - probabilities of leaf predictions, shape: [n_samples, n_leaves]"""
        probs_leaf = all_nodes_probs.transpose(0,1)[:,:self.hierarchy.num_leaves] 
        hier_confidence = probs_leaf.max(dim=1)[0]
        if torch.all(hier_confidence >= theta):
            return hier_confidence, preds_leaf
        h = 1
        height_mask = torch.zeros_like(all_nodes_probs)
        height_mask[:,:] = self.height_masks[0].to(all_nodes_probs.device)
        while not torch.all(hier_confidence >= theta) and h < len(self.height_masks):
            # get indices of unconfident samples
            unconfident_samples = (hier_confidence < theta)
            # create a height mask according to the confidence of each sample
            height_mask[:,unconfident_samples] = self.height_masks[h].to(all_nodes_probs.device)
            masked_probs = all_nodes_probs * height_mask
            hier_confidence, hier_preds = masked_probs.max(dim=0)
            h += 1
        return hier_confidence, hier_preds

    def get_thresholds(self, all_nodes_probs, preds_leaf, labels):
        labels_onehot = torch.transpose(F.one_hot(labels, num_classes=self.hierarchy.num_nodes),0,1).float()
        labels_ancestors = self.hierarchy.anc_matrix @ labels_onehot
        hier_preds = preds_leaf

        h = 0
        height_mask = torch.zeros_like(all_nodes_probs)
        height_mask[:,:] = self.height_masks[0]
        thresholds = torch.ones_like(preds_leaf).float()
        while not self.hierarchy.correctness(hier_preds, labels).sum() == len(labels):
            # get indices of wrong samples
            preds_onehot = torch.transpose(F.one_hot(hier_preds, num_classes=self.hierarchy.num_nodes),0,1)
            labels_onehot = torch.transpose(F.one_hot(labels, num_classes=self.hierarchy.num_nodes),0,1)
            labels_ancestors = self.hierarchy.anc_matrix @ labels_onehot.float()
            correct = torch.logical_and(preds_onehot, labels_ancestors)
            wrong_samples = correct.sum(dim=0) == 0
            # create a height mask according to the confidence of each sample
            height_mask[:,wrong_samples] = self.height_masks[h].to(all_nodes_probs.device)
            masked_probs = all_nodes_probs * height_mask
            thresholds, hier_preds = masked_probs.max(dim=0)
            h += 1
        thresholds[preds_leaf == labels] = 0.0
        return thresholds

class MaxCoverageInferenceRule(InferenceRuleBase):
    """Out of all samples that have confidence >= theta, choose the one with the highest coverage"""
    def __init__(self, hierarchy) -> None:
        super().__init__(hierarchy)  
    
    def predict(self, all_nodes_probs, preds_leaf, theta):
        mask = all_nodes_probs >= theta
        coverage_mat = self.hierarchy.coverage_vec.expand(-1, all_nodes_probs.shape[1])
        masked_coverage = mask * coverage_mat
        max_coverage_value = masked_coverage.max(dim=0)[0].reshape(1,-1)
        max_coverage_mask = masked_coverage == max_coverage_value
        masked_probs = max_coverage_mask * all_nodes_probs
        # in case there are multiple samples with the same coverage, choose the one with the highest confidence
        max_confidence_value = masked_probs.max(dim=0)[0].reshape(1,-1)
        max_confidence_mask = masked_probs == max_confidence_value
        masked_probs = max_confidence_mask * masked_probs
        return masked_probs.max(dim=0)

    def get_tight_thresholds(self, all_nodes_probs, preds_leaf, labels, epsilon=1e-6):
        thresholds = []
        # get list of all label ancestors probabilities:
        for i, label in enumerate(labels):
            # correct prediction, threshold is 0
            if preds_leaf[i] == label:
                thresholds.append(0.0)
                continue
            labels_ancestors = self.hierarchy.get_ancestors(self.hierarchy.nodes[label])
            labels_ancestors_nodes = self.hierarchy.get_ancestors_nodes(self.hierarchy.nodes[label])
            labels_ancestors_probs = all_nodes_probs[labels_ancestors,i]
            found = False
            prev_theta = 0.0
            # set each of the ancestor probs as theta
            for theta in sorted(labels_ancestors_probs):
                # leave all nodes with prob >= theta
                greater_than_theta = all_nodes_probs[:,i] >= theta
                # get node with max coverage
                mask = all_nodes_probs[:,i] >= theta
                masked_coverage = mask.reshape(-1,1) * self.hierarchy.coverage_vec
                max_coverage_value = masked_coverage.max(dim=0)[0]
                max_coverage_mask = masked_coverage == max_coverage_value
                masked_probs = max_coverage_mask * all_nodes_probs[:,i].reshape(-1,1)
                confidence_hier, preds_hier = masked_probs.max(dim=0)
                # check if we got the correct prediction
                if self.hierarchy.correctness(preds_hier, label.reshape(-1)):
                    thresholds.append(prev_theta + epsilon * (theta.item() - prev_theta))
                    found = True
                    break

            if not found:
                thresholds.append(1.0)
            
        thresholds_res = torch.tensor(thresholds)
        return thresholds_res

class MaxLikelihoodInferenceRule(InferenceRuleBase):
    """Start from most confident leaf, and climb up the path to the root until confident >= theta"""
    def __init__(self, hierarchy, climb_all=False) -> None:
        super().__init__(hierarchy)
        self.num_leaves = self.hierarchy.num_leaves
        self.leaf_anc_mask = self.hierarchy.anc_matrix[:,:self.num_leaves]
        self.max_path_len = self.leaf_anc_mask.sum(dim=0).max().item()
        self.path_probs = None
        self.climb_all = climb_all

    def get_path_probs(self, all_nodes_probs):
        if self.path_probs is not None:
            return self.path_probs
        ancestors = torch.transpose(self.hierarchy.anc_matrix, 0,1)
        log_probs = -torch.log(all_nodes_probs)
        nll = ancestors @ log_probs
        self.path_probs = nll
        # normalize by path length
        path_lengths = ancestors.sum(dim=1, keepdim=True)
        self.path_probs = nll * path_lengths     
        
    def predict(self, all_nodes_probs, preds_leaf, theta):
        """theta - confidence threshold
        preds - flat predictions, shape: [n_samples]
        probs_leaf - probabilities of leaf predictions, shape: [n_samples, n_leaves]"""
        self.get_path_probs(all_nodes_probs)
        leaf_path_probs = self.path_probs[:1000,:]
        most_probable_leaf_path = leaf_path_probs.argmin(dim=0)

        anc_mask = self.hierarchy.anc_matrix[:,most_probable_leaf_path]
        anc_probs = all_nodes_probs * anc_mask
        anc_probs[self.hierarchy.root_index,:] = 1.0 # manually set the probability of the root to 1.0
        anc_probs[anc_probs < theta] = np.inf # we take the min prob later, don't want any prob below theta
        anc_probs[anc_probs == 0.0] = np.inf # don't want any prob of 0 either
        hier_preds = anc_probs.argmin(dim=0)

        # climb only for unconfident samples
        if not self.climb_all:
            confidence_leaf = all_nodes_probs[:1000,:].max(dim=0)[0]
            confident_preds_leaf = confidence_leaf >= theta
            hier_preds[confident_preds_leaf] = preds_leaf[confident_preds_leaf]
        return hier_preds

def get_inference_rule(inference_rule_name, hierarchy):
    if inference_rule_name == 'Selective':
        return SelectiveInferenceRule(hierarchy)
    elif inference_rule_name == 'Climbing':
        return ClimbingInferenceRule(hierarchy)
    elif inference_rule_name == 'Jumping':
        return JumpingInferenceRule(hierarchy)
    elif inference_rule_name == 'MaxCoverage':
        return MaxCoverageInferenceRule(hierarchy)
    elif inference_rule_name == 'MaxLikelihood':
        return MaxLikelihoodInferenceRule(hierarchy)
    else:
        raise ValueError(f'Unknown inference rule name: {inference_rule_name}')