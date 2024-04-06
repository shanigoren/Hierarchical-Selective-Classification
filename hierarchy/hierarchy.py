import math
import pickle
import numpy as np
import networkx as nx
import csv
import torch
from nltk.corpus import wordnet as wn
from torch.nn import functional as F
import os

class Hierarchy():
    """All hierarchy initialization and operations
    To initialize Imagenet1K hierarchy call build_imagenet_tree()
    To initialize another hierarchy with an existing graph call build_tree(graph)
    """
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_name_from_wn_id(name):
        """Gets class name from worndet ID."""
        return wn.synset_from_pos_and_offset('n', int(name[1:])).lemma_names()[0]

    @staticmethod
    def build_parent_child_dict():
        """Parses the hierarchy csv and returns a dictionary that maps child->parent."""
        child_parent_dict = {}
        fname = 'resources/imagenet_fiveai.csv'
        with open(fname) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) != 2:
                    raise ValueError('invalid row', row)
                child_parent_dict[row[1]] = row[0]
        return child_parent_dict

    def build_wn_rels(self, wordnet_isa_path='resources/wordnet_is_a.txt'):
        """Parses the hierarchy csv and returns a dictionary that maps child->parent."""
        child_parent_dict = {}
        for leaf in self.get_leaf_classes():
            child_parent_dict[leaf] = ''
        fname = wordnet_isa_path
        changed = True
        while changed:
            changed = False
            with open(fname, 'r') as f:
                for line in f:
                    parent, child = line.strip().split()
                    if child in child_parent_dict.keys():
                        if child_parent_dict[child] == '':
                            child_parent_dict[child] = parent
                            child_parent_dict[parent] = ''
                            changed = True
        return child_parent_dict

    @staticmethod
    def get_leaf_classes():
        """Parses the imagenet classes and returns a list of leaf classes."""
        with open("resources/LOC_synset_mapping.txt", "r") as f:
            leaf_classes = [line.strip().split(" ")[0] for line in f]
        return leaf_classes
    
    def build_ancestor_matrix(self):
        """Creates ancestor matrix where [i,j] = 1 if i is an ancestor of j"""
        # if len(self.nodes) > 10000:
        #     return self.build_ancestor_matrix_inat()
        n = len(self.adj_matrix)
        anc_matrix = torch.eye(n, dtype=int)
        def dfs(u):
            for v in range(n):
                if self.adj_matrix[u][v] and not anc_matrix[u][v]:
                    anc_matrix[u][v] = 1
                    dfs(v)
                    for w in range(n):
                        if anc_matrix[v][w]:
                            anc_matrix[u][w] = 1
        # Run DFS from each node
        for u in range(n):
            dfs(u)
        return anc_matrix.float().to(self.device)

    def compute_coverage(self):
        """Computes coverage and stores it in the node for each node in the tree."""
        leaf_descendants = self.anc_matrix[self.num_leaves:,:self.num_leaves]
        internal_num_leaves = torch.sum(leaf_descendants, axis=1)
        root_entropy = np.log2(self.num_leaves)
        self.coverage_vec = torch.ones((self.num_nodes,1), device=self.device)
        for node in self.nodes:
            if node[1]['is_leaf']:
                node[1]['coverage'] = 1.0
            else:
                node_entropy = np.log2(internal_num_leaves[node[0]-self.num_leaves].item())
                coverage = 1 - node_entropy/root_entropy
                node[1]['coverage'] = coverage
                self.coverage_vec[node[0]] = coverage

    def compute_heights(self):
        """Computes height and stores it as a list in the node for each node in the tree."""
        """All internal nodes have a single item list."""
        """Leaf nodes may have more than one height, if they are not at the deepest level of the tree."""
        layers = list(nx.bfs_layers(self.tree, self.root_index))
        self.height_vec = torch.zeros((self.num_nodes,1), device=self.device)
        max_depth = len(layers)
        for i, layer in enumerate(reversed(layers)):
            for node in layer:
                if self.tree.nodes[node]['is_leaf']:
                    self.tree.nodes[node]['height'] = list(range(0, i+1))
                else:
                    # Equivalent to adding dummy nodes - the leaf's heights are between its actual height and the height of its parent
                    children_heights = [self.tree.nodes[child]['height'][-1] for child in self.tree.successors(node)]
                    # make sure all children have the same height
                    assert len(set(children_heights)) == 1
                    height = children_heights[0]+1
                    self.tree.nodes[node]['height'] = [height]
                    self.height_vec[node] = height
        self.root_height = self.root_node['height'][0]
        assert self.root_height == max_depth-1

    def build_imagenet_tree(self, wordnet_isa_path='resources/wordnet_is_a.txt'):
        """Builds Imagenet1K hierarchy."""
        self.tree = nx.DiGraph()
        class2idx = {}
        self.idx2class = {}

        # add leaf nodes
        index = 0
        for leaf_class in self.get_leaf_classes():
            class_name = self.get_name_from_wn_id(leaf_class)
            self.tree.add_node(index, wn_id=leaf_class, name=class_name, is_leaf=True)
            class2idx[leaf_class] = index
            self.idx2class[index] = class_name
            index += 1
        self.num_leaves = index

        child_parent_dict = self.build_wn_rels(wordnet_isa_path=wordnet_isa_path)

        # add internal nodes and edges
        for child in child_parent_dict.keys():
            parent = child_parent_dict[child]
            if parent == '':
                continue
            if child not in class2idx.keys():
                class_name = self.get_name_from_wn_id(child)
                self.tree.add_node(index, wn_id=child, name=class_name, is_leaf=False)
                class2idx[child] = index
                self.idx2class[index] = class_name
                index += 1
            if parent not in class2idx.keys():
                class_name = self.get_name_from_wn_id(parent)
                self.tree.add_node(index, wn_id=parent, name=class_name, is_leaf=False)
                class2idx[parent] = index
                self.idx2class[index] = class_name
                index += 1
            self.tree.add_edge(class2idx[parent], class2idx[child])
        
        self.prune_one_child_nodes()
        self.sort_nodes_by_coverage()
        self.compute_coverage()

        self.nodes = [x for x in self.tree.nodes.data()]
        self.leaf_nodes = [x for x in self.tree.nodes.data() if x[1]['is_leaf']]
        self.internal_nodes = [x for x in self.tree.nodes.data() if not x[1]['is_leaf']]

        roots = [node for node, in_degree in self.tree.in_degree() if in_degree == 0]
        self.root_index =  roots[0]
        self.root_node =  self.tree.nodes[self.root_index]

        # adj matrix - [i,j] = 1 if i->j (i is a parent of j)
        self.adj_matrix = torch.tensor(nx.adjacency_matrix(self.tree).todense()).to(self.device)
        # anc matrix - [i,j] = 1 if i->...->j (i is an ancestor of j)
        self.anc_matrix = self.build_ancestor_matrix()
        self.desc_matrix = self.anc_matrix[:,:self.num_leaves]

        self.compute_heights()
        self.compute_lcas()
        self.unit_test()

    def build_inat21_tree(self):
        """Builds iNat21 hierarchy."""
        self.tree = nx.DiGraph()
        self.class2idx = {}
        self.idx2class = {}

        child_parent_dict = {}
        fname = 'resources/inat21.csv'
        with open(fname) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) != 2:
                    raise ValueError('invalid row', row)
                child_parent_dict[row[1]] = row[0]

        classes = [f for f in os.listdir('/2021_valid')]
        classes.sort()
        leaf_classes = [f"{n.split('_')[-2]} {n.split('_')[-1]}" for n in classes]
        self.num_leaves = len(leaf_classes)
        nodes = [s for s in set([k for k in child_parent_dict.keys()] + [v for v in child_parent_dict.values()])]
        slash_nodes = [n.split('/') for n in nodes if '/' in n]
        
        self.anc_matrix = torch.eye(16344, dtype=int)
        internal_index = self.num_leaves
        for c in classes:
            split = c.split('_')
            for parent, child in zip(split[1:-1], split[2:]):
                if [parent,child] in slash_nodes:
                    split[split.index(child)] = f"{parent}/{child}"
            split[-1] = f"{split[-2]} {split[-1]}"
            leaf_name = split[-1]
            leaf_idx = int(split[0])
            self.tree.add_node(leaf_idx, wn_id=None, name=leaf_name, is_leaf=True)
            self.class2idx[leaf_name] = leaf_idx
            self.idx2class[leaf_idx] = leaf_name
            for i, (parent, child) in enumerate(zip(split[1:-1], split[2:])):
                if child not in self.class2idx.keys():
                    self.tree.add_node(internal_index, wn_id=None, name=child, is_leaf=False)
                    self.class2idx[child] = internal_index
                    self.idx2class[internal_index] = child
                    internal_index += 1
                if parent not in self.class2idx.keys():
                    self.tree.add_node(internal_index, wn_id=None, name=parent, is_leaf=False)
                    self.class2idx[parent] = internal_index
                    self.idx2class[internal_index] = parent
                    internal_index += 1
                self.tree.add_edge(self.class2idx[parent], self.class2idx[child])
                
            # update ancestor matrix:
            for i, parent in enumerate(split[1:]):
                ancestor_idx = self.class2idx[parent]
                for j, descendant_name in enumerate(split[i+1:]):
                    descendant_idx = self.class2idx[descendant_name]
                    self.anc_matrix[ancestor_idx, descendant_idx] = 1

        self.nodes = [x for x in self.tree.nodes.data()]
        roots = [node for node, in_degree in self.tree.in_degree() if in_degree == 0]
        self.root_index =  len(self.nodes)
        self.tree.add_node(self.root_index, wn_id=None, name='Life', is_leaf=False)
        # add edges to all roots 
        for root in roots:
            self.tree.add_edge(self.root_index, root)
        self.root_node =  self.tree.nodes[self.root_index]
        self.anc_matrix[self.root_index:] = 1
        self.nodes = [x for x in self.tree.nodes.data()]
        self.prune_one_child_nodes()
        self.desc_matrix = self.anc_matrix[:,:self.num_leaves]

        self.nodes = sorted(self.nodes, key=lambda x: x[0])
        # reinsert all nodes in the tree in the correct order
        new_tree = nx.DiGraph()
        for i, node in enumerate(self.nodes):
            new_tree.add_node(i, **node[1])
        for edge in self.tree.edges:
            new_tree.add_edge(*edge)
        self.tree = new_tree
        self.nodes = [x for x in self.tree.nodes.data()]
        self.num_nodes = len(self.nodes)
        self.compute_coverage()
        self.compute_lcas()

    def build_tree(self, tree):
        """Builds hierarchy, assuming all nodes and edges exist in the hierarchy tree object."""
        self.tree = tree
        self.nodes = [x for x in self.tree.nodes.data()]
        self.leaf_nodes = [x for x in self.tree.nodes.data() if x[1]['is_leaf']]
        self.internal_nodes = [x for x in self.tree.nodes.data() if not x[1]['is_leaf']]
        self.num_leaves = len(self.leaf_nodes)

        self.num_nodes = len(self.nodes)

        roots = [node for node, in_degree in self.tree.in_degree() if in_degree == 0]
        self.root_index =  roots[0]
        self.root_node =  self.tree.nodes[self.root_index]

        # adj matrix - [i,j] = 1 if i->j (i is a parent of j)
        self.adj_matrix = torch.tensor(nx.adjacency_matrix(self.tree).todense())
        # anc matrix - [i,j] = 1 if i->...->j (i is an ancestor of j)
        self.anc_matrix = self.build_ancestor_matrix()

        self.compute_coverage()
        self.compute_heights()
    
    def all_nodes_probs(self, probs_leaf):
        """returns summed probability of all nodes, shape: [n_classes, n_samples] - important: transposed from probs_leaf"""
        probs_t = torch.transpose(probs_leaf,0,1).to(self.device)
        if self.desc_matrix.device.type == 'cpu':
            self.desc_matrix = self.desc_matrix.to(self.device)
        return self.desc_matrix.to(probs_t.dtype) @ probs_t
    
    def sum_nodes_probs(self, all_nodes_probs):
        """returns summed probability of all nodes, when internal nodes have probability > 0. shape: [n_classes, n_samples] - important: transposed from probs_leaf"""
        # # # # leaf score = sum of leaf and only its parent's scores
        # adj = torch.transpose(0.5*self.adj_matrix+torch.eye(self.num_nodes).cuda(),0,1)
        # scores_sum = (all_nodes_probs @ adj)
        # return scores_sum
        
        # # leaf score = sum of ancestors scores
        # probs_t = torch.transpose(all_nodes_probs,0,1).to(self.device)
        # desc = torch.transpose(self.desc_matrix,0,1).to(probs_t.dtype)
        # scores_sum = desc @ probs_t
        # # scores_sum = (desc @ (probs_t * self.coverage_vec)) / desc_sum
        # return scores_sum

        # new: score(node) = sum(descendants)
        log_probs = -torch.log(all_nodes_probs)
        scores_sum = log_probs @ torch.transpose(self.anc_matrix.to(log_probs.dtype),0,1)
        return scores_sum

        # old: score(node) = sum(ancestors)
        # probs_t = torch.transpose(all_nodes_probs,0,1).to(self.device)
        # scores_sum = self.anc_matrix.to(probs_t.dtype) @ probs_t
        # tried some other ways to normalize and none worked.
        # normalized_sum = scores_sum/self.anc_matrix.sum(dim=1).unsqueeze(1)
        return normalized_sum
    
    def get_parent(self, idx):
        """Returns the name of parent of a class given its index."""
        parents = list(self.tree.predecessors(idx))
        if len(parents) == 0:
            return -1
        return list(self.tree.predecessors(idx))[0]
    
    def get_ancestors(self, node_idx):
        """Returns a tensor of all ancestors of a node including itself."""
        return torch.where(self.anc_matrix[:,node_idx] == 1)[0]
    
    def get_ancestors(self, node):
        """Returns a tensor of all ancestors of a node including itself."""
        return torch.where(self.anc_matrix[:,node[0]] == 1)[0]

    def get_ancestors_nodes(self, node):
        """Returns node data of all ancestors of a node including itself."""
        ancestors_indices = torch.where(self.anc_matrix[:,node[0]] == 1)[0]
        return [self.nodes[i] for i in ancestors_indices]

    def compute_lcas(self):
        pairs_lca = nx.all_pairs_lowest_common_ancestor(self.tree)
        self.lcas = torch.zeros(self.num_nodes, self.num_nodes).cpu()
        for pair, lca in pairs_lca:
            self.lcas[pair[0], pair[1]] = lca
            self.lcas[pair[1], pair[0]] = lca
        self.lcas = self.lcas.long()

    def get_lcas(self, labels):
        """Returns the LCAs of a batch of labels."""
        lcas = self.lcas[labels[:,0], labels[:,1]]
        return lcas

    def correctness(self, preds, labels):
        """Calculates the number of correct predictions (either the ground truth leaf or any of its ancestors)."""
        preds_onehot = torch.transpose(F.one_hot(preds, num_classes=self.num_nodes),0,1)
        labels_onehot = torch.transpose(F.one_hot(labels, num_classes=self.num_nodes),0,1).to(self.anc_matrix.dtype)
        if preds_onehot.device.type == 'cpu':
            labels_ancestors = torch.matmul(self.anc_matrix.cpu(), labels_onehot)
        else:
            if self.anc_matrix.device.type == 'cpu':
                self.anc_matrix = self.anc_matrix.to(self.device)
            labels_ancestors = torch.matmul(self.anc_matrix, labels_onehot)
        return torch.sum(torch.logical_and(preds_onehot, labels_ancestors), dim=0)

    def coverage(self, preds):
        """Returns the average coverage of the predictions."""
        coverage = torch.mean(self.coverage_vec[preds])
        return coverage.item()
    
    def avg_height(self, preds):
        """Returns the average height of the predictions."""
        avg_height = torch.mean(self.height_vec[preds])
        return avg_height.item()

    def save_to_file(self, path):
        """Saves hierarchy to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load_from_file(self, path):
        """Loads hierarchy from file."""
        with open(path, 'rb') as f:
            hierarchy = pickle.load(f)
            return hierarchy

    def prune_one_child_nodes(self):
        """Prunes nodes with a single child."""
        nodes_with_one_child = [node for node in self.tree.nodes if self.tree.out_degree(node) == 1]
        for node in nodes_with_one_child:
            parent = list(self.tree.predecessors(node))[0]
            child = list(self.tree.successors(node))[0]
            self.tree.add_edge(parent, child)
            self.tree.remove_node(node)
        # create a new tree to reindex the nodes
        new_tree = nx.DiGraph()
        old2new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(self.tree.nodes)}
        for old_idx, new_idx in old2new_idx.items():
            new_tree.add_node(new_idx, **self.tree.nodes[old_idx])
        for edge in self.tree.edges:
            new_tree.add_edge(old2new_idx[edge[0]], old2new_idx[edge[1]])
        self.tree = new_tree

    def sort_nodes_by_coverage(self):
        """Rebuilds the tree with the nodes indices sorted by coverage"""
        self.adj_matrix = torch.tensor(nx.adjacency_matrix(self.tree).todense()).to(self.device)
        self.anc_matrix = self.build_ancestor_matrix()
        self.nodes = [x for x in self.tree.nodes.data()]
        self.num_nodes = len(self.nodes)
        self.compute_coverage()

        new_tree = nx.DiGraph()
        nodes_sorted = sorted(self.nodes, key=lambda x: x[1]['coverage'], reverse=True)
        old2new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate([node[0] for node in nodes_sorted])}
        for old_idx, new_idx in old2new_idx.items():
            new_tree.add_node(new_idx, **self.tree.nodes[old_idx])
        for edge in self.tree.edges:
            new_tree.add_edge(old2new_idx[edge[0]], old2new_idx[edge[1]])
        self.tree = new_tree
        self.adj_matrix = torch.tensor(nx.adjacency_matrix(self.tree).todense()).to(self.device)
        self.anc_matrix = self.build_ancestor_matrix()
        self.nodes = [x for x in self.tree.nodes.data()]
        self.compute_coverage()