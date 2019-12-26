"""
John Milmore
Implementation of a simple decision tree
"""

import math

import numpy as np


class Leaf:
    """A Leaf simply stores a predicted target value"""
    def __init__(self, pred_value):
        self.pred_value = pred_value

    def predict(self, example=None):
        """Predict target value from example input."""
        return self.pred_value


class Node:
    """A Node represents an internal node of the decision tree.

        feat_idx -- index of the feature to split on at this node
        children -- dict mapping values of feature to next Node/Leaf in the tree
    """
    def __init__(self, feat_idx):
        self.feat_idx = feat_idx
        self.children = {}

    def add_child(self, val, child):
        self.children[val] = child

    def predict(self, example):
        return self.children[example[self.feat_idx]].predict(example)


def learn_decision_tree(X, y, feat_idxs, depth_limit):
    """Train a decision tree on input data.

        Arguments:
            X -- array-like, shape (num_examples, num_features)
                feature values
            y -- array-like, shape (num_examples)
                target values
            depth_limit -- int
            feat_idxs -- list
                indexes of the features (columns) to consider splitting in the decision tree

        Returns:
            tree -- Node
    """

    def decision_tree_learning(examples, targets, feat_idxs, parent_targets=[], depth=0):
        """Learn the decision tree structure.

            Arguments:
                examples -- array-like, shape (curr_num_examples, num_features)
                    examples that satisfy the current splitting of the tree
                attrs -- list
                    feature that have not been split on yet
                parent_examples -- array-like, shape (par_num_examples, num_features)
                    examples that satisfy the split at this nodes parent
                depth -- int
                    current depth of the decision tree

            Returns:
                tree - Node
                    learned decision tree

        p. 702 of Russell and Norvig AIMA
        """

        def entropy(targets):
            """Takes a list of examples and returns their entropy w.r.t. the target attribute"""
            length = len(targets)
            _, cts = np.unique(targets, return_counts=True)
            e = 0.0
            for ct in cts:
                p = ct / length
                e -= math.log2(p) * p
            return e

        def information_gain(parent_targets, children_targets):
            """
            Takes a `parent_targets` set and a set of subsets of the parent, `children`.
            Returns the information gain due to splitting `children` from `parent`.
            """
            par_len = len(parent_targets)
            child_entr = 0
            for child_targets in children_targets:
                p = len(child_targets) / par_len
                child_entr += p * entropy(child_targets)
            return entropy(parent_targets) - child_entr

        def plurality_value(targets):
            """Return the most frequent element in targets"""
            vals, cts = np.unique(targets, return_counts=True)
            return vals[np.argmax(cts)]

        def get_children_targets(examples, feat_idx, targets):
            """Return subsets of targets based on splitting examples by feat_idx"""
            children_targets = []
            for val in domains[feat_idx]:
                idx = np.where(examples.T[feat_idx] == val)
                children_targets.append(targets[idx])
            return children_targets

        def filter_examples_targets(examples, targets, feat_idx, val):
            """Return the values of the feature indexed by feat_idx that equal val along with their corresponding target
            values."""
            idx = np.where(examples.T[feat_idx] == val)
            return examples[idx], targets[idx]

        if len(examples) == 0:
            return Leaf(plurality_value(parent_targets))
        if len(np.unique(targets)) == 1:
            return Leaf(targets[0])
        if not feat_idxs:
            return Leaf(plurality_value(targets))
        if depth >= depth_limit:
            return Leaf(plurality_value(targets))

        # Best split: attribute that results in highest information gain when split on
        max_info_gain = -1
        best_attr = None
        for feat_idx in feat_idxs:
            info_gain = information_gain(targets, get_children_targets(examples, feat_idx, targets))
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attr = feat_idx

        # Recursively add nodes to the decision tree
        tree = Node(best_attr)
        for val in domains[best_attr]:
            examples_val, targets_val = filter_examples_targets(examples, targets, best_attr, val)
            subtree = decision_tree_learning(examples_val, targets_val, [idx for idx in feat_idxs if idx != best_attr],
                                             targets, depth + 1)
            tree.add_child(val, subtree)

        return tree

    domains = [list(set(x)) for x in X.T]

    return decision_tree_learning(X, y, feat_idxs)
