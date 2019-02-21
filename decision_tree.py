from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        trees = {}
        if entropy(y) != 0:
            info = {}
            for i in range(len(X[0])):
                unique = list(set([x[i] for x in X]))
                for j in unique:
                    X_left, X_right, y_left, y_right = partition_classes(X, y, i, j)
                    info[(i, j)] = information_gain(y, [y_left, y_right])

            vals = list(info.values())
            max_vals = max(vals)
            split_vals = list(info.keys())
            split_attr, split_vals2 = split_vals[vals.index(max_vals)]

            X_left, X_right, y_left, y_right = partition_classes(X, y, split_attr, split_vals2)

            trees['split_attr'] = split_attr
            trees['split_vals'] = split_vals2
            trees['left_tree'] = (X_left, y_left)
            trees['right_tree'] = (X_right, y_right)
        else:
            trees['leaf'] = 1
            trees['label'] = y[0]

        self.tree = trees
        
    def classify(self, record):
        tree1 = self.tree
        def build_tree(record,tree1):
            if not isinstance(tree1, tuple) and 'split_attr' in tree1.keys() :
              if record[tree1['split_attr']] <= tree1['split_vals']:
                if tree1['left_tree']:
                  tree1 = tree1['left_tree']
                  tree2 = build_tree(record,tree1)
              else:
                if tree1['right_tree']:
                  tree1 = tree1['right_tree']
                  tree2 = build_tree(record,tree1)
            elif not isinstance(tree1, tuple) and 'label' in tree1.keys():
              tree2 = tree1['label']
            elif isinstance(tree1, tuple):
              tree2 = max(set(tree1[-1]), key=tree1[-1].count)
            return tree2
        return(build_tree(record,tree1))
        
        
    
