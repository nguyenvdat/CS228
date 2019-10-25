import os
import sys
import numpy as np
from scipy.misc import logsumexp
from collections import Counter
import random
import __future__

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges

# pseudocounts for uniform dirichlet prior
alpha = 0.1


def renormalize(cnt):
    '''
    renormalize a Counter()
    '''
    tot = 1. * sum(cnt.values())
    for a_i in cnt:
        cnt[a_i] /= tot
    return cnt

# --------------------------------------------------------------------------
# Naive bayes CPT and classifier
# --------------------------------------------------------------------------


class NBCPT(object):
    '''
    NB Conditional Probability Table (CPT) for a child attribute.  Each child
    has only the class variable as a parent
    '''

    def __init__(self, A_i):
        '''
        TODO create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT
            - A_i: the index of the child variable
        '''
        self.A_i = A_i
        self.cpt = np.zeros([2, 2])

    def learn(self, A, C):
        '''
        TODO populate any instance variables specified in __init__ to learn
        the parameters for this CPT
            - A: a 2-d numpy array where each row is a sample of assignments 
            - C: a 1-d n-element numpy where the elements correspond to the
              class labels of the rows in A
        '''
        m_class1 = sum(C)
        m_class0 = len(C) - sum(C)
        idx_class1 = np.flatnonzero(C)
        idx_class0 = np.flatnonzero(1 - C)
        self.cpt[1, 0] = (sum(A[idx_class0, self.A_i]) +
                          alpha) / (m_class0 + 2*alpha)
        self.cpt[0, 0] = 1 - self.cpt[1, 0]
        self.cpt[1, 1] = (sum(A[idx_class1, self.A_i]) +
                          alpha) / (m_class1 + 2*alpha)
        self.cpt[0, 1] = 1 - self.cpt[1, 1]

    def get_cond_prob(self, entry, c):
        '''
        TODO return the conditional probability P(X|Pa(X)) for the values
        specified in the example entry and class label c
            - entry: full assignment of variables 
                e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
            - c: the class 
        '''
        return self.cpt[entry[self.A_i], c]


class NBClassifier(object):
    '''
    NB classifier class specification
    '''

    def __init__(self, A_train, C_train):
        '''
        TODO create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to
        Suggestions for the attributes in the classifier:
            - P_c: the probabilities for the class variable C
            - cpts: a list of NBCPT objects
        '''
        n = A_train.shape[1]
        self.p_c = np.zeros(2)
        self.cpts = [NBCPT(i) for i in range(n)]
        self._train(A_train, C_train)

    def _train(self, A_train, C_train):
        '''
        TODO train your NB classifier with the specified data and class labels
        hint: learn the parameters for the required CPTs
            - A_train: a 2-d numpy array where each row is a sample of assignments 
            - C_train: a 1-d n-element numpy where the elements correspond to
              the class labels of the rows in A
        '''
        for cpt in self.cpts:
            cpt.learn(A_train, C_train)
        self.p_c[1] = (sum(C_train) + alpha) / (len(C_train) + 2*alpha)
        self.p_c[0] = 1 - self.p_c[1]

    def classify(self, entry):
        '''
        TODO return the log probabilites for class == 0 and class == 1 as a
        tuple for the given entry
        - entry: full assignment of variables 
        e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
        NOTE this must return both the predicated label {0,1} for the class
        variable and also the log of the conditional probability of this
        assignment in a tuple, e.g. return (c_pred, logP_c_pred)
        '''
        join_c_0 = self.sum_helper(entry, 0)
        join_c_1 = self.sum_helper(entry, 1)
        cond_c_0 = join_c_0 / (join_c_0 + join_c_1)
        cond_c_1 = 1 - cond_c_0
        if cond_c_1 > cond_c_0:
            return (1, np.log(cond_c_1))
        else:
            return (0, np.log(cond_c_0))

    def sum_helper(self, entry, c):
        missing_idxs = np.flatnonzero(entry == -1)
        if missing_idxs.size == 0:
            logP_join_c0 = np.sum(
                [np.log(cpt.get_cond_prob(entry, 0)) for cpt in self.cpts]) + np.log(self.p_c[0])
            logP_join_c1 = np.sum(
                [np.log(cpt.get_cond_prob(entry, 1)) for cpt in self.cpts]) + np.log(self.p_c[1])
            if c == 0:
                return np.exp(logP_join_c0)
            return np.exp(logP_join_c1)
        else:
            missing_idx = missing_idxs[0]
            entry0 = np.copy(entry)
            entry0[missing_idx] = 0
            entry1 = np.copy(entry)
            entry1[missing_idx] = 1
            return self.sum_helper(entry0, c) + self.sum_helper(entry1, c)


# --------------------------------------------------------------------------
# TANB CPT and classifier
# --------------------------------------------------------------------------
class TANBCPT(object):
    '''
    TANB CPT for a child attribute.  Each child can have one other attribute
    parent (or none in the case of the root), and the class variable as a
    parent
    '''

    def __init__(self, A_i, A_p):
        '''
        TODO create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT
         - A_i: the index of the child variable
         - A_p: the index of its parent variable (in the Chow-Liu algorithm,
           the learned structure will have a single parent for each child)
        '''
        self.A_i = A_i
        self.A_p = A_p
        self.cpt = np.zeros([2, 2, 2]) if A_p is not None else NBCPT(A_i)

    def learn(self, A, C):
        '''
        TODO populate any instance variables specified in __init__ to learn
        the parameters for this CPT
         - A: a 2-d numpy array where each row is a sample of assignments 
         - C: a 1-d n-element numpy where the elements correspond to the class
           labels of the rows in A
        '''
        if self.A_p is None:
            self.cpt.learn(A, C)
            return
        c_idx = [1 - C, C]
        p_idx = [1 - A[:, self.A_p], A[:, self.A_p]]
        for c_val in range(2):
            for p_val in range(2):
                parents_idx = c_idx[c_val] * p_idx[p_val]
                m = np.sum(parents_idx)
                self.cpt[1, c_val, p_val] = (
                    np.sum(A[np.flatnonzero(parents_idx), self.A_i]) + alpha) / (m + 2*alpha)
                self.cpt[0, c_val, p_val] = 1 - self.cpt[1, c_val, p_val]

    def get_cond_prob(self, entry, c):
        '''
        TODO return the conditional probability P(X|Pa(X)) for the values
        specified in the example entry and class label c  
            - entry: full assignment of variables 
                    e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
            - c: the class               
        '''
        if self.A_p is None:
            return self.cpt.get_cond_prob(entry, c)
        return self.cpt[entry[self.A_i], c, entry[self.A_p]]


class TANBClassifier(NBClassifier):
    '''
    TANB classifier class specification
    '''

    def __init__(self, A_train, C_train):
        '''
        TODO create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to
        _train()
            - A_train: a 2-d numpy array where each row is a sample of
              assignments 
            - C_train: a 1-d n-element numpy where the elements correspond to
              the class labels of the rows in A
        '''
        n = A_train.shape[1]
        self.p_c = np.zeros(2)
        self.cpts = [None] * n
        self._train(A_train, C_train)

    def _train(self, A_train, C_train):
        '''
        TODO train your TANB classifier with the specified data and class labels
        hint: learn the parameters for the required CPTs
            - A_train: a 2-d numpy array where each row is a sample of
              assignments 
            - C_train: a 1-d n-element numpy where the elements correspond to
              the class labels of the rows in A
        hint: you will want to call functions imported from tree.py:
            - get_mst(): build the mst from input data
            - get_tree_root(): get the root of a given mst
            - get_tree_edges(): iterate over all edges in the rooted tree.
              each edge (a,b) => a -> b
        '''
        mst = get_mst(A_train, C_train)
        root = get_tree_root(mst)
        edges = get_tree_edges(mst, root)
        for edge in edges:
            A_p, A_i = edge
            self.cpts[A_i] = TANBCPT(A_i, A_p)
            self.cpts[A_i].learn(A_train, C_train)
        self.cpts[root] = TANBCPT(root, None)
        self.cpts[root].learn(A_train, C_train)
        self.p_c[1] = (sum(C_train) + alpha) / (len(C_train) + 2*alpha)
        self.p_c[0] = 1 - self.p_c[1]

    def classify(self, entry):
        '''
        TODO return the log probabilites for class == 0 and class == 1 as a
        tuple for the given entry
        - entry: full assignment of variables 
        e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
        NOTE: this class inherits from NBClassifier and it is possible to
        write this method in NBClassifier, such that this implementation can
        be removed
        NOTE this must return both the predicated label {0,1} for the class
        variable and also the log of the conditional probability of this
        assignment in a tuple, e.g. return (c_pred, logP_c_pred)
        '''
        join_c_0 = self.sum_helper(entry, 0)
        join_c_1 = self.sum_helper(entry, 1)
        cond_c_0 = join_c_0 / (join_c_0 + join_c_1)
        cond_c_1 = 1 - cond_c_0
        if cond_c_1 > cond_c_0:
            return (1, np.log(cond_c_1))
        else:
            return (0, np.log(cond_c_0))

    def sum_helper(self, entry, c):
        missing_idxs = np.flatnonzero(entry == -1)
        if missing_idxs.size == 0:
            logP_join_c0 = np.sum(
                [np.log(cpt.get_cond_prob(entry, 0)) for cpt in self.cpts]) + np.log(self.p_c[0])
            logP_join_c1 = np.sum(
                [np.log(cpt.get_cond_prob(entry, 1)) for cpt in self.cpts]) + np.log(self.p_c[1])
            if c == 0:
                return np.exp(logP_join_c0)
            return np.exp(logP_join_c1)
        else:
            missing_idx = missing_idxs[0]
            entry0 = np.copy(entry)
            entry0[missing_idx] = 0
            entry1 = np.copy(entry)
            entry1[missing_idx] = 1
            return self.sum_helper(entry0, c) + self.sum_helper(entry1, c)

    def missing_posterior(self, entry, missing_idx):
        entry0 = np.copy(entry)
        entry0[missing_idx] = 0
        entry1 = np.copy(entry)
        entry1[missing_idx] = 1
        join_m_0 = self.sum_helper(entry0, 0) + self.sum_helper(entry0, 1)
        join_m_1 = self.sum_helper(entry1, 0) + self.sum_helper(entry1, 1)
        cond_m_0 = join_m_0 / (join_m_0 + join_m_1)
        cond_m_1 = 1 - cond_m_0
        if cond_m_1 > cond_m_0:
            return (1, np.log(cond_m_1))
        else:
            return (0, np.log(cond_m_0))


# load all data
A_base, C_base = load_vote_data()


def evaluate(classifier_cls, train_subset=False):
    '''
    evaluate the classifier specified by classifier_cls using 10-fold cross
    validation
    - classifier_cls: either NBClassifier or TANBClassifier
    - train_subset: train the classifier on a smaller subset of the training
      data
    NOTE you do *not* need to modify this function

    '''
    global A_base, C_base

    A, C = A_base, C_base

    # score classifier on specified attributes, A, against provided labels,
    # C
    def get_classification_results(classifier, A, C):
        results = []
        pp = []
        for entry, c in zip(A, C):
            c_pred, _ = classifier.classify(entry)
            results.append((c_pred == c))
            pp.append(_)
        # print 'logprobs', np.array(pp)
        return results
    # partition train and test set for 10 rounds
    M, N = A.shape
    tot_correct = 0
    tot_test = 0
    step = int(M / 10)
    for holdout_round, i in enumerate(range(0, M, step)):
        A_train = np.vstack([A[0:i, :], A[i+step:, :]])
        C_train = np.hstack([C[0:i], C[i+step:]])
        A_test = A[i:i+step, :]
        C_test = C[i:i+step]
        if train_subset:
            A_train = A_train[:16, :]
            C_train = C_train[:16]

        # train the classifiers
        classifier = classifier_cls(A_train, C_train)

        train_results = get_classification_results(
            classifier, A_train, C_train)
        # print '  train correct {}/{}'.format(np.sum(nb_results), A_train.shape[0])
        test_results = get_classification_results(classifier, A_test, C_test)
        tot_correct += sum(test_results)
        tot_test += len(test_results)

    return 1.*tot_correct/tot_test, tot_test


def evaluate_incomplete_entry(classifier_cls):

    global A_base, C_base

    # train a TANB classifier on the full dataset
    classifier = classifier_cls(A_base, C_base)

    # load incomplete entry 1
    entry = load_incomplete_entry()

    c_pred, logP_c_pred = classifier.classify(entry)

    print('  P(C={}|A_observed) = {:2.4f}'.format(c_pred, np.exp(logP_c_pred)))

    return

def evaluate_missing_posterior(classifier_cls, missing_idx):
    global A_base, C_base

    # train a TANB classifier on the full dataset
    classifier = classifier_cls(A_base, C_base)

    # load incomplete entry 1
    entry = load_incomplete_entry()

    m_pred, logP_m_pred = classifier.missing_posterior(entry, missing_idx)

    print('  P(M={}|A_observed) = {:2.4f}'.format(m_pred, np.exp(logP_m_pred)))

    return


def main():
    '''
    TODO modify or add calls to evaluate() to evaluate your implemented
    classifiers
    '''
    # print('NB Classifier')
    # accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
    # print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    #     accuracy, num_examples))

    # print('TANB Classifier')
    # accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
    # print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    #     accuracy, num_examples))

    # print('Naive Bayes Classifier on missing data')
    # evaluate_incomplete_entry(NBClassifier)

    # print('TANB Classifier on missing data')
    # evaluate_incomplete_entry(TANBClassifier)

    print('Posterior of A12')
    evaluate_missing_posterior(TANBClassifier, 11)


if __name__ == '__main__':
    main()
