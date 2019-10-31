###############################################################################
# Finishes PA 3
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

# Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *
from matplotlib.pyplot import cm

def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices

    return values:
    G: generator matrix
    H: parity check matrix
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H


def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)  

    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde  
    yTilde_i is obtained by flipping y_i with probability epsilon 
    '''
    ###############################################################################
    # TODO: Your code here!
    yTilde = y.flatten()
    mask = np.random.rand(len(y)) < epsilon
    yTilde[mask] = 1 - yTilde[mask]
    ###############################################################################
    return yTilde


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructFactorGraph(yTilde, H, epsilon):
    '''
    :param - yTilde: observed codeword
        type: numpy.ndarray containing 0's and 1's
        shape: 2N
    :param - H parity check matrix 
             type: numpy.ndarray
             shape: N x 2N
    :param epsilon - the probability that each bit is flipped to its complement
    return G factorGraph

    You should consider two kinds of factors:
    - M unary factors 
    - N each parity check factors
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = FactorGraph(numVar=M, numFactor=N+M)
    G.var = list(range(M))
    ##############################################################
    # To do: your code starts here
    factors = []
    # Add unary factors
    for var in range(M):
        scope = [var]
        card = [2]
        if yTilde[var] == 0:
            val = np.array([1 - epsilon, epsilon])
        else:
            val = np.array([epsilon, 1 - epsilon])
        G.varToFactor[var].append(len(factors))
        G.factorToVar[len(factors)] = [var]
        factors.append(Factor(scope=scope, card=card, val=val))
    # Add parity factors
    # You may find the function itertools.product useful
    # (https://docs.python.org/2/library/itertools.html#itertools.product)
    all_var = np.array(range(M))
    for factor in range(M, M+N):
        n_var = sum(H[factor-M, :])
        mask = [True if x == 1 else False for x in H[factor-M, :]]
        scope = list(all_var[mask])
        card = [2] * len(scope)
        val = np.zeros(np.prod(card))
        val_idx = 0
        for var in scope:
            G.varToFactor[var].append(factor)
        G.factorToVar[factor] = scope
        for ass in itertools.product([0, 1], repeat=n_var):
            val[val_idx] = 1 if sum(ass) % 2 == 0 else 0
            val_idx += 1
        factors.append(Factor(scope=scope, card=card, val=val))
    G.factors = factors
    ##############################################################

    return G


def do_part_a():
    yTilde = np.array([[1, 1, 1, 1, 1, 1]]).reshape(6, 1)
    print("yTilde.shape", yTilde.shape)
    H = np.array([
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    G = constructFactorGraph(yTilde, H, epsilon)
    ##############################################################
    # To do: your code starts here
    # Design two invalid codewords ytest1, ytest2 and one valid codewords ytest3.
    #  Report their weights respectively.

    ##############################################################
    ytest1 = [0, 1, 1, 0, 1, 0]
    ytest2 = [1, 0, 0, 0, 1, 0]
    ytest3 = [0, 0, 0, 0, 0, 0]
    print(
        G.evaluateWeight(ytest1),
        G.evaluateWeight(ytest2),
        G.evaluateWeight(ytest3))


def do_part_c():
    '''
    In part b, we provide you an all-zero initialization of message x, you should 
    apply noise on y to get yTilde, and then do loopy BP to obatin the
    marginal probabilities of the unobserved y_i's.
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    print(H)
    print(H.shape)
    epsilon = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    ##############################################################
    # To do: your code starts here
    M = len(y)
    yTilde = applyChannelNoise(y, epsilon)
    G = constructFactorGraph(yTilde, H, epsilon)
    G.runParallelLoopyBP(50)

    pos_one = [G.estimateMarginalProbability(i)[1] for i in range(M)]
    var_list = range(M)
    plt.plot(var_list, pos_one)
    plt.ylabel('posterior of one')
    plt.xlabel('codeword bit')
    plt.show()
    y_new = G.getMarginalMAP()
    print(sum(y_new == y.flatten()))
    assert(sum(y_new == y.flatten()) == M)
    ##############################################################


def do_part_de(numTrials, error, iterations=50):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability 
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    ##############################################################
    # To do: your code starts here
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    M = len(y)
    color=iter(cm.rainbow(np.linspace(0,1,numTrials)))
    legend_list = []
    legend_name = []
    for ex in range(numTrials):
        print(ex)
        yTilde = applyChannelNoise(y, error)
        G = constructFactorGraph(yTilde, H, error)
        hamming_distance = np.zeros(iterations)
        for i in range(iterations):
            G.runParallelLoopyBP(1)
            marginal_map = G.getMarginalMAP()
            hamming_distance[i] = np.sum(marginal_map != y.flatten())
        line, = plt.plot(range(iterations), hamming_distance, c=next(color))
        legend_list.append(line)
        legend_name.append('Exp ' + str(ex + 1))
    plt.ylabel('hamming distance')
    plt.xlabel('iterations')
    plt.legend(legend_list, legend_name)
    plt.xticks(np.arange(0, iterations+1, 1))
    plt.show()

    ##############################################################


def do_part_fg(error):
    '''
    param - error: the transmission error probability 
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    ##############################################################
    # To do: your code starts here
    # You should flattern img first and treat it as the message x in the previous parts.

    ################################################################
# print('Doing part (a): Should see 0.0, 0.0, >0.0')
# do_part_a()
# print('Doing part (c)')
# do_part_c()
print('Doing part (d)')
do_part_de(10, 0.06)
# print('Doing part (e)')
#do_part_de(10, 0.08)
#do_part_de(10, 0.10)
# print('Doing part (f)')
# do_part_fg(0.06)
# print('Doing part (g)')
# do_part_fg(0.10)


# yTilde = [1, 1, 0, 1]
# H = np.array([[1, 1, 0, 0], [1, 0, 0, 1]])
# G = constructFactorGraph(yTilde, H, 0.2)
# print(G.var)
# print()
# print(G.domain)
# print()
# print(G.varToFactor)
# print()
# print(G.factorToVar)
# print()
# print(G.factors)
# G.runParallelLoopyBP(10)
# print(G.estimateMarginalProbability(0))
# print(G.estimateMarginalProbability(1))
# print(G.estimateMarginalProbability(2))
# print(G.estimateMarginalProbability(3))
# print()
# print(G.getMarginalMAP())
