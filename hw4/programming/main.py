# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Bo Wang, Kratarth Goel, Aditya Grover
# Date : 2/17/2017

import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pickle
import copy


def markov_blanket(i, j, Y, X):
    '''
    return:
        the a list of Y values that are markov blanket of Y[i][j]
        e.g. if i = j = 1,
            the function should return [Y[0][1], Y[1][0], Y[1][2], Y[2][1], X[1][1]]
    '''
    ########
    # TODO #
    ########
    blanket = []
    for i_idx in [-1, 1]:
        if i + i_idx >= 0 and i + i_idx < len(Y):
            blanket.append(Y[i + i_idx][j])
    for j_idx in [-1, 1]:
        if j + j_idx >= 0 and j + j_idx < len(Y[0]):
            blanket.append(Y[i][j + j_idx])
    blanket.append(X[i][j])
    return blanket


def sampling_prob(markov_blanket):
    '''
    markov_blanket: a list of the values of a variable's Markov blanket
        The order doesn't matter (see part (a)). e.g. [1,1,-1,1]
    return:
         a real value which is the probability of a variable being 1 given its Markov blanket
    '''
    ########
    # TODO #
    ########
    raw_prob_one = np.exp(sum(markov_blanket))
    raw_prob_minus_one = np.exp(-sum(markov_blanket))
    return raw_prob_one/(raw_prob_one + raw_prob_minus_one)


def sample(i, j, Y, X, DUMB_SAMPLE=0):
    '''
    return a new sampled value of Y[i][j]
    It should be sampled by
        (i) the probability condition on all the other variables if DUMB_SAMPLE = 0
        (ii) the consensus of Markov blanket if DUMB_SAMPLE = 1
    '''
    blanket = markov_blanket(i, j, Y, X)

    if not DUMB_SAMPLE:
        prob = sampling_prob(blanket)
        if random.random() < prob:
            return 1
        else:
            return -1
    else:
        c_b = Counter(blanket)
        if c_b[1] >= c_b[-1]:
            return 1
        else:
            return -1


def get_posterior_by_sampling(filename, initialization='same', logfile=None, DUMB_SAMPLE=0):
    '''
    Do Gibbs sampling and compute the energy of each assignment for the image specified in filename.
    If not dumb_sample, it should run MAX_BURNS iterations of burn in and then
    MAX_SAMPLES iterations for collecting samples.
    If dumb_sample, run MAX_SAMPLES iterations and returns the final image.
    filename: file name of image in txt
    initialization: 'same' or 'neg' or 'rand'
    logfile: the file name that stores the energy log (will use for plotting later)
        look at the explanation of plot_energy to see detail
    DUMB_SAMPLE: equals 1 if we want to use the trivial reconstruction in part (e)
    return value: posterior, Y, frequencyZ
        posterior: an 2d-array with the same size of Y, the value of each entry should
            be the probability of that being 1 (estimated by the Gibbs sampler)
        Y: The final image (for DUMB_SAMPLE = 1, in part (e))
        frequencyZ: a dictionary with key: count the number of 1's in the Z region
                                      value: frequency of such count
    '''
    print("Read the file")
    X = read_txt_file(filename)

    ########
    # TODO #
    ########
    nabla = 1
    beta = 1
    B = 100
    S = 1000
    if initialization == 'same':
        Y = copy.deepcopy(X)
    elif initialization == 'neg':
        Y = [[-X[i][j] for j in range(len(X[0]))] for i in range(len(X))]
    else:
        Y = [[1 if random.random() < 0.5 else -1 for j in range(len(X[0]))]
             for i in range(len(X))]
    if not DUMB_SAMPLE:
        f = open(logfile, "w")
    posterior = np.zeros_like(Y)
    count_Z = []
    for t in range(B + S):
        old_Y = copy.deepcopy(Y)
        if t % 100 == 0:
            print(t)
        energy = 0
        for i in range(len(Y)):
            for j in range(len(Y[0])):
                Y[i][j] = sample(i, j, Y, X, DUMB_SAMPLE)
                energy -= nabla * Y[i][j] * X[i][j]
                if i >= 1:
                    energy -= beta * Y[i][j] * Y[i - 1][j]
                if j >= 1:
                    energy -= beta * Y[i][j] * Y[i][j - 1]
        if DUMB_SAMPLE:
            if old_Y == Y:
                print('Converged')
                break
            if t >= 30:
                print('Not converged')
                break
        else:
            if t < B:
                f.write(str(t + 1) + "\t" + str(energy) + "\t" + "B\n")
            else:
                f.write(str(t + 1) + "\t" + str(energy) + "\t" + "S\n")
                posterior = posterior + (np.array(Y) == 1)
                count_Z.append(np.sum(np.array(Y)[125:163, 143:174] == 1))
    if not DUMB_SAMPLE:
        f.close()
    posterior = posterior / S
    frequencyZ = Counter(count_Z)
    return posterior, Y, frequencyZ


def denoise_image(filename, initialization='rand', logfile=None, DUMB_SAMPLE=0):
    '''
    Do Gibbs sampling on the image and return the denoised one and frequencyZ
    '''
    posterior, Y, frequencyZ = \
        get_posterior_by_sampling(
            filename, initialization, logfile=logfile, DUMB_SAMPLE=DUMB_SAMPLE)

    if DUMB_SAMPLE:
        # for i in range(len(Y)):
        #     for j in range(len(Y[0])):
        #         Y[i][j] = .5*(1.0-Y[i][j])  # 1, -1 --> 1, 0
        return Y, frequencyZ
    else:
        denoised = -np.ones(posterior.shape)
        denoised[np.where(posterior > .5)] = 1
        return denoised, frequencyZ


# ===========================================
# Helper functions for plotting etc
# ===========================================

def plot_energy(filename):
    '''
    filename: a file with energy log, each row should have three terms separated by a \t:
        iteration: iteration number
        energy: the energy at this iteration
        S or B: indicates whether it's burning in or a sample
    e.g.
        1   -202086.0   B
        2   -210446.0   S
        ...
    '''
    its_burn, energies_burn = [], []
    its_sample, energies_sample = [], []
    with open(filename, 'r') as f:
        for line in f:
            it, en, phase = line.strip().split()
            if phase == 'B':
                its_burn.append(it)
                energies_burn.append(float(en))
            elif phase == 'S':
                its_sample.append(it)
                energies_sample.append(float(en))
            else:
                print("bad phase: -%s-" % phase)

    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_sample, energies_sample, 'b')
    plt.title(filename)
    plt.legend([p1, p2], ["burn in", "sampling"])
    plt.yticks(np.arange(min(min(energies_burn), min(energies_sample)),
                         max(max(energies_burn), max(energies_sample))+1, 10000))
    plt.xticks(np.arange(0, 1100, 200))
    plt.savefig(filename)
    plt.close()


def read_txt_file(filename):
    '''
    filename: image filename in txt
    return:   2-d array image
    '''
    f = open(filename, "r")
    lines = f.readlines()
    height = int(lines[0].split()[1].split("=")[1])
    width = int(lines[0].split()[2].split("=")[1])
    Y = [[0]*(width+2) for i in range(height+2)]
    for line in lines[2:]:
        i, j, val = [int(entry) for entry in line.split()]
        Y[i+1][j+1] = val
    return Y


def convert_to_png(denoised_image, title):
    '''
    save array as a png figure with given title.
    '''
    plt.imshow(denoised_image, cmap=plt.cm.gray)
    plt.title(title)
    plt.savefig(title + '.png')


def get_error(img_a, img_b):
    '''
    compute the fraction of all pixels that differ between the two input images.
    '''
    N = len(img_b[0])*len(img_b)*1.0
    return sum([sum([1 if img_a[row][col] != img_b[row][col] else 0 for col in range(len(img_a[0]))])
                for row in range(len(img_a))]
               ) / N


# ==================================
# doing part (c), (d), (e), (f)
# ==================================

def perform_part_c():
    '''
    Run denoise_image function with different initialization and plot out the energe functions.
    '''
    ########
    # TODO #
    ########
    # denoise_image('noisy_20.txt', 'rand', 'log_rand')
    # denoise_image('noisy_20.txt', 'neg', 'log_neg')
    denoised, frequencyZ = denoise_image('noisy_20.txt', 'same', 'log_same')

    # plot out the energy functions
    plot_energy("log_rand")
    plot_energy("log_neg")
    plot_energy("log_same")


def perform_part_d():
    '''
    Run denoise_image function with different noise levels of 10% and 20%, and report the errors between denoised images and original image
    '''
    ########
    # TODO #
    ########

    # save denoised images and original image to png figures
    print('Perform part d')
    denoised_10, _ = denoise_image('noisy_10.txt', 'same', 'log_same_10')
    denoised_20, _ = denoise_image('noisy_20.txt', 'same', 'log_same_10')
    file = open('denoised_10', 'wb')
    pickle.dump(denoised_10, file)
    file = open('denoised_20', 'wb')
    pickle.dump(denoised_20, file)
    file.close()
    orig_img = read_txt_file('orig.txt')
    convert_to_png(denoised_10, "denoised_10")
    convert_to_png(denoised_20, "denoised_20")
    convert_to_png(orig_img, "orig_img")
    columns = 3
    rows = 1
    noise_level = ['10', '20']
    print('Noisy 10 error: ' + str(get_error(denoised_10, orig_img)))
    print('Noisy 20 error: ' + str(get_error(denoised_20, orig_img)))
    for noise in noise_level:
        fig = plt.figure(figsize=(9, 3))
        ax = []
        ax.append(fig.add_subplot(rows, columns, 1))
        ax[-1].set_title('Original')
        img = plt.imread('orig_img.png')
        plt.imshow(img)
        plt.axis('off')
        ax.append(fig.add_subplot(rows, columns, 2))
        ax[-1].set_title('Noisy')
        img = plt.imread('noisy_' + noise + '.png')
        plt.imshow(img)
        plt.axis('off')
        ax.append(fig.add_subplot(rows, columns, 3))
        ax[-1].set_title('Denoised')
        img = plt.imread('denoised_' + noise + '.png')
        plt.imshow(img)
        plt.axis('off')
        plt.savefig('compare_'+str(noise))
    plt.close()


def perform_part_e():
    '''
    Run denoise_image function using dumb sampling with different noise levels of 10% and 20%.
    '''
    ########
    # TODO #
    ########
    denoised_dumb_10, _ = denoise_image(filename='noisy_10.txt', DUMB_SAMPLE=1)
    denoised_dumb_20, _ = denoise_image(filename='noisy_20.txt', DUMB_SAMPLE=1)
    # save denoised images to png figures
    convert_to_png(denoised_dumb_10, "denoised_dumb_10")
    convert_to_png(denoised_dumb_20, "denoised_dumb_20")

    orig_img = read_txt_file('orig.txt')
    noise_level = ['10', '20']
    print('Dumb Noisy 10 error: ' + str(get_error(denoised_dumb_10, orig_img)))
    print('Dumb Noisy 20 error: ' + str(get_error(denoised_dumb_20, orig_img)))
    rows = 1
    columns = 3
    for noise in noise_level:
        fig = plt.figure(figsize=(9, 3))
        ax = []
        ax.append(fig.add_subplot(rows, columns, 1))
        ax[-1].set_title('Original')
        img = plt.imread('orig_img.png')
        plt.imshow(img)
        plt.axis('off')
        ax.append(fig.add_subplot(rows, columns, 2))
        ax[-1].set_title('Noisy')
        img = plt.imread('noisy_' + noise + '.png')
        plt.imshow(img)
        plt.axis('off')
        ax.append(fig.add_subplot(rows, columns, 3))
        ax[-1].set_title('Denoised')
        img = plt.imread('denoised_dumb_' + noise + '.png')
        plt.imshow(img)
        plt.axis('off')
        plt.savefig('compare_dumb'+str(noise))
    plt.close()


def perform_part_f():
    '''
    Run Z square analysis
    '''
    orig_img = read_txt_file('orig.txt')
    d, f = denoise_image('noisy_10.txt',
                         initialization='rand', logfile='log_same')
    print('Noisy 10 error: ' + str(get_error(d, orig_img)))
    width = 1.0
    plt.clf()
    plt.bar(f.keys(), f.values(), width, color='b')
    plt.show()
    d, f = denoise_image('noisy_20.txt',
                         initialization='rand', logfile='log_same')
    print('Noisy 20 error: ' + str(get_error(d, orig_img)))
    
    plt.clf()
    plt.bar(f.keys(), f.values(), width, color='b')
    plt.show()


if __name__ == "__main__":
    # perform_part_c()
    # perform_part_d()
    # perform_part_e()
    perform_part_f()
