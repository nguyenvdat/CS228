"""
CS 228: Probabilistic Graphical Models
Winter 2018
Programming Assignment 1: Bayesian Networks
Author: Aditya Grover
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
from scipy.misc import logsumexp


def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
    '''
    Plots a histogram.
    '''

    plt.figure()
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def get_p_z1(z1_val):
    '''
    Helper. Computes the prior probability for variable z1 to take value z1_val.
    P(Z1=z1_val)
    '''

    return bayes_net['prior_z1'][z1_val]


def get_p_z2(z2_val):
    '''
    Helper. Computes the prior probability for variable z2 to take value z2_val.
    P(Z2=z2_val)
    '''

    return bayes_net['prior_z2'][z2_val]


def get_p_xk_cond_z1_z2(z1_val, z2_val, k):
    '''
    Helper. Computes the conditional probability that variable xk assumes value 1 
    given that z1 assumes value z1_val and z2 assumes value z2_val
    P(Xk = 1 | Z1=z1_val , Z2=z2_val)
    '''

    return bayes_net['cond_likelihood'][(z1_val, z2_val)][0, k-1]


def get_p_x_cond_z1_z2(z1_val, z2_val):
    '''
    Computes the conditional probability of the entire vector x,
    given that z1 assumes value z1_val and z2 assumes value z2_val
    TODO
    '''
    x_mean = np.array([get_p_xk_cond_z1_z2(z1_val, z2_val, i+1)
                       for i in range(28*28)])
    return x_mean


def get_pixels_sampled_from_p_x_joint_z1_z2():
    '''
    This function should sample from the joint probability distribution specified by the model, 
    and return the sampled values of all the pixel variables (x). 
    Note that this function should return the sampled values of ONLY the pixel variables (x),
    discarding the z part.
    TODO. 
    '''
    p_z1_vals = [get_p_z1(z1_val) for z1_val in disc_z1]
    p_z2_vals = [get_p_z2(z2_val) for z2_val in disc_z2]
    z1_val = disc_z1[np.random.multinomial(1, p_z1_vals).astype(bool)][0]
    z2_val = disc_z2[np.random.multinomial(1, p_z2_vals).astype(bool)][0]
    sample_image = np.zeros(28*28)
    for i in range(28*28):
        sample_image[i] = np.random.binomial(
            1, get_p_xk_cond_z1_z2(z1_val, z2_val, i+1))
    return sample_image


def get_conditional_expectation(data):
    z1_mean = np.zeros(len(data))
    z2_mean = np.zeros(len(data))
    for i, image in enumerate(data):
        sum_cond_ll = 0
        z_mean = np.zeros(2)
        for z1_val in disc_z1:
            for z2_val in disc_z2:
                cond_ll = np.exp(
                    get_log_conditional_expectation(image, z1_val, z2_val))
                z_mean += cond_ll * np.array([z1_val, z2_val])
                sum_cond_ll += cond_ll
        z_mean /= sum_cond_ll
        z1_mean[i] = z_mean[0]
        z2_mean[i] = z_mean[1]
    return z1_mean, z2_mean


def q4():
    '''
    Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
    Your job is to implement get_pixels_sampled_from_p_x_joint_z1_z2
    '''

    plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2().reshape(
            28, 28), cmap='gray')
        plt.title('Sample: ' + str(i+1))
    plt.tight_layout()
    plt.savefig('a4', bbox_inches='tight')
    plt.show()
    plt.close()

    return


def q5():
    '''
    Plots the expected images for each latent configuration on a 2D grid.
    Your job is to implement get_p_x_cond_z1_z2
    '''

    canvas = np.empty((28*len(disc_z1), 28*len(disc_z2)))
    for i, z1_val in enumerate(disc_z1):
        for j, z2_val in enumerate(disc_z2):
            canvas[(len(disc_z1)-i-1)*28:(len(disc_z2)-i)*28, j*28:(j+1)*28] = \
                get_p_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

    plt.figure()
    plt.imshow(canvas, cmap='gray')
    plt.tight_layout()
    plt.savefig('a5', bbox_inches='tight')
    plt.show()
    plt.close()

    return


def get_log_conditional_expectation(data, z1_val, z2_val):
    '''
    TODO
    '''
    ll_one = bayes_net['cond_likelihood'][(z1_val, z2_val)][0, :]
    # ll = [get_p_xk_cond_z1_z2(z1_val, z2_val, i+1) if xi == 1 else 1 -
    #       get_p_xk_cond_z1_z2(z1_val, z2_val, i+1) for i, xi in enumerate(data)]
    ll = data * ll_one + (1 - data) * (1 - ll_one)
    return np.sum(np.log(ll)) + np.log(get_p_z1(z1_val)) + np.log(get_p_z2(z2_val))


def marginal_log_likelihood(data):
    log_likelihood = np.zeros(len(data))
    for i, image in enumerate(data):
        if i == 100:
            print(i)
        cond_ll = [get_log_conditional_expectation(
            image, z1_val, z2_val) for z1_val in disc_z1 for z2_val in disc_z2]
        log_likelihood[i] = logsumexp(cond_ll)
    return log_likelihood


def q6():
    '''
    Loads the data and plots the histograms. Rest is TODO.
    '''

    mat = loadmat('q6.mat')
    val_data = mat['val_x']
    test_data = mat['test_x']

    '''
	TODO
	'''
    real_marginal_log_likelihood = []
    corrupt_marginal_log_likelihood = []
    marginal_ll_val = marginal_log_likelihood(val_data)
    mean_val = np.mean(marginal_ll_val)
    std_val = np.std(marginal_ll_val)
    marginal_ll_test = marginal_log_likelihood(test_data)
    for ll in marginal_ll_test:
        if abs(ll - mean_val) > 3*std_val:
            corrupt_marginal_log_likelihood.append(ll)
        else:
            real_marginal_log_likelihood.append(ll)

    plot_histogram(real_marginal_log_likelihood, title='Histogram of marginal log-likelihood for real data',
                   xlabel='marginal log-likelihood', savefile='a6_hist_real')

    plot_histogram(corrupt_marginal_log_likelihood, title='Histogram of marginal log-likelihood for corrupted data',
                   xlabel='marginal log-likelihood', savefile='a6_hist_corrupt')

    return


def q7():
    '''
    Loads the data and plots a color coded clustering of the conditional expectations. Rest is TODO.
    '''

    mat = loadmat('q7.mat')
    data = mat['x']
    labels = mat['y']

    mean_z1, mean_z2 = get_conditional_expectation(data)

    plt.figure()
    plt.scatter(mean_z1, mean_z2, c=labels)
    plt.colorbar()
    plt.grid()
    plt.savefig('a7', bbox_inches='tight')
    plt.show()
    plt.close()

    return


def load_model(model_file):
    '''
    Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
    '''

    with open('trained_mnist_model', 'rb') as infile:
        cpts = pkl.load(infile)

    model = {}
    model['prior_z1'] = cpts[0]
    model['prior_z2'] = cpts[1]
    model['cond_likelihood'] = cpts[2]

    return model


def main():

    global disc_z1, disc_z2
    n_disc_z = 25
    disc_z1 = np.linspace(-3, 3, n_disc_z)
    disc_z2 = np.linspace(-3, 3, n_disc_z)

    global bayes_net
    bayes_net = load_model('trained_mnist_model')

    '''
	TODO: Using the above Bayesian Network model, complete the following parts.
	'''
    # q4()
    # q5()
    # q6()
    q7()

    return


if __name__ == '__main__':

    main()
