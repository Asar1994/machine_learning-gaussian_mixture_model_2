import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
from mpl_toolkits.mplot3d import Axes3D


def generate_samples(mu, var):
    np.random.seed(1)
    samples = np.zeros((8000, 2))
    for i in range(len(mu)):
        samples[i*2000:(i+1)*2000] = np.random.multivariate_normal(mu[i], var[i], 2000)
    plt.plot(samples[:, 0], samples[:, 1], 'x')
    plt.axis('equal')
    plt.show()
    return samples


def initiate():
    pis = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    mus = np.array([[-2.3, -1.1], [-2.3, 1.8], [1.1, -2.1], [2.2, 1.8]])
    sigmas = np.array([[[0.100, 0.002], [0.001, 0.100]],
                       [[0.120, 0.000], [0.001, 0.060]],
                       [[0.040, 0.001], [0.000, 0.005]],
                       [[0.021, 0.002], [0.000, 0.083]]])
    return pis, mus, sigmas


def solve(data, max_iter, pi, mu, cov):

    print("starting with:")
    print("pis = ", pi)
    print("mus = \n", mu)
    print("covs = \n", cov)
    print()

    plot_data(mu, cov, pi)

    converged = False
    wait = 0
    for it in range(max_iter):
        if not converged:
            """E-Step"""
            r, m_c, pi = e_step(data, mu, cov, pi)

            """M-Step"""
            mu0, cov = m_step(data, r, m_c)

            print("iteration", it, ":")
            print("pis = ", pi)
            print("mus = \n", mu0)
            print("cov = \n", cov)
            print()
            if it % 2 == 0:
                plot_data(mu, cov, pi)

            """convergence condition"""
            shift = np.linalg.norm(np.array(mu) - np.array(mu0))
            mu = mu0
            if shift < 0.0001:
                wait += 1
                if wait > 5:
                    converged = True
            else:
                wait = 0
    final_result(data, pi, mu, cov)


def e_step(data, mu, cov, pi):
    clusters_number = len(pi)

    """creating estimations gaussian density functions"""
    gaussian_pdf_list = []
    for j in range(clusters_number):
        gaussian_pdf_list.append(multivariate_normal(mu[j], cov[j]))

    """Create the array r with dimensionality nxK"""
    r = np.zeros((len(data), clusters_number))

    """Probability for each data point x_i to belong to gaussian g """
    for c, g, p in zip(range(clusters_number), gaussian_pdf_list, pi):
        r[:, c] = p * g.pdf(data)

    """Normalize the probabilities 
    each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to cluster c"""
    for i in range(len(r)):
        sum1 = np.dot([1, 1, 1, 1], r[i, :].reshape(clusters_number, 1))
        r[i] = r[i] / sum1

    """calculate m_c
    For each cluster c, calculate the m_c and add it to the list m_c"""
    m_c = []
    for c in range(clusters_number):
        m = np.sum(r[:, c])
        m_c.append(m)

    """calculate pi
    probability of occurrence for each cluster"""
    for k in range(clusters_number):
        pi[k] = (m_c[k] / np.sum(m_c))

    return r, m_c, pi


def m_step(data, r, m_c):
    clusters_number = len(m_c)

    mu = []
    """calculate mu"""
    for k in range(clusters_number):
        mu.append(np.dot(r[:, k].reshape(len(data)), data) / m_c[k])
    mu = np.array(mu)

    cov = []
    """calculate sigma"""
    for c in range(clusters_number):
        dr = np.stack((r[:, c], r[:, c]), axis=-1)
        temp = (dr * (data - mu[c])).T @ (data - mu[c])
        cov.append(temp / m_c[c])
    cov = np.array(cov)

    return mu, cov


def plot_data(mu, cov, p):
    X, Y, gaussians = drawable_gaussian(p, mu, cov)

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, gaussians, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def final_result(samples, p, mu, cov):
    X, Y, gaussians = drawable_gaussian(p, mu, cov)

    fig, ax = plt.subplots()

    CS = ax.contour(X, Y, gaussians)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('contour of Gaussian distributions and the data')

    new_list1 = np.array(random.sample(list(samples), 500))
    plt.plot(new_list1[:, 0], new_list1[:, 1], 'x')

    plt.axis('equal')
    plt.show()


def drawable_gaussian(p, mu, cov):
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-2, 2, 0.01)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    gaussians = 0
    for i in range(len(p)):
        gaussians += p[i] * multivariate_normal(mu[i], cov[i]).pdf(pos)
    return X, Y, gaussians


def main():
    """generate samples from different distributions"""
    mu0 = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    sig0 = [[[0.1, 0.01], [0.01, 0.2]], [[0.2, 0], [0, 0.2]], [[0.18, 0], [0, 0.19]], [[0.12, 0], [0, 0.41]]]
    samples = generate_samples(mu0, sig0)

    """initialize pi mu and sigma of distributions"""
    pis, mus, sigmas = initiate()

    """solve the problem using EM algorithm"""
    solve(samples, 200, pis, mus, sigmas)


if __name__ == '__main__':
    main()
