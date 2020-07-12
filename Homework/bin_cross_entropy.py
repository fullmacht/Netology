import numpy as np
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, n_features=1, centers=None, cluster_std=1.0,
                            center_box=(-10.0, 10.0), shuffle=True, random_state=None, return_centers=False)

m, n = np.shape(x)

x = np.c_[ np.ones(m), x]

alpha = 0.01
theta = np.ones(2)
numIterations = 1000

def gradient_descent_2(alpha, x, y, numIterations,theta):
    x_transpose = x.transpose()
    theta_transpose = theta.transpose()
    for iter in range(0, numIterations):
        sigma = 1 / (1 + np.exp(np.dot(-theta_transpose,x_transpose)))
        # J= -y * np.log(sigma)-(1-y)*np.log(1-sigma)
    # print( "iter{} | J: {}".format(iter,J ) )
        gradient = (sigma - y) * x_transpose
        grad = gradient.T
        theta = theta - alpha * grad
        print('Theta0{},\n theta1{}'.format(theta[0],theta[1]))
    return theta

theta = gradient_descent_2(alpha, x, y, 1000,theta)

