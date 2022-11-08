import time

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.signal as s
import scipy.linalg as la
import scipy.optimize as o

from tsa_python.basic_analysis import basic_analysis



def get_y_components(data, p, t, free_A):
    # TODO (Gustaf): Documentation
    if free_A is None:
        return -data[t-p:t][::-1]
    else:
        inds = np.arange(t-p,t)[::-1][free_A[1:]]
        out = -data[inds]
        return out

def get_e_components(data, est, p, q, t, free_C):
    # TODO (Gustaf): Documentation
    if free_C is None:
        e_inds = np.arange(t-q, t)[::-1]
    else:
        e_inds = np.arange(t-q,t)[::-1][free_C[1:]]

    out = np.zeros((e_inds.shape[0],))
    e_inds_change = e_inds[e_inds > p]
    
    for i, e_ind in enumerate(e_inds_change):
        out[i] = data[e_ind] - est[e_ind]
    return out

def pem_iteration(data, theta, p, q, free_A, free_C):
    # TODO (Gustaf): Documentation
    # returns the residual, the loss, and the gradient in theta
    # assert len(theta) == p + q, "Length of the parameter vector must be equal to p + q"

    est = np.zeros_like(data)
    gradients = np.zeros((theta.shape[0], data.shape[0] - p))
    for t in range(p, data.shape[0]):
        y_components = get_y_components(data, p, t, free_A)
        e_components = get_e_components(data, est, p, q, t, free_C)
        x = np.concatenate((y_components, e_components))
        yhat = np.dot(x, theta).squeeze()
        est[t] = yhat
        res_loc = data[t] - yhat
        gradients[:, t-p] = -2*res_loc*x.squeeze()/data.shape[0]
        # print(f"est.shape {est.shape}")
    res = (est - data).squeeze()[p:]

    gradient = np.sum(gradients, axis=1)
    gradient = np.expand_dims(gradient, axis=1)
    return res, np.sum(res**2)/res.shape[0], gradient
    # the number of estimations to be done is the sample data length minus p
    # for t in range(p, )

def theta_to_AC(theta, A_guess, C_guess, free_A=None, free_C=None):
    """Theta vector to estimate of A and C polynomials

    Args:
        theta (np.array): Array containing the estimated polynomials
        A (list): _description_
        C (list): _description_
        free_A (np.array, optional): Defaults to None.
        free_C (np.array, optional): Defaults to None.

    Returns:
        np.array: A polynomial
        np.array: C polynomial
    """
    if free_A is None:
        free_A = np.ones_like(A_guess, dtype=bool)
    if free_C is None:
        free_C = np.ones_like(C_guess, dtype=bool)
    free_A[0] = False
    free_C[0] = False
    A_out = np.zeros((A_guess.shape[0],))
    C_out = np.zeros_like(C_guess, dtype=float)
    A_out[0] = 1
    C_out[0] = 1
    A_inds = np.where(free_A)[0]
    for index, item in enumerate(A_inds):
        print(index)
        A_out[item] = theta[index,0]

    C_inds = np.where(free_C)[0]
    for index, item in enumerate(C_inds):
        C_out[item] = theta[index + len(A_inds)]
    return A_out, C_out

def pem(C_guess, A_guess, data, plot=False, free_C=None, free_A=None, verbose=False, step_size=0.01, gradient_norm_stop=0.1):
    """
    Prediction error minimization scheme.

    Starts out with gradient descent to get an increasingly better estimate of the driving white noise,
    which is estimated by the residual of the model in each time point. When the 2-norm of the gradient is small enough,
    the residuals are assumed to be a good approximation of the driving white noise, and an analytical solution is calculated.

    Args:
        C (list): Nominator polynomial, starting with 1
        A (list): Denominator polynomial, starting with 1 
        data (np.array): Data array of shape (1, N_samples). Each sample is in the "horizontal" dimension of the vector.
        plot (bool, optional): True if you want to plot basic analysis of residuals. Defaults to False.
        free_C (np.array, optional): A mask (one dimensional array with booleans) determining which parts of C are free. The non free parts are assumed to be zero. Defaults to None.
        free_A (np.array, optional): A mask (one dimensional array with booleans) determining which parts of A are free. Defaults to None.
        verbose (bool, optional): True if you want printouts from pem
        step_size (float, optional): Step size of the initial gradient descent. If the pem does not work for your data, you may try a smaller step size.
        gradient_norm_stop (float, optional): When the norm of the gradient is smaller than this number, the gradient descent stops.
    """
    assert A_guess[0] == 1, "First element of A polynomial must be one"
    assert C_guess[0] == 1, "First element of C polynomial must be one"
    if free_A is not None:
        assert len(free_A) == len(A_guess), "Length of the 'free_A' vector must be the same as the length of 'A'."
    if free_C is not None:
        assert len(free_C) == len(C_guess), "Length of the 'free_C' vector must be the same as the length of 'C'."
    p = len(A_guess) - 1
    q = len(C_guess) - 1

    est = np.zeros_like(data) # only samples from p and up contains estimates

    # first element is never free, always = 1
    if free_A is not None:
        free_A[0] = 0
    if free_C is not None:
        free_C[0] = 0
    
    # number of free variables for A and C part
    n_A = sum(free_A) if free_A is not None else len(A) - 1
    n_C = sum(free_C) if free_C is not None else len(C) - 1

    theta_length = n_A + n_C
    theta_start = np.zeros((theta_length, 1))
    if free_A is not None:
        theta_start[:n_A,0] = A_guess[free_A] # first part of the theta vector is the A part
    else:
        theta_start[:n_A,0] = A_guess[1:]
    if free_C is not None:
        theta_start[n_A:,0] = C_guess[free_C] # second part of the theta vector is the C part
    else:
        theta_start[n_A:,0] = C_guess[1:]
        
    # Start out with gradient descent
    theta = theta_start
    last_loss = 1e50
    for i in range(1000):
        res, loss, gradient = pem_iteration(data, theta, p, q, free_A, free_C)
        if verbose:
            print(f"gradient norm at iteration {i}: {la.norm(gradient)}")
            print(f"loss at iteration {i}: {loss}")

        if la.norm(gradient) < gradient_norm_stop: 
            if verbose:
                print(f"norm of gradient is lower than {gradient_norm_stop}. exiting.")
            break
        
        if loss > last_loss:
            if verbose:
                print("loss getting higher. exiting")
            break
        last_loss = loss
        theta = theta - step_size*gradient
    # Results after gradient descent
    if verbose:
        print(f"Result after gradient descent:\n\titerations: {i + 1}\n\tloss: {loss}")

    # Finish it with analytical solution
    X = np.zeros((data.shape[0] - p, len(theta_start)))
    Y = np.expand_dims(data[p:], axis=1)

    est = np.zeros_like(data)
    est[p:] = res + data[p:]
    for i, t in enumerate(range(p, data.shape[0])):
        # construct X
        y_components = get_y_components(data, p, t, free_A)
        e_components = get_e_components(data, est, p, q, t, free_C)
        x = np.concatenate((y_components, e_components))
        X[i,:] = x
    A = X.T.dot(X)
    B = X.T.dot(Y)
    theta_opt = la.solve(A,B)

    # do this just to get the final loss and see if it got worse or better
    res_final, loss_final, gradient_final = pem_iteration(data, theta_opt, p, q, free_A, free_C)
    if loss_final > loss:
        theta_out = theta
        res_out = res
    else:
        theta_out = theta_opt
        res_out = res_final

    if plot:
        basic_analysis(res_out)
    # print(f"A: {A}")
    A_out, C_out = theta_to_AC(theta_out, A_guess, C_guess, free_A=free_A, free_C=free_C)
    return C_out, A_out, res_out