import time

import numpy as np
import scipy.signal as s

import tsa_python.pem as p
import tsa_python.basic_analysis as ba

def main():
    # Generate some driving white noise
    mean = 0
    std = 1 
    num_samples = 10000
    samples = np.random.normal(mean, std, size=num_samples)

    # Define an ARMA process
    A = np.array([1, 0, -0.4, 0.4])
    C = np.array([1, 0.5])
   
    # Filter the white noise with the above defined polynomials with scipy.signal.lfilter
    filtered_samples = s.lfilter(C, A, samples)

    # Discard the first 9000 samples, "burn-in"
    filtered_samples = filtered_samples[9000:]
    ba.basic_analysis(filtered_samples)
    # Now "forget what you know" about the process and try fit a model to it
    A_guess = np.array([1, 0, 0, 0])
    C_guess = np.array([1, 0])
    
    # If you want to limit what degrees are free, create boolean masks as below
    free_C = C != 0
    free_A = A != 0

    tic = time.time()
    C_est, A_est, res = p.pem(C_guess, A_guess, filtered_samples, plot=True, free_A=free_A, free_C=free_C, verbose=False)
    toc = time.time()
    print(f"C estimation: \t {C_est}")
    print(f"A estimation: \t {A_est}")
    
    print(f"time elapsed: {toc - tic}")

if __name__ == '__main__':
    main()

