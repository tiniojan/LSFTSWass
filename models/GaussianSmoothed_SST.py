import os
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd 
import torch
from numba import njit
import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import cm


from tensordict import TensorDict

import platform
if platform.system() == 'Darwin':
    device = torch.device("mps")
elif platform.system() == 'Linux':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


params = {'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 8, # 10
          'text.usetex': True,
          'figure.figsize': (10, 8)}
plt.rcParams.update(params)
import seaborn as sns

from scr.utils import *
from scr.utils import ECDFTorch
from scr.kernels import Kernel

from scipy.stats import wasserstein_distance

def response_with_noise(num_samples, j_selected, T, time_series, sigma=0.1):
    """
    Generate response curves Y_j(t) for the selected j indices with added noise.
    
    Parameters:
    - j_selected (list of int): List of specific sample indices to generate responses for.
    - T (int): Number of time steps per sample (length of each Y_j(t)).
    - time_series (numpy array): The full time series data.
    - sigma (float): Standard deviation of the Gaussian noise.

    Returns:
    - Y (numpy array): A (len(j_selected), T) matrix containing response curves with noise.
    """

    Y = np.zeros((T-1, len(j_selected)))
    
    # Compute Y_j(t) for each selected j with noise
    for i, j in enumerate(j_selected):  
        epsilon = np.random.normal(0, sigma, size=T)  
        for t in range(1, T):
            Y[t-1, i] = time_series[num_samples*(t-1) + j] + epsilon[t]  # Add noise independently at each t
    
    return Y

@njit
def generate_L_rep_responses(num_samples, replications, j_selected, T, time_series, sigma):
    """
    Generate L replications of response curves Y_j(t) for each selected j with added noise.

    Parameters:
    - replications (int): Number of times to replicate the process (L).
    - j_selected (list of int): List of specific sample indices to generate responses for.
    - T (int): Number of time steps per sample (length of each Y_j(t)).
    - time_series (numpy array): The full time series data.
    - sigma (float): Standard deviation of the Gaussian noise.

    Returns:
    - Y_replications (numpy array): A (len(j_selected), replications, T) matrix 
                                    containing L noisy response curves for each j.
    """
    num_selected = len(j_selected)  
    Y_replications = np.zeros((T-1, replications, num_selected))  

    for i, j in enumerate(j_selected):  
        for l in range(replications):  
            epsilon = np.random.normal(0, sigma, size=T)  
            for t in range(1, T):
                Y_replications[t-1, l, i] = time_series[num_samples*(t-1) + j] + epsilon[t]  

    return Y_replications

from numba import njit, prange

@njit(parallel=True)
def generate_L_rep_responses_sigma(num_samples, replications, j_selected, T, time_series, times_sigma):
    """
    Generate L replications of response curves Y_j(t) for each selected j and multiple sigma values.
    Also gathers `Y_t` values for each `t` across all replications.

    Parameters:
    - replications (int): Number of times to replicate the process (L).
    - j_selected (numpy array of int64): Sample indices.
    - T (int): Number of time steps per sample (length of each Y_j(t)).
    - time_series (numpy array): The full time series data.
    - times_sigma (numpy array of float64): Different standard deviations of Gaussian noise.

    Returns:
    - Y_replications (numpy array): `(len(times_sigma), len(j_selected), replications, T)`, storing L noisy curves.
    - Y_t (numpy array): `(len(times_sigma), len(j_selected), T, replications)`, storing all replications for each `t`.
    """

    num_sigmas = len(times_sigma)
    num_selected = len(j_selected)

    # Initialize arrays with the correct shape
    Y_replications = np.zeros((num_sigmas, num_selected, replications, T-1), dtype=np.float64)
    Y_t = np.zeros((num_sigmas, num_selected, T-1, replications), dtype=np.float64)  # Gathered values for each t

    for s in prange(num_sigmas):  # Iterate over sigma values
        sigma = times_sigma[s]
        for j_idx in prange(num_selected):  # Iterate over selected j values
            j = j_selected[j_idx]
            for l in prange(replications):  # Iterate over L replications
                epsilon = np.random.normal(0, sigma, size=T)  # Independent noise
                for t in range(1, T):
                    Y_replications[s, j_idx, l, t-1] = time_series[num_samples * (t-1) + j] + epsilon[t]  

            # Gather Y_t values across replications for each `t`
            for t in range(1, T):
                Y_t[s, j_idx, t-1, :] = Y_replications[s, j_idx, :, t-1]  # All replications for t

    return Y_replications, Y_t

def cvbandwidth(h_values, T, X, Y_replications, j_selected, times_sigma, space_kernel, time_kernel):
    """
    Perform cross-validation to select the best bandwidth for kernel smoothing, 
    considering different values of `j_selected` and `sigma` **for a fixed T**.

    Parameters:
    - h_values (list): List of bandwidth values to test.
    - T (int): Fixed length of time steps per sample.
    - X (numpy array): Functional covariates of shape (T, num_samples).
    - Y_replications (numpy array): NumPy array of shape `(len(times_sigma), len(j_selected), replications, T)`.
    - j_selected (numpy array): Selected sample indices (j values).
    - times_sigma (numpy array): Different sigma values (noise levels).
    - space_kernel (str): Spatial kernel type.
    - time_kernel (str): Temporal kernel type.

    Returns:
    - best_h_dict (dict): Dictionary `{sigma: {j: best_h}}` mapping (sigma, j) to the best bandwidth.
    """

    best_h_dict = {sigma: {} for sigma in times_sigma}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Convert h_values to tensor once
    h_values = torch.tensor(h_values, dtype=torch.float32, device=device)

    # Precompute mask for leave-one-out cross-validation
    mask = torch.eye(T-1, dtype=torch.bool, device=device)

    # Convert X to a PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)  

    for s_idx, sigma in enumerate(times_sigma):  # Iterate over different noise levels
        for j_idx, j in enumerate(j_selected):  # Iterate over different sample indices
            best_h = None
            best_score = float('inf')

            # Load response Y for the specific (T, j, sigma)
            Y = torch.tensor(Y_replications[s_idx, j_idx], dtype=torch.float32, device=device)  

            # Take the mean of the replications (averaging L)
            Y = Y.mean(dim=0)  

            # Ensure Y has the correct shape
            if Y.dim() > 1:
                Y = Y.squeeze()  # Remove batch dimension if present

            # Initialize kernel
            GaussUnifKernel = Kernel(T=T, space_kernel=space_kernel, time_kernel=time_kernel)

            # Vectorized computation of kernel weights for all h_values
            for h in h_values:
                GaussUnifKernel.bandwidth = h

                # Compute weights for all t in T in a single batch
                weights = torch.stack([GaussUnifKernel.fit(X_tensor, t)[:T-1] for t in range(T-1)], dim=0)

                # Exclude diagonal elements (leave-one-out)
                weights_excluded = weights[~mask].reshape(T-1, T-2)

                Y_expanded = Y.unsqueeze(1).expand(T-1, T-1)  
                Y_excluded = Y_expanded[~mask].reshape(T-1, T-2)  

                # Compute m_hat_s using batch matrix multiplication
                m_hat_s = torch.sum(weights_excluded * Y_excluded, dim=1)
                scores = (Y - m_hat_s) ** 2
                mean_score = torch.mean(scores).item()

                if mean_score < best_score:
                    best_score = mean_score
                    best_h = h

            # Store the best bandwidth for the current (j, sigma)
            if best_h is not None:
                best_h_dict[sigma][j] = best_h.item()

    return best_h_dict

def computation_weights(best_h_dict, times_t, T, n_replications, X, time_kernel, space_kernel, device):
    """
    Compute Gaussian kernel weights using the best bandwidth for each (sigma, j)

    Returns:
    - gaussian_weights: `{sigma: {j: {t: {replication: weights}}}}`
    """
    gaussian_weights = {sigma: {j: {} for j in best_h_dict[sigma]} for sigma in best_h_dict}

    X_tensor = torch.tensor(X[:T-1], dtype=torch.float32, device=device) 

    for sigma, j_dict in best_h_dict.items():
        for j, best_h in j_dict.items():
            gaussian_kernel = Kernel(T=T-1, bandwidth=best_h, space_kernel=space_kernel, time_kernel=time_kernel, device=device)

            for t in times_t:  
                gaussian_weights[sigma][j][t] = {
                    replication: gaussian_kernel.fit(X_tensor, t-1) for replication in range(n_replications)
                }

    return gaussian_weights

def empirical_cdf(times_t, T, device, Y_t, j_selected, times_sigma):
    """
    Compute empirical CDFs using the precomputed Y_t values.

    Parameters:
    - times_t (list): List of time indices to compute CDFs.
    - T (int): Fixed number of time steps per sample.
    - device (str): Computation device ('cpu' or 'cuda').
    - Y_t (numpy array or tensor): Shape `(len(times_sigma), len(j_selected), replications, T)`.
    - j_selected (numpy array): List of selected j indices.
    - times_sigma (numpy array): List of different sigma values.

    Returns:
    - empirical_cdfs (dict): `{sigma: {j: {t: ecdf_y}}}`
    """

    # Ensure Y_t is converted to a PyTorch tensor
    if isinstance(Y_t, np.ndarray):
        Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)

    empirical_cdfs = {}

    for sigma_idx, sigma in enumerate(times_sigma):  # Loop over sigma values
        sigma_str = f"sigma:{float(sigma)}"
        empirical_cdfs[sigma_str] = {}

        for j_idx, j in enumerate(j_selected):  # Loop over selected j indices
            empirical_cdfs[sigma_str][f"j:{j}"] = {}

            tensor_dict = TensorDict(
                {
                    f"t:{t}_T:{T}": ECDFTorch(
                        Y_t[sigma_idx, j_idx, t, :].float().view(-1)  
                    )(
                        Y_t[sigma_idx, j_idx, t, :].float().view(-1)  
                    )
                    for t in times_t 
                },
                device=device,
            )

            empirical_cdfs[sigma_str][f"j:{j}"] = tensor_dict

    return empirical_cdfs

from datetime import datetime

def wasserstein_distances(times_t, T, n_replications, gaussian_weights, empirical_cdfs, 
                          Y_replications, times_sigma, j_selected, device, pplot=None):
    
    wass_distances_empirical_meanNW_iterations = {}

    for sigma_idx, sigma in enumerate(times_sigma):
        sigma_str = f"sigma:{float(sigma)}"

        for j_idx, j in enumerate(j_selected):
            tic = datetime.now()
            print('-' * 100)
            print(f"Running Wasserstein distances for σ={sigma}, j={j} at {tic} ...")

            x_rep = TensorDict(
                               
                {
                    f"t:{t}_T:{T}": torch.zeros((n_replications, T), dtype=torch.float16, device=device)
                    for t in times_t
                },
                device=device,
            )

            y_rep = TensorDict(
                {
                    f"t:{t}_T:{T}": torch.zeros((n_replications, T), dtype=torch.float16, device=device)
                    for t in times_t
                },
                device=device,
            )

            for t_idx, t in enumerate(times_t):  

                for replication in range(n_replications):
                    weights_array = torch.tensor(gaussian_weights[sigma][j][t][replication], dtype=torch.float32, device=device)
                    y_values = torch.tensor(Y_replications[sigma_idx, j_idx, t_idx, :], dtype=torch.float32, device=device)

                    weighted_ecdf = ECDFTorch(y_values, weights_array)

                    key = f"t:{t}_T:{T}"
                    x_rep[key][replication] = weighted_ecdf.x
                    y_rep[key][replication] = weighted_ecdf.y

                    if pplot is not None:
                        x = x_rep[key][replication].detach().cpu().numpy()
                        y = y_rep[key][replication].detach().cpu().numpy()
                        plt.plot(x, y, label=f"t:{t}_T:{T}")
                        plt.xlabel(r'$y$')
                        plt.ylabel(r'$\hat{F}_t(y|x)$')
                        plt.title(r'NW CDF estimators')
                        plt.legend()
                        plt.tight_layout()

            if pplot is not None:
                plt.show()

            # Compute Wasserstein distance
            wass_distances_empirical_meanNW = {}

            for t_idx, t in enumerate(times_t):
                emp_ccf = empirical_cdfs[sigma_str][f"j:{j}"][f"t:{t}_T:{T}"].detach().cpu().numpy()
                emp_mean_nw = y_rep[f"t:{t}_T:{T}"].mean(axis=0).detach().cpu().numpy()
                wass_distances_empirical_meanNW[f"t:{t}_T:{T}"] = wasserstein_distance(emp_ccf, emp_mean_nw)
    
            wass_distances_empirical_meanNW_iterations[f"sigma:{float(sigma)}_j:{j}"] = wass_distances_empirical_meanNW

            toc = datetime.now()
            print(f"Wasserstein distances finished at {toc}; time elapsed = {toc - tic}.")

    return wass_distances_empirical_meanNW_iterations

def plot_wasserstein_distances(wass_distances_results):
    """
    Creates separate figures for each chosen j, plotting Wasserstein distances for different sigma values over time.

    Parameters:
    - wass_distances_results (dict): Dictionary of Wasserstein distances with keys formatted as "sigma:{value}_j:{value}"
      and values as dictionaries of {t: distance}.
    """

    # Extract unique j values
    j_values = sorted(set([key.split("_j:")[1] for key in wass_distances_results.keys()]))
    
    # Extract unique sigma values
    sigma_values = sorted(set([key.split("_j:")[0] for key in wass_distances_results.keys()]))

    # Define color palette, markers, and linestyles
    colorlist = ["light orange", "dark orange", "salmon pink", "neon pink", "cornflower", "cobalt blue",
                 "blue green", "aquamarine", "dark orange", "golden yellow", "reddish pink", "black", "reddish purple"]
    colors = sns.xkcd_palette(colorlist)
    markers = ['o', 'p', 's', 'd', 'h', '<', '>', '8', 'P']
    linestylev = ['-', '--', ':', '-.', '--', ':']

    for j in j_values:
        plt.figure(figsize=(6, 4))
        
        for i, sigma in enumerate(sigma_values):
            key = f"{sigma}_j:{j}"
            if key in wass_distances_results:
                t_values = sorted(wass_distances_results[key].keys(), key=lambda x: int(x.split("_T:")[0].split(":")[1]))
                distances = [wass_distances_results[key][t] for t in t_values]
                
                # Convert time steps to normalized values (t/T)
                normalized_t_values = [int(t.split(":")[1].split("_T")[0]) / int(t.split("_T:")[1]) for t in t_values]

                # Extract sigma value as float
                sigma_float = float(sigma.split(":")[1])
                
                # Construct sigma label in exponential form
                exponent = int(np.log10(sigma_float))
                if exponent == 0:
                    sigma_label = r"$\sigma$: 1"
                else:
                    sigma_label = rf"$\sigma$: $10^{{{exponent}}}$"

                # Plot
                plt.plot(
                    normalized_t_values, distances,
                    label=sigma_label,
                    color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], 
                    markersize=6, 
                    lw=2,
                    linestyle=linestylev[i % len(linestylev)]
                )

        plt.xlabel(r'$t/T$', fontsize=16)
        plt.ylabel('Wasserstein Distance', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=14, loc='best', ncol=1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def main():
    if platform.system() == 'Darwin':
        device = torch.device("mps")
    elif platform.system() == 'Linux' or platform.system() == 'Windows':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")    

    import sys
    sys.path.append('D:/Users/tiniojan/ExperimentsPhD/Feb2025/LSFTSWass')
    path_data = 'D:/Users/tiniojan/ExperimentsPhD/Feb2025/LSFTSWass/data/'

    path_fig = '../figs'

    # Choose kerl type fitting
    space_kernel = "silverman" 
    time_kernel = "uniform"
    ttest =True
    
    df_SST = pd.read_csv(path_data + 'SST.csv')
    df_SST.head(), df_SST.shape

    sns.set(style="darkgrid")
    plt.rcParams["text.usetex"] = False
    plt.rcParams["figure.figsize"] = (10,3)
    plt.plot(df_SST['NINO1+2'], label="original data")
    plt.legend()
    plt.xlabel("$s$", fontsize= 16)
    plt.ylabel("SST", fontsize= 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    # Load data 
    time_series = df_SST['NINO1+2'].values  

    num_years = 75  
    num_months = num_years * 12  

    # If the dataset is shorter than expected, raise an error
    if len(time_series) != num_months:
        raise ValueError(f"Dataset length is {len(time_series)}, expected {num_months} months (1950-2024).")

    # Reshape into continuous sample curves
    num_samples = 36 # To increase T, refine the splitting to 12 and 6
    T = time_series.shape[0] //num_samples  

    sample_curves = time_series.reshape((T, num_samples))

    # Plot the continuous sample curves
    plt.figure(figsize=(10, 4))
    for i in range(T):
        plt.plot(range(1, num_samples + 1), sample_curves[i], linestyle='-', lw=.9, label=f"Sample {i+1}")

    plt.xlabel("$j$", fontsize= 20)
    plt.ylabel("SST", fontsize= 20)
    plt.yticks(np.linspace(20, 28, 5), fontsize=18)
    plt.xticks(np.round(np.linspace(1, num_samples, 7)).astype(int), fontsize=18)
    plt.show()

    Y = np.zeros((T-1, num_samples))

    for t in range(1, T):
        for j in range(num_samples):
            Y[t-1, j] = time_series[num_samples*(t-1) + j]

    # Plot all Y_t(j) curves
    plt.figure(figsize=(10, 4))

    for j in range(num_samples):
        plt.plot(range(1, T), Y[:, j], linestyle='-', alpha=0.7)

    plt.xlabel("t")
    plt.ylabel("SST")
    plt.xticks(np.linspace(1, 25, 6, dtype=int))
    plt.yticks(np.linspace(20, 28, 5))
    plt.show()

    # Select specific j values and plot Y_j(t) for all t
    j_selected = [0, 1, 2, 3, 4]  # Selected indices

    plt.figure(figsize=(10, 4))

    for j in j_selected:
        plt.plot(range(1, T), Y[:, j], linestyle='-', label = rf"$Y_{{t, T}}({j + 1})$")

    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$Y_{t, T}(j)$", fontsize=20)
    plt.xticks(np.linspace(1, T, 6, dtype=int), fontsize=18)
    plt.yticks(np.linspace(min(Y.flatten()), max(Y.flatten()), 5), fontsize=18) 
    plt.legend()
    plt.show()

    # Simulation parameters setting
    times_t = [9, 10, 11, 12, 13, 14, 15, 16] #Change this when changing the splitting parameter
    j_selected = np.array([7, 21, 35], dtype=np.int64) #Change this when changing the splitting parameter
    times_sigma = np.array([0.1, 0.01, 0.001, 0.0001], dtype=np.float64)
    n_replications = 500

    # Generating data
    Y_replications, Y_t = generate_L_rep_responses_sigma(num_samples, n_replications, j_selected, T, time_series, times_sigma)

    input_dir = "simulation_results_real"
    h_values = np.linspace(0.01, 0.99, 100)

    X = time_series[:num_samples * T].reshape((num_samples, T)).T  

    best_h_dict = cvbandwidth(h_values, T, X, Y_replications, j_selected, times_sigma, space_kernel, time_kernel)
    print("Best bandwidths found:", best_h_dict)

    ### Weights calculation
    input_dir = "simulation_results"
    output_dir = "gaussian_weights_output"
    os.makedirs(output_dir, exist_ok=True)
    gaussian_weights = computation_weights(best_h_dict, times_t, T, n_replications, X, time_kernel, space_kernel, device)

    # Empirical CDF calculation
    empirical_cdfs_results = empirical_cdf(times_t, T, device, Y_t, j_selected, times_sigma)

    ### Wasserstein distances
    wass_distances_empirical_meanNW_iterations = wasserstein_distances(times_t, T, n_replications, gaussian_weights, empirical_cdfs_results, Y_replications,
                                                                    times_sigma, j_selected, device, pplot=None)
    
    wass_distances_empirical_meanNW_iterations

    ### Plot of results
    plot_wasserstein_distances(wass_distances_empirical_meanNW_iterations)

if __name__ == '__main__':
    main()