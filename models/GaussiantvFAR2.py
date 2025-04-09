

"""
Ref:  A. van Delft and M. Eichler. Locally stationary functional time series. Electronic Journal of Statistics, 12:107–170, 2018.
"""
import os
import warnings
warnings.filterwarnings('ignore')

import time
from datetime import datetime
import numpy as np
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

path_fig = '../figs'
process = 'GaussiantvFAR(2)'

# Choose kerl type fitting
space_kernel = "gaussian"  
time_kernel = "tricube"

ttest =True
device = torch.device("cpu")

# Define the construct_A_u function 
def construct_A_u_1(J):
    i_indices = np.arange(1, J + 1, dtype=float).reshape(-1, 1)
    j_indices = np.arange(1, J + 1, dtype=float).reshape(1, -1)
    sigma_ij = np.sqrt(np.exp(- (i_indices - 3) - (j_indices -3)))
    A_u_1 = np.random.normal(0, sigma_ij, size=(J, J))
    singular_values_1 = np.linalg.svd(A_u_1, compute_uv=False)
    schatten_infinity_norm_1 = np.max(singular_values_1)
    return A_u_1, schatten_infinity_norm_1

def construct_A_u_2(J):
    i_indices = np.arange(1, J + 1, dtype=float).reshape(-1, 1)
    j_indices = np.arange(1, J + 1, dtype=float).reshape(1, -1)
    sigma_ij = np.sqrt(1 / (i_indices**4 + j_indices))
    A_u_2 = np.random.normal(0, sigma_ij, size=(J, J))
    singular_values_2 = np.linalg.svd(A_u_2, compute_uv=False)
    schatten_infinity_norm_2 = np.max(singular_values_2)
    return A_u_2, schatten_infinity_norm_2

# Define the construct_B_t function 
def construct_B_t_1(J, u):
    A_u_1, schatten_infinity_norm_1 = construct_A_u_1(J)
    B_t_1 = ((0.4 * np.cos(1.5 - np.cos(np.pi * u))) * A_u_1) / schatten_infinity_norm_1 if schatten_infinity_norm_1 != 0 else ((0.4 * np.cos(1.5 - np.cos(np.pi * u))) * A_u_1)
    return B_t_1

def construct_B_t_2(J):
    A_u_2, schatten_infinity_norm_2 = construct_A_u_2(J)
    B_t_2 = (-0.5 * A_u_2) / schatten_infinity_norm_2 if schatten_infinity_norm_2 != 0 else -0.5 * A_u_2
    return B_t_2

# Fourier basis functions
@njit
def fourier_basis(j, tau):
    if j % 2 == 0:  # Cosine terms for even indices
        return np.sqrt(2) * np.cos((j // 2) * np.pi * tau)
    else:           # Sine terms for odd indices
        return np.sqrt(2) * np.sin((j // 2 + 1) * np.pi * tau)

# Define the eta_t function based on Fourier basis functions
@njit
def eta_t_fourier(J, tau):
    eta_t = np.zeros(J)
    for j in range(1, J):
        sigma_j = (np.pi * (j - 1.5))**-2
        coeff = np.random.normal(0, np.sqrt(sigma_j))
        eta_t += coeff * fourier_basis(j, tau)
    return eta_t

# Simulation process
def simulation_covariate(T, N, J, tau):
    curve = np.zeros((T, J))
    X = np.zeros((T, N))
    for t in range(1, T):
        u = t / T
        B_t_1_matrix = construct_B_t_1(J, u)  
        B_t_2_matrix = construct_B_t_2(J)     
        eta_t = eta_t_fourier(J, tau)    
        curve[t, :] = B_t_1_matrix @ curve[t - 1, :] + B_t_2_matrix @ curve[t - 2, :] + eta_t
    
        for j in range(J):
            X[t, :] += curve[t, j] * fourier_basis(j, tau)
    return curve, X


# Define the regression operator m(u, x)
@njit
def m_star(u, X_t, tau):
    integral = np.trapz(np.cos(np.pi * X_t), tau)  # Compute the integral over tau
    return 2.5 * np.sin(2 * np.pi * u) * integral


# Define the response variable generation function
@njit
def generate_1_response(T, N, tau, X):
    Y = np.zeros(T)  # Initialize response variable
    for t in range(T):
        # Compute u = t / T
        u = t / T
        
        # Compute the regression operator m(u, X_t)
        m_u_x = m_star(u, X[t, :], tau)
        
        # Generate noise ε_t,T
        epsilon = np.random.normal(0, 1)
        
        # Compute Y_t,T
        Y[t] = m_u_x + epsilon

    return Y

@njit
def generate_L_rep_responses(T, N, tau, X, L):
    Y_replication = np.zeros((L, T))  

    for replication in range(L):
        Y_replication[replication, :] = generate_1_response(T, N, tau, X)

    return Y_replication


def simulation_L_reps_process(times_T, times_t_dict, iterations, n_replications, N, J, tau, output_dir):
    tic = datetime.now()
    print('-' * 100)
    print("Simulation of L-replications with T-samples of process ...")

    Y_iteration = None
    Y_replications = {f"T:{T}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for T in times_T}
    X_dict = {f"T:{T}": {} for T in times_T}
    curve = {f"T:{T}": {} for T in times_T}

    for iteration in range(iterations):
        # Initialize or reset variables for each iteration
        Y_replications = {f"T:{T}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for T in times_T}
        X_dict = {f"T:{T}": {} for T in times_T}
        curve = {f"T:{T}": {} for T in times_T}
        Y_t = TensorDict(
            {f"t:{t}_T:{T}": torch.empty(n_replications, dtype=torch.float16)
             for T in times_T for t in times_t_dict[T]},
            device=device
        )

        for T in times_T:
            for replication in range(n_replications):
                curve_T, X_T = simulation_covariate(T, N, J, tau)
                Y_replications[f"T:{T}"][replication] = torch.tensor(generate_1_response(T, N, tau, X_T), dtype=torch.float16).to(device)
                X_dict[f"T:{T}"][replication] = X_T
                curve[f"T:{T}"][replication] = curve_T

        for T in times_T:
            times_t = times_t_dict[T]
            for t in times_t:
                # Extract data using list comprehension and avoid creating intermediate tensors
                data = [Y_replications[f"T:{T}"][replication][t - 1].item()
                        for replication in range(n_replications)]
                Y_t[f"t:{t}_T:{T}"][:] = torch.tensor(data, dtype=torch.float16, device=device)

        # Save the current iteration's results to disk using pickle
        iteration_results = {
            "Y_replications": Y_replications,
            "X_dict": X_dict,
            "Y_iteration": Y_t 
        }
        with open(os.path.join(output_dir, f"iteration_{iteration + 1}.pkl"), "wb") as f:
            pickle.dump(iteration_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Keep the last iteration's X_tvar_1 for returning
        Y_iteration = Y_t

        # Release memory after saving
        del Y_replications, X_dict, iteration_results
        torch.cuda.empty_cache()  # Use if running on GPU

        print(f"Montecarlo {iteration + 1} completed.")

    toc = datetime.now()
    print(f"Simulation completed; time elapsed = {toc - tic}.")

    # Ensure that variables are returned correctly
    if Y_iteration is not None:
        return Y_iteration, {f"T:{T}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for T in times_T}, {f"T:{T}": {} for T in times_T}
    else:
        raise ValueError("Y_t was not properly assigned during the iterations.")


def cvbandwidth(h_values, times_T, X_dict, Y_replications, iterations, n_replications, space_kernel, time_kernel, input_dir):
    best_h_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Convert h_values to tensor once
    h_values = torch.tensor(h_values, dtype=torch.float16, device=device)

    # Precompute constants and reusable tensors
    T_max = max(times_T)
    mask = torch.eye(T_max, dtype=torch.bool, device=device)

    # Precompute the mask for each T in times_T
    masks = {T: mask[:T, :T] for T in times_T}

    
    iteration = 0
    best_h_dict[f"iteration_{iteration+1}"] = {}

    with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
        iteration_results = pickle.load(f)

    X_dict = iteration_results["X_dict"]
    Y_replications = iteration_results["Y_replications"]

    replication = 0
    best_h_dict[f"iteration_{iteration+1}"][f"replication_{replication}"] = {}

    for T in times_T:
        best_h = None
        best_score = float('inf')

        X = torch.tensor(X_dict[f"T:{T}"][replication], dtype=torch.float16, device=device)
        Y = torch.tensor(Y_replications[f"T:{T}"][replication], dtype=torch.float16, device=device)

        # Initialize kernel once per T
        GaussUnifKernel = Kernel(T=T, space_kernel=space_kernel, time_kernel=time_kernel)

        # Precompute the mask for the current T
        current_mask = masks[T]

        for h in h_values:
            GaussUnifKernel.bandwidth = h

            # Compute weights for all t in T in a single batch
            weights = torch.stack([GaussUnifKernel.fit(X, t) for t in range(T)], dim=0)

            # Exclude diagonal elements
            weights_excluded = weights[~current_mask].reshape(T, T-1)
            Y_excluded = Y.repeat(T, 1)[~current_mask].reshape(T, T-1)

            m_hat_s = torch.sum(weights_excluded * Y_excluded, dim=1)
            scores = (Y - m_hat_s) ** 2
            mean_score = torch.mean(scores).item()

            if mean_score < best_score:
                best_score = mean_score
                best_h = h

        # Store the best h for the current T and replication
        best_h_dict[f"iteration_{iteration+1}"][f"replication_{replication}"][f"T:{T}"] = best_h.item()

    return best_h_dict


def computation_weights(best_h_dict, times_t_dict, times_T, iterations, n_replications, X_dict,
                        time_kernel, space_kernel, input_dir, output_dir, device):
    
    # Extract best bandwidths for each T
    uniform_bandwidths = {
        T: best_h_dict.get("iteration_1", {}).get("replication_0", {}).get(f"T:{T}", None)
        for T in times_T
    }
    
    for iteration in range(iterations):
        tic = datetime.now()
        print('-' * 100)
        print(f"Running computation weights for Montecarlo {iteration + 1} starts at {tic} ...")
        
        # Initialize Gaussian kernel for each T using its uniform bandwidth
        gaussian_kernel = {}
        for T in times_T:
            best_h = uniform_bandwidths.get(T, None)
            if best_h is None:
                raise ValueError(f"Best bandwidth not found for T={T}.")
            gaussian_kernel[f"T:{T}"] = Kernel(T=T, bandwidth=best_h, space_kernel=space_kernel, time_kernel=time_kernel, device=device)
        
        gaussian_weights = {
            f"t:{t}_T:{T}": {}
            for T in times_T for t in times_t_dict[T]
        }

        # Load data from pickle files
        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)
    
        X_dict = iteration_results["X_dict"]

        for T in times_T:
            times_t = times_t_dict[T]
            for t in times_t:
                gaussian_weights[f"t:{t}_T:{T}"] = {
                    str(replication): gaussian_kernel[f"T:{T}"].fit(
                        torch.tensor(X_dict[f"T:{T}"].get(str(replication), X_dict[f"T:{T}"][replication]), device=device),
                        t - 1
                    ) for replication in range(n_replications)
                }
    
        # Save the gaussian_weights for the current iteration to disk using pickle
        with open(os.path.join(output_dir, f"gaussian_weights_iteration_{iteration + 1}.pkl"), "wb") as f:
            pickle.dump(gaussian_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        # Release memory after saving
        del gaussian_kernel, X_dict, iteration_results
        torch.cuda.empty_cache()  # Clear GPU cache if using CUDA

        toc = datetime.now()
        print(f"Weights computation complete for Montecarlo {iteration + 1} at {toc}; time elapsed = {toc - tic}.")
    
    # Return the result after all iterations are complete
    return gaussian_weights

def empirical_cdf(times_t_dict, times_T, device, iterations, input_dir, Y_t):
    empirical_cdfs_iterations = {}

    for iteration in range(iterations):

        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)

        Y_t = iteration_results["Y_iteration"]

        empirical_cdfs = TensorDict(
            {
                f"t:{t}_T:{T}": ECDFTorch(Y_t[f"t:{t}_T:{T}"]).y
                for T in times_T for t in times_t_dict[T]  
            },
            device=device,
        )
    
        empirical_cdfs_iterations[f"Iteration:{iteration + 1}"] = empirical_cdfs
    
    return empirical_cdfs_iterations

def wasserstein_distances(times_t_dict, times_T, n_replications, iterations, gaussian_weights, empirical_cdfs_iterations, Y_replications, 
                          input_dir, input_dir_weights, output_dir, device, pplot=None):
    
    wass_distances_empirical_meanNW_iterations = {}

    for iteration in range(iterations):
        tic = datetime.now()
        print('-' * 100)
        print(f"Running wasserstein distances starts at {tic} ...")

        # Load gaussian_weights_tensor from disk for the current iteration
        with open(os.path.join(input_dir_weights, f"gaussian_weights_iteration_{iteration + 1}.pkl"), "rb") as f:
            gaussian_weights = pickle.load(f)

        # Load the iteration results for the current iteration to access Y_replications
        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)
    
        Y_replications = iteration_results["Y_replications"]

        # Initialize x_rep and y_rep TensorDicts for the current iteration
        x_rep = TensorDict(
            {
                f"t:{t}_T:{T}": torch.zeros((n_replications, T + 1), dtype=torch.float16)
                for T in times_T for t in times_t_dict[T]
            },
            device=device,
        )

        y_rep = TensorDict(
            {
                f"t:{t}_T:{T}": torch.zeros((n_replications, T + 1), dtype=torch.float16)
                for T in times_T for t in times_t_dict[T]
            },
            device=device,
        )

        for replication in range(n_replications):
            for T in times_T:
                times_t = times_t_dict[T]
                for t in times_t:
                    # Calculate the weighted ECDF using data loaded from disk
                    weighted_ecdf = ECDFTorch(
                        Y_replications[f"T:{T}"][replication],
                        gaussian_weights[f"t:{t}_T:{T}"][str(replication)]
                    )

                    x_rep[f"t:{t}_T:{T}"][replication] = weighted_ecdf.x
                    y_rep[f"t:{t}_T:{T}"][replication] = weighted_ecdf.y

                    # Optional plotting
                    if pplot is not None:
                        x = x_rep[f"t:{t}_T:{T}"][replication].detach().cpu().numpy()
                        y = y_rep[f"t:{t}_T:{T}"][replication].detach().cpu().numpy()
                        plt.plot(x, y, label=f"t:{t}_T:{T}")
                        plt.xlabel(r'$y$')
                        plt.ylabel(r'$\hat{F}_t(y|x)$')
                        plt.title(r'NW CDF estimators, $\hat{F}_{t}(y|{x})=\sum_{a=1}^T\omega_{a}(\frac{t}{T},{x})\mathbf{1}_{Y_{a,T}\leq y}$')
                        plt.legend()
                        plt.tight_layout()

                    if pplot is not None:
                        plt.show()
        
        iteration_empirical_cdfs = empirical_cdfs_iterations[f"Iteration:{iteration + 1}"]

        wass_distances_empirical_meanNW = {}

        for T in times_T:
            times_t = times_t_dict[T]  
            for t in times_t:
                emp_ccf = iteration_empirical_cdfs[f"t:{t}_T:{T}"].detach().cpu().numpy()
                emp_mean_nw = y_rep[f"t:{t}_T:{T}"].mean(axis=0).detach().cpu().numpy()
                wass_distances_empirical_meanNW[f"t:{t}_T:{T}"] = wasserstein_distance(emp_ccf, emp_mean_nw)
    
        wass_distances_empirical_meanNW_iterations[f"Iteration:{iteration + 1}"] = wass_distances_empirical_meanNW

        # Save the results to disk using pickle
        with open(f"{output_dir}/wass_distances_empirical_meanNW_iterations.pkl", "wb") as f:
            pickle.dump(wass_distances_empirical_meanNW_iterations, f)

        toc = datetime.now()
        print(f"Wasserstein distances at {toc}; time elapsed = {toc - tic}.")
    
    return wass_distances_empirical_meanNW_iterations


def wass_stats(input_dir, output_dir, times_T, times_t_dict, iterations, wass_distances_empirical_meanNW_iterations):
    with open(os.path.join(f"{input_dir}/wass_distances_empirical_meanNW_iterations.pkl"), "rb") as f:
        wass_distances_empirical_meanNW_iterations = pickle.load(f)

    wass_distances_stats = {}

    for T in times_T:
        times_t = times_t_dict[T]  
        for t in times_t:
            distances_all_iterations = []

            for iteration in range(iterations):
                iteration_wass_distances = wass_distances_empirical_meanNW_iterations[f"Iteration:{iteration + 1}"]
                distances_all_iterations.append(iteration_wass_distances[f"t:{t}_T:{T}"])

            if distances_all_iterations:
                distances_array = np.array(distances_all_iterations)
                mean_distance = distances_array.mean()
                std_distance = distances_array.std()
            else:
                mean_distance = None
                std_distance = None

            wass_distances_stats[f"t:{t}_T:{T}"] = {
                "mean": mean_distance,
                "std": std_distance
            }

    # Print the calculated mean and standard deviation for verification

    for key, stats in wass_distances_stats.items():
        mean = stats['mean']
        std = stats['std']

    # Save the new stats to a pickle file in the specified folder
    with open(f"{output_dir}/wass_distances_stats.pkl", "wb") as f:
        pickle.dump(wass_distances_stats, f)

    return wass_distances_stats

import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(times_t_dict, times_T, n_replications, iterations, process, wass_distances_stats, time_kernel, space_kernel, input_dir):
    # Ensure seaborn and matplotlib styles
    plt.rcParams['text.usetex'] = False
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 4))

    # Define colors, markers, and linestyles
    colorlist = ["cobalt blue", "jade green", "neon pink", "golden yellow"]
    colors = sns.xkcd_palette(colorlist)
    markert = ['o', 'p', 's', 'd', 'h']
    linestylev = ['-', '--', ':', '-']

    # Loop through the provided time intervals
    for i, T in enumerate(times_T):
        times_t = times_t_dict[T]
        mean_distances = []
        std_distances = []
        filtered_times_t = []  #

        for t in times_t:
            key = f"t:{t}_T:{T}"
            if key in wass_distances_stats:
                mean = wass_distances_stats[key]['mean']
                std = wass_distances_stats[key]['std']

                if mean is not None and std is not None:
                    mean_distances.append(mean)
                    std_distances.append(std)
                    filtered_times_t.append(t)

        # Normalize valid times for plotting
        normalized_times_t = [t / T for t in filtered_times_t]

        # Convert to numpy arrays for consistent processing
        mean_distances = np.array(mean_distances)
        std_distances = np.array(std_distances)

        # Skip if no valid data for this T
        if len(mean_distances) == 0:
            print(f"No valid data for T={T}, skipping.")
            continue

        color = colors[i % len(colors)]
        marker = markert[i % len(markert)]
        linestyle = linestylev[i % len(linestylev)]

        plt.plot(
            normalized_times_t,
            mean_distances,
            lw=2,
            marker=marker,
            markersize=10,
            c=color,
            linestyle=linestyle,
            label=f"T={T}"
        )

        plt.fill_between(
            normalized_times_t,
            mean_distances - std_distances,
            mean_distances + std_distances,
            color=color,
            alpha=0.2,
        )

    plt.xlabel(r'$t/T$', fontsize=16)
    plt.xlim(0.4, 0.6)
    plt.xticks(fontsize = 16)
    plt.yticks(np.linspace(0.007, 0.034, 5), fontsize=16)
    plt.ylabel('Wasserstein Distance', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=15, loc="best")
    plt.tight_layout()
    plt.savefig(f"EW1_tvFAR(2)_timekernel-{time_kernel}_spacekernel-{space_kernel}_rep-{n_replications}_iter-{iterations}.pdf", dpi=300)
    plt.show()



def main():
    if platform.system() == 'Darwin':
        device = torch.device("mps")
    elif platform.system() == 'Linux' or platform.system() == 'Windows':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")

    # Plots
    T = 1000  # Sample size
    J = 7     # Number of basis functions
    N = 100   # Number of spatial discretization points (length of tau)
    tau = np.linspace(0, 1, N)

    # Run the simulation
    curve, X = simulation_covariate(T, N, J, tau)

    # Define the sample size to show in plots
    T_visualize = 100

    # Visualization parameters
    plt.rcParams['text.usetex'] = False
    colormap = 'viridis'  
    alpha_value = 0.8     

    # Visualization of the result (for T_visualize)
    time_visual = np.arange(T_visualize)
    T_mesh, Tau_mesh = np.meshgrid(time_visual, tau, indexing="ij")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_mesh, Tau_mesh, X[:T_visualize, :], cmap=cm.get_cmap(colormap), edgecolor='k', alpha=alpha_value)

    # Figure Title and Labels
    ax.set_xlabel("t", fontsize=16)
    ax.set_ylabel(r"$\tau$", fontsize=16)
    ax.set_zlabel(r"$X_{t,T}(\tau)$", fontsize=16)

    # Colorbar and Style Options
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

    # Plot a few simulated curves (for T_visualize)
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 3))
    for t in range(1, T_visualize + 1, T_visualize // 10):  
        plt.plot(tau, X[t, :], label=f"t={t}")
    plt.xlabel(r"$\tau$", fontsize=16)
    plt.ylabel(r"$X_{t,T}(\tau)$", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.4)
    plt.legend(loc="upper right", fontsize="small")
    plt.show()

    # Plot for 10 sample values of tau (up to T_visualize)
    plt.figure(figsize=(10, 3))
    sampled_tau_indices = np.linspace(0, len(tau) - 1, 10, dtype=int)  # Select 10 tau indices
    for idx in sampled_tau_indices:
        plt.plot(range(T_visualize), X[:T_visualize, idx], label=f"$\\tau={tau[idx]:.2f}$")
    plt.xlabel(r"t", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel(r"$X_{t,T}(\tau)$", fontsize=16)
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(alpha=0.4)
    plt.show()

    # Generate the response variable Y
    Y = generate_1_response(T, N, tau=np.linspace(0, 1, 100), X=X)

    # Plot the response variable Y over time
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 3))
    plt.plot(range(len(Y)), Y, lw=1)
    plt.xlabel("t")
    plt.ylabel("$Y_{t, T}$")
    plt.grid(alpha=0.4)
    plt.show()


    # Setting simulation parameters
    times_T = [500, 1000, 5000, 10000]
    times_t_dict = {
        500: [200, 214, 228, 242, 257, 271, 285, 300],
        1000: [400, 428, 457, 485, 514, 542, 571, 600],
        5000: [2000, 2150, 2280, 2420, 2560, 2710, 2860, 3000],
        10000: [4000, 4290, 4570, 4850, 5130, 5420, 5710, 6000],
    }
    iterations = 50 # Change to 50
    n_replications = 500 # Change to 500

    # Data generation
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    Y_t, Y_replications, X_dict = simulation_L_reps_process(times_T, times_t_dict, iterations, n_replications, N, J, tau, output_dir)

    # Bandwidth selection using CV
    input_dir = "simulation_results"
    process = 'GaussiantvFAR(2)'
    space_kernel = "gaussian" 
    time_kernel = "tricube"
    h_values = np.linspace(0.01, 0.99, 100)
    best_h_dict = cvbandwidth(h_values, times_T, X_dict, Y_replications, iterations, n_replications, space_kernel, time_kernel, input_dir)

    ### Weights calculation
    input_dir = "simulation_results"
    output_dir = "gaussian_weights_output"
    os.makedirs(output_dir, exist_ok=True)

    h_values = np.linspace(0.01, 0.99, 100)
    gaussian_weights = computation_weights(best_h_dict, times_t_dict, times_T, iterations, n_replications, X_dict, 
                        time_kernel, space_kernel, input_dir, output_dir, device)
    
    ### Empirical CDF calculation
    input_dir = "simulation_results"
    empirical_cdfs_iterations = empirical_cdf(times_t_dict, times_T, device, iterations, input_dir, Y_t)
    
    ### Wasserstein distances
    input_dir = "simulation_results"
    input_dir_weights = "gaussian_weights_output"
    output_dir = "tvFAR2_Wass"
    os.makedirs(output_dir, exist_ok=True)
    wass_distances_empirical_meanNW_iterations = wasserstein_distances(times_t_dict, times_T, n_replications, iterations, gaussian_weights, 
                                                                       empirical_cdfs_iterations, Y_replications, input_dir, input_dir_weights, 
                                                                       output_dir, device, pplot=None)
    
    ### Mean Wasserstein distance
    input_dir = "tvFAR2_Wass"
    output_dir = "tvFAR2WassStats"
    os.makedirs(output_dir, exist_ok=True)
    wass_distances_stats = wass_stats(input_dir, output_dir, times_T, times_t_dict, iterations, wass_distances_empirical_meanNW_iterations)
    wass_distances_stats

    ### Plot of results
    input_dir = "tvFAR2WassStats"
    plot_results(times_t_dict, times_T, n_replications, iterations, process, wass_distances_stats, time_kernel, space_kernel, input_dir)

if __name__ == '__main__':
    main()