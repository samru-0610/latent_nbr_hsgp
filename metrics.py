'''
For now, the "outputs" from one model's fit.samples will be used for metrics.py
1. z's discriminability across tissue - plot and score
2. SNR for each marker
3. variances at each layer. Need to change fit.py to return
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import math
from matplotlib.lines import Line2D


def z_discriminability(outputs, labels, savepath=None):
    '''
    plots z1,z2 for each cell (mean over posterior samples) and colors by tissue type. 
        the more discriminate, the encoder is learning a more seperate representation
    returns the silhouette score
    outputs - prior, posterior_predictive samples
    labels - tissue labels
    '''
    
    z = np.asarray(outputs["posterior_predictive"]["z"])
    labels = np.asarray(labels)

    z_mean = z.mean(axis=0)  # mean over samples
    
    score = silhouette_score(z_mean, labels)

    regions = pd.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(regions)))

    plt.figure(figsize=(6, 6))

    for region, color in zip(regions, colors):
        mask = labels == region
        plt.scatter(
            z_mean[mask, 0],
            z_mean[mask, 1],
            s=5,
            alpha=0.4,
            label=region,
            color=color,
        )

    plt.title(f"Posterior mean z by tissue | silhouette={score:.3f}")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return score
    


def snr(outputs, markers, savepath=None, eps=1e-8):
    '''
    ratio of alpha/sigma for each marker, average over posterior samples
    Alpha is the amplitude of variation over the different cells for each marker. 

    The more alpha is for a marker than the sigma, implies the way alpha was informed (z->HSGP) 
    captured the amplitude of variation more than sigma which captures the unexplained noise/ variation


    The function makes a dataframe and returns [marker, alpha, noise, snr] for each marker, 
    and also plots snr per marker.
    '''
    alpha = np.asarray(outputs["posterior"]["alpha"])   # (S, M)
    sigma = np.asarray(outputs["posterior"]["sigma"])   # (S, M)

    markers = np.asarray(markers)

    alpha_sample_mean = alpha.mean(axis=0)   # (M,)
    sigma_sample_mean = sigma.mean(axis=0)   # (M,)

    snr_ratio = alpha_sample_mean / (sigma_sample_mean + eps)

    df_snr = pd.DataFrame({
        "marker": markers,
        "signal_alpha": alpha_sample_mean,
        "noise_sigma": sigma_sample_mean,
        "snr": snr_ratio,
    }).sort_values("snr", ascending=False)

    order = df_snr["marker"].values
    x = np.arange(len(order))

    plt.figure(figsize=(10, 4))
    plt.plot(x, df_snr["signal_alpha"], marker="o", label="GP amplitude alpha")
    plt.plot(x, df_snr["noise_sigma"], marker="o", label="Noise sigma")
    plt.xticks(x, order, rotation=90)
    plt.ylabel("Scale")
    plt.title("GP Signal Amplitude vs Noise per Marker")
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return df_snr


    
def prior_posterior(outputs, markers, savepath=None, bins=30, seed=None):
    '''
    used in two settings, global: only changes for markers(alpha, sigma, intercept) , 
                          cell: changes for each cell (f, loc, obs)

    The point of these plots is to see how much shifting happened between the prior and the posterior. 
    Ideally, 
    
    1. no significant dotted lines
    2. global - alpha > sigma (on x axis)
    3. cell - 

    
    '''
    # per marker
    if seed is not None:
        np.random.seed(seed)

    markers = np.asarray(markers)

    prior = outputs["prior"]
    prior_predictive = outputs["prior_predictive"]
    posterior = outputs["posterior"]
    posterior_predictive = outputs["posterior_predictive"]
    
    M = len(markers)
    ncols = 4
    nrows = math.ceil(M / ncols)

    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes1 = axes1.flatten()

    params1 = ["alpha", "intercept", "sigma"]
    # alpha should have the shape of S,num_markers
    colors1 = {"alpha": "orange", "intercept": "green", "sigma": "blue"}

    for m in range(M):
        ax = axes1[m]

        for p in params1:
            pr = np.asarray(prior[p])[:, m]
            po = np.asarray(posterior[p])[:, m]

            ax.hist(pr, bins=bins, density=True, histtype="step",
                    linestyle="--", color=colors1[p], linewidth=1.5)

            ax.hist(po, bins=bins, density=True, histtype="step",
                    linestyle="-", color=colors1[p], linewidth=1.5)

        ax.set_title(str(markers[m]), fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

    for i in range(M, len(axes1)):
        fig1.delaxes(axes1[i])

    legend1 = [
        Line2D([0], [0], color="black", linestyle="--", label="prior"),
        Line2D([0], [0], color="black", linestyle="-", label="posterior"),
        Line2D([0], [0], color="orange", label="alpha"),
        Line2D([0], [0], color="green", label="intercept"),
        Line2D([0], [0], color="blue", label="sigma"),
    ]

    fig1.legend(handles=legend1, loc="upper right")
    fig1.suptitle("Alpha, Intercept, Sigma: Prior vs Posterior", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    n_cells = posterior_predictive["loc"].shape[1]
    cell_id = np.random.randint(0, n_cells)

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes2 = axes2.flatten()

    # per cell per marker 
    params2 = ["loc", "f", "obs"]
    colors2 = {"loc": "orange", "f": "red", "obs": "blue"}

    for m in range(M):
        ax = axes2[m]

        for p in params2:
            pr = np.asarray(prior_predictive[p])[:, cell_id, m]
            po = np.asarray(posterior_predictive[p])[:, cell_id, m]

            ax.hist(pr, bins=bins, density=True, histtype="step",
                    linestyle="--", color=colors2[p], linewidth=1.5)

            ax.hist(po, bins=bins, density=True, histtype="step",
                    linestyle="-", color=colors2[p], linewidth=1.5)

        ax.set_title(f"{markers[m]} | cell {cell_id}", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

    for i in range(M, len(axes2)):
        fig2.delaxes(axes2[i])

    legend2 = [
        Line2D([0], [0], color="black", linestyle="--", label="prior"),
        Line2D([0], [0], color="black", linestyle="-", label="posterior"),
        Line2D([0], [0], color="orange", label="loc"),
        Line2D([0], [0], color="red", label="f"),
        Line2D([0], [0], color="blue", label="obs"),
    ]

    fig2.legend(handles=legend2, loc="upper right")
    fig2.suptitle(f"loc, f, obs: Prior vs Posterior (cell {cell_id})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    if savepath is not None:
        fig1.savefig(savepath + "_global.png", dpi=200, bbox_inches="tight")
        fig2.savefig(savepath + "_cell.png", dpi=200, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()



def signal_propagation(outputs, hidden_dims, X, nonlin=np.tanh, eps=1e-8):
    posterior = outputs["posterior"]
    z_true = np.asarray(outputs["posterior_predictive"]["z"])
    X = np.asarray(X)

    S = np.asarray(posterior["beta"]).shape[0]

    activations = {}
    rows = []

    current = np.broadcast_to(X[None, :, :], (S, X.shape[0], X.shape[1]))
    x_var = X.var(axis=0).mean()

    rows.append({
        "stage": "X",
        "mean_variance_across_cells": x_var,
        "variance_ratio_vs_input": 1.0,
        "variance_ratio_vs_previous": np.nan,
    })

    prev_var = x_var

    # -------------------------
    # Linear encoder case
    # -------------------------
    if len(hidden_dims) == 0:
        w_linear = np.asarray(posterior["w_linear"])  # (S, input_dim, 2)

        z = np.matmul(current, w_linear)  # (S, N, 2)

        if "b_linear" in posterior:
            b_linear = np.asarray(posterior["b_linear"])  # (S, 2)
            z = z + b_linear[:, None, :]
            stage_name = "z_pre = XW_linear + b_linear"
        else:
            stage_name = "z_pre = XW_linear"

        activations["z_pre"] = z

        z_pre_var = z.var(axis=1).mean()

        rows.append({
            "stage": stage_name,
            "mean_variance_across_cells": z_pre_var,
            "variance_ratio_vs_input": z_pre_var / (x_var + eps),
            "variance_ratio_vs_previous": z_pre_var / (prev_var + eps),
        })

    # -------------------------
    # Nonlinear encoder case
    # -------------------------
    else:
        for i in range(len(hidden_dims)):
            layer = i + 1

            w = np.asarray(posterior[f"w_{layer}"])
            h_pre = np.matmul(current, w)

            if f"b_{layer}" in posterior:
                b = np.asarray(posterior[f"b_{layer}"])
                h_pre = h_pre + b[:, None, :]
                pre_stage_name = f"h{layer}_pre = h{layer-1}W{layer} + b{layer}"
            else:
                pre_stage_name = f"h{layer}_pre = h{layer-1}W{layer}"

            h = nonlin(h_pre)

            activations[f"h{layer}_pre"] = h_pre
            activations[f"h{layer}"] = h

            h_pre_var = h_pre.var(axis=1).mean()
            h_var = h.var(axis=1).mean()

            rows.append({
                "stage": pre_stage_name,
                "mean_variance_across_cells": h_pre_var,
                "variance_ratio_vs_input": h_pre_var / (x_var + eps),
                "variance_ratio_vs_previous": h_pre_var / (prev_var + eps),
            })

            rows.append({
                "stage": f"h{layer} = nonlin(h{layer}_pre)",
                "mean_variance_across_cells": h_var,
                "variance_ratio_vs_input": h_var / (x_var + eps),
                "variance_ratio_vs_previous": h_var / (h_pre_var + eps),
            })

            current = h
            prev_var = h_var

        w_final = np.asarray(posterior["w_final"])
        z = np.matmul(current, w_final)

    # same z standardization as encoder
    z = (z - z.mean(axis=1, keepdims=True)) / (
        z.std(axis=1, keepdims=True) + eps
    )

    activations["z"] = z

    if not np.allclose(z, z_true, atol=1e-5):
        print("Warning: recomputed z is not exactly same as posterior_predictive['z']")
        print("max abs diff:", np.max(np.abs(z - z_true)))

    z_var = z.var(axis=1).mean()

    rows.append({
        "stage": "z = standardized(z_pre)",
        "mean_variance_across_cells": z_var,
        "variance_ratio_vs_input": z_var / (x_var + eps),
        "variance_ratio_vs_previous": z_var / (prev_var + eps),
    })

    df = pd.DataFrame(rows)
    # print(df)

    return df, activations    
    