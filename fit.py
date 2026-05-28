'''
fits the BNN + HSGP and also samples the priors and posteriors. 
usage- fit.fit; fit.plot_ELBO; fit.samples
(ofc along with corresponding parameters)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import numpyro.optim as optim
import time
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float, Int
from numpyro.contrib.module import random_nnx_module
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.svi import SVIRunResult, SVIState
from arviz_plots import plot_lm, style
import importlib
import model_functions2
importlib.reload(model_functions2)
from numpyro.infer import Predictive


def get_model(hidden_dims):
    if len(hidden_dims) == 0:
        return model_functions2.hsgp_2d_linear_encoder
    else:
        return model_functions2.hsgp_2d


# given X,y and all h-params, function returns svi results and guide post fitting

def fit(
    neighbourhood,
    gene_expression, 
    wf = 0.1,
    hidden_dims = [10], # if [], its linear.
    
    m1 = 14,
    m2 = 14,
    n_steps = 1000,
    lr = 0.001,
    seed = 0,
    bias = False,
):
    D = gene_expression.shape[1]
    model = get_model(hidden_dims)
    guide = AutoNormal(model)

    svi = SVI(
        model,
        guide,
        optim.Adam(lr),
        loss=Trace_ELBO(num_particles=4),
        # 4 particles averages over over 4 samples from the posterior and guide
    )
    
    static_kwargs = dict(
        D=D,
        m1=m1,
        m2=m2,
        wf=wf,
        bias=bias, 
        marginal_sd_prior_shape=(0.5, 0.2),
        noise_sd_prior_shape=(0.5, 0.1), 
        intercept_prior_shape=(0.5, 0.2),
        rho1=0.75,
        rho2=0.75,
    )

    if len(hidden_dims) > 0:
        static_kwargs["hidden_dims"] = hidden_dims

    svi_result = svi.run(
        jax.random.PRNGKey(seed),
        num_steps=n_steps,
        neighbourhood=neighbourhood,
        gene_expression=gene_expression,
        **static_kwargs,
    )

    return guide, svi, svi_result, static_kwargs

def plot_ELBO(svi_result, static_kwargs, savepath = None):
    plt.figure(figsize=(8, 3))
    plt.plot(svi_result.losses)
    plt.xlabel("Step")
    plt.ylabel("ELBO loss")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    plt.show()


def make_return_sites(hidden_dims, bias=False):
    parameter_sites = ["alpha", "sigma", "intercept", "beta"]

    if len(hidden_dims) == 0:
        parameter_sites.append("w_linear")
        if bias:
            parameter_sites.append("b_linear")
    else:
        for i in range(len(hidden_dims)):
            parameter_sites.append(f"w_{i+1}")
            if bias:
                parameter_sites.append(f"b_{i+1}")
        parameter_sites.append("w_final")
        if bias:
            parameter_sites.append("b_final")   # only if your model has this

    predictive_sites = ["z", "f", "loc", "obs"]
    return parameter_sites, predictive_sites

    
def samples(
    n_samples,
    model_kwargs,
    guide,
    svi_results,
    sample_what=("prior", "prior_predictive", "posterior", "posterior_predictive"),
    seed=0,
):
    sample_what = set(sample_what)

    # valid = {"prior", "prior_predictive", "posterior", "posterior_predictive"}
    # unknown = sample_what - valid
    # if unknown:
    #     raise ValueError(f"Unknown sample_what: {unknown}")

    rng_key = random.PRNGKey(seed)
    rng_key_prior, rng_key_priorpred, rng_key_post, rng_key_postpred = random.split(rng_key, 4)

    hidden_dims = model_kwargs.get("hidden_dims", [])
    bias = model_kwargs.get("bias", False)

    model = get_model(hidden_dims)

    parameter_sites, predictive_sites = make_return_sites(
        hidden_dims=hidden_dims,
        bias=bias,
    )

    outputs = {
        "parameter_sites": parameter_sites,
        "predictive_sites": predictive_sites,
    }

    # prior = parameters sampled from prior
    if "prior" in sample_what:
        prior_param_predictive = Predictive(
            model,
            num_samples=n_samples,
            return_sites=parameter_sites,
        )

        outputs["prior"] = prior_param_predictive(
            rng_key_prior,
            **model_kwargs,
            gene_expression=None,
        )

    # prior_predictive = z, f, loc, obs from prior parameters
    if "prior_predictive" in sample_what:
        prior_predictive = Predictive(
            model,
            num_samples=n_samples,
            return_sites=predictive_sites,
        )

        outputs["prior_predictive"] = prior_predictive(
            rng_key_priorpred,
            **model_kwargs,
            gene_expression=None,
        )

    posterior_samples = None

    # posterior = parameters sampled from guide after SVI
    if "posterior" in sample_what or "posterior_predictive" in sample_what:
        guide_predictive = Predictive(
            guide,
            params=svi_results.params,
            num_samples=n_samples,
        )

        posterior_samples = guide_predictive(
            rng_key_post,
            **model_kwargs,
            gene_expression=None,
        )

        if "posterior" in sample_what:
            outputs["posterior"] = posterior_samples

    # posterior_predictive = z, f, loc, obs from posterior parameters
    if "posterior_predictive" in sample_what:
        model_predictive = Predictive(
            model,
            posterior_samples=posterior_samples,
            return_sites=predictive_sites,
        )

        outputs["posterior_predictive"] = model_predictive(
            rng_key_postpred,
            **model_kwargs,
            gene_expression=None,
        )

    return outputs


    
    

    
        
    
    