import jax.numpy as jnp
import numpyro
import numpy as np
import numpyro.distributions as dist

PI = jnp.pi

def _sqrt_spd_se_2d(alpha, rho1, rho2, w1, w2):
    return alpha * jnp.sqrt(2.0 * PI * rho1 * rho2) * jnp.exp(
        -0.25 * (rho1**2 * w1**2 + rho2**2 * w2**2)
    )

def nonlin(x):
    return jnp.tanh(x)
    
def LinearEncoder(X, wf=0.1, bias=False):
    
    N, in_dim = X.shape
    W = numpyro.sample(
        "w_linear",
        dist.Normal(
            jnp.zeros((in_dim, 2)),
            (wf/jnp.sqrt(in_dim))*jnp.ones((in_dim, 2)),
        ).to_event(2),
    )
    if bias:
        b = numpyro.sample(
            "b_linear",
            dist.Normal(jnp.zeros((2,)), jnp.ones((2,))).to_event(1),
        )
        z = jnp.matmul(X, W) + b
    else:
        z = jnp.matmul(X, W)
    assert z.shape == (N, 2)
    eps = 1e-8
    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + eps)   
    
    return z
    
def BNNEncoder(X, hidden_dims, wf = 0.1, bias = False):
        
    N = X.shape[0]
    dims = [X.shape[1]] + list(hidden_dims)
    z = X 
    for i in range(len(dims) - 1):
        w = numpyro.sample(
            f"w_{i+1}",
            dist.Normal(
                jnp.zeros((dims[i], dims[i + 1])),
                (wf/jnp.sqrt(dims[i]))*jnp.ones((dims[i], dims[i + 1])),
            ).to_event(2),
        )
        if bias == True:
            b = numpyro.sample(
                f"b_{i+1}",
                dist.Normal(
                    jnp.zeros((dims[i + 1],)),
                    jnp.ones((dims[
                        i + 1],)),
                ).to_event(1),
            )
            z = nonlin(jnp.matmul(z, w) + b)
        else:
            z = nonlin(jnp.matmul(z,w))

    w_final = numpyro.sample("w_final",
            dist.Normal(
                jnp.zeros((dims[-1], 2)),
                (wf/jnp.sqrt(dims[-1]))*jnp.ones((dims[-1], 2)),
            ).to_event(2),
        )
    
    z = jnp.matmul(z,w_final)
    assert z.shape == (N, 2)
    eps = 1e-8
    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + eps)    
    
    return z

def hsgp_2d_linear_encoder(
    neighbourhood,
    D,
    wf, 
    bias,
    m1,
    m2,
    marginal_sd_prior_shape,
    noise_sd_prior_shape,
    intercept_prior_shape,
    rho1,
    rho2,
    gene_expression=None,
):
    N = neighbourhood.shape[0]
    m_total = m1 * m2

    z = LinearEncoder(neighbourhood, wf=wf, bias=bias)
    L = 6

    z1 = z[:, 0]
    z2 = z[:, 1]

    alpha = numpyro.sample(
        "alpha",
        dist.TruncatedNormal(
            marginal_sd_prior_shape[0],
            marginal_sd_prior_shape[1],
            low=0,
            high=1.1,
        ).expand([D]).to_event(1),
    )

    sigma = numpyro.sample(
        "sigma",
        dist.TruncatedNormal(
            noise_sd_prior_shape[0],
            noise_sd_prior_shape[1],
            low=0,
        ).expand([D]).to_event(1),
    )

    intercept = numpyro.sample(
        "intercept",
        dist.TruncatedNormal(
            intercept_prior_shape[0],
            intercept_prior_shape[1],
            low=0,
            high=1.1,
        ).expand([D]).to_event(1),
    )

    beta = numpyro.sample(
        "beta",
        dist.Normal(0.0, 1.0).expand([D, m_total]).to_event(2),
    )

    m1_vec = jnp.arange(1, m1 + 1)
    m2_vec = jnp.arange(1, m2 + 1)

    factor1 = (m1_vec * PI) / (2.0 * L)
    factor2 = (m2_vec * PI) / (2.0 * L)

    phi_z1 = (1.0 / jnp.sqrt(L)) * jnp.sin((z1[:, None] + L) * factor1[None, :])
    phi_z2 = (1.0 / jnp.sqrt(L)) * jnp.sin((z2[:, None] + L) * factor2[None, :])

    PHI_2d = (phi_z1[:, :, None] * phi_z2[:, None, :]).reshape(N, m_total)

    w1_grid = jnp.repeat(factor1, m2)
    w2_grid = jnp.tile(factor2, m1)

    sqrt_spectral_density = _sqrt_spd_se_2d(
        alpha[:, None], rho1, rho2, w1_grid[None, :], w2_grid[None, :]
    )

    f = PHI_2d @ (sqrt_spectral_density * beta).T
    loc = intercept[None, :] + f

    with numpyro.plate("cells", N):
        numpyro.sample(
            "obs",
            dist.Normal(loc, sigma[None, :]).to_event(1),
            obs=gene_expression,
        )

    numpyro.deterministic("z", z)
    numpyro.deterministic("f", f)
    numpyro.deterministic("loc", loc)

    
def hsgp_2d(
    neighbourhood,
    D,
    wf, 
    hidden_dims,
    bias,
    m1,
    m2,
    marginal_sd_prior_shape,
    noise_sd_prior_shape,
    intercept_prior_shape,
    rho1,
    rho2,
    gene_expression=None,
):
    N = neighbourhood.shape[0]
    in_dim_bnn = neighbourhood.shape[1]
    m_total = m1 * m2

    z = BNNEncoder(neighbourhood, wf=wf, hidden_dims=hidden_dims, bias = bias)
    L = 6
    
    z1 = z[:, 0]
    z2 = z[:, 1]

    alpha = numpyro.sample(
        "alpha",
        dist.TruncatedNormal(
            marginal_sd_prior_shape[0],
            marginal_sd_prior_shape[1],
            low=0,
            high=1.1,
        ).expand([D]).to_event(1),
    )

    sigma = numpyro.sample(
        "sigma",
        dist.TruncatedNormal(
            noise_sd_prior_shape[0],
            noise_sd_prior_shape[1],
            low=0,
        ).expand([D]).to_event(1),
    )

    intercept = numpyro.sample(
        "intercept",
        dist.TruncatedNormal(
            intercept_prior_shape[0],
            intercept_prior_shape[1],
            low=0,
            high=1.1,
        ).expand([D]).to_event(1),
    )

    beta = numpyro.sample(
        "beta",
        dist.Normal(0.0, 1.0).expand([D, m_total]).to_event(2),
    )

    m1_vec = jnp.arange(1, m1 + 1)
    m2_vec = jnp.arange(1, m2 + 1)

    factor1 = (m1_vec * PI) / (2.0 * L)
    factor2 = (m2_vec * PI) / (2.0 * L)
    
    phi_z1 = (1.0 / jnp.sqrt(L)) * jnp.sin((z1[:, None] + L) * factor1[None, :])
    phi_z2 = (1.0 / jnp.sqrt(L)) * jnp.sin((z2[:, None] + L) * factor2[None, :])

    PHI_2d = (phi_z1[:, :, None] * phi_z2[:, None, :]).reshape(N, m_total)

    w1_grid = jnp.repeat(factor1, m2)
    w2_grid = jnp.tile(factor2, m1)

    sqrt_spectral_density = _sqrt_spd_se_2d(
        alpha[:, None], rho1, rho2, w1_grid[None, :], w2_grid[None, :]
    )

    sqrt_spectral_density_beta = sqrt_spectral_density * beta
    f = PHI_2d @ sqrt_spectral_density_beta.T
    loc = intercept[None, :] + f

    with numpyro.plate("cells", N):
        numpyro.sample(
            "obs",
            dist.Normal(loc, sigma[None, :]).to_event(1),
            obs=gene_expression,
        )
    numpyro.deterministic("z", z)
    numpyro.deterministic("f", f)
    numpyro.deterministic("loc", loc)
    