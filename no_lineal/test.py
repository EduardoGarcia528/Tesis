import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import mi_libreria as ml

def standardize(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (x - np.mean(x)) / np.std(x)


def compute_measures(x):
    """
    Calcula PE, S_theta y gamma_R para una serie.
    """
    x = standardize(x)

    PE = ml.permutation_entropy(x, m=5, tau=1)

    S_theta = ml.indice_S_eff_fast(x, tau=1)

    C, g = ml.gamma_index_rank(x, max_gamma=1, mu=2)
    gamma_R = g[0]

    return {
        "PE": PE,
        "S_theta": S_theta,
        "gamma_R": gamma_R,
    }

def iaaft_surrogate(x, n_iter=500, tol=1e-8, rng=None):
    """
    Sustituto IAAFT de una serie real x.
    
    Conserva aproximadamente:
    - distribución marginal
    - espectro de potencia
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    N = len(x)

    x_sorted = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))

    # inicialización: permutación aleatoria
    y = rng.permutation(x)

    old_error = np.inf

    for _ in range(n_iter):
        # imponer espectro
        Y = np.fft.rfft(y)
        phases = np.exp(1j * np.angle(Y))
        y_spec = np.fft.irfft(target_amp * phases, n=N)

        # imponer distribución marginal por rangos
        ranks = np.argsort(np.argsort(y_spec))
        y_new = x_sorted[ranks]

        # criterio de convergencia sobre espectro
        new_amp = np.abs(np.fft.rfft(y_new))
        error = np.mean((new_amp - target_amp)**2) / np.mean(target_amp**2)

        y = y_new

        if abs(old_error - error) < tol:
            break

        old_error = error

    return y

def null_test_iaaft(
    x,
    n_surr=100,
    n_iter_iaaft=500,
    seed=None
):
    """
    Calcula medidas observadas, distribución nula IAAFT y Z-scores.
    """
    rng = np.random.default_rng(seed)

    x = standardize(x)

    obs = compute_measures(x)

    null_values = {
        "PE": [],
        "S_theta": [],
        "gamma_R": [],
    }
    xs = ml.iaaft(x, ns = n_surr)
    for j in range(n_surr):
        # xs = iaaft_surrogate(
        #     x,
        #     n_iter=n_iter_iaaft,
        #     rng=rng
        # )

        ms = compute_measures(xs[j])

        for key in null_values:
            null_values[key].append(ms[key])

    rows = []

    for key in null_values:
        null_arr = np.asarray(null_values[key], dtype=float)

        mu_null = np.mean(null_arr)
        sigma_null = np.std(null_arr, ddof=1)

        if sigma_null == 0:
            Z = np.nan
        else:
            Z = (obs[key] - mu_null) / sigma_null

        rows.append({
            "measure": key,
            "obs": obs[key],
            "mu_null": mu_null,
            "sigma_null": sigma_null,
            "Z": Z,
            "absZ": abs(Z),
        })

    return pd.DataFrame(rows)

def simulate_nonlinear_ar1(
    N=4000,
    transient=1000,
    a=0.55,
    lam=0.0,
    sigma=1.0,
    seed=None
):
    """
    x_{t+1} = a x_t + lam sin(x_t) + sigma eta_t

    lam = 0 corresponde a AR(1) lineal.
    lam > 0 introduce no linealidad dinámica.
    """
    rng = np.random.default_rng(seed)

    total = N + transient
    x = np.zeros(total)

    x[0] = rng.normal()

    for t in range(total - 1):
        x[t+1] = a*x[t] + lam*np.sin(x[t]) + sigma*rng.normal()

    return standardize(x[transient:])

def experiment_nonlinearity_sweep(
    lambdas=np.linspace(0, 1.5, 11),
    N=4000,
    n_reps=20,
    n_surr=100,
    n_iter_iaaft=500,
    seed=123
):
    rng = np.random.default_rng(seed)

    all_rows = []

    for lam in tqdm(lambdas, desc="Barrido lambda"):
        for rep in range(n_reps):
            serie_seed = rng.integers(0, 2**32 - 1)

            x = simulate_nonlinear_ar1(
                N=N,
                lam=lam,
                seed=serie_seed
            )

            df = null_test_iaaft(
                x,
                n_surr=n_surr,
                n_iter_iaaft=n_iter_iaaft,
                seed=rng.integers(0, 2**32 - 1)
            )

            df["experiment"] = "nonlinear_ar1_sweep"
            df["lambda"] = lam
            df["rep"] = rep
            df["N"] = N

            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)

lambdas = np.linspace(0, 1.5, 11)

df_lambda = experiment_nonlinearity_sweep(
    lambdas=lambdas,
    N=4000,
    n_reps=20,
    n_surr=100,
    n_iter_iaaft=500,
    seed=123
)

df_lambda.to_csv("sensibilidad_lambda_IAAFT.csv", index=False)
df_lambda.head()

"""
Experimento B:
"""

def simulate_ar1(
    N=4000,
    transient=1000,
    a=0.75,
    sigma=1.0,
    seed=None
):
    rng = np.random.default_rng(seed)

    total = N + transient
    x = np.zeros(total)
    x[0] = rng.normal()

    for t in range(total - 1):
        x[t+1] = a*x[t] + sigma*rng.normal()

    return standardize(x[transient:])

def experiment_static_transforms(
    N=4000,
    n_reps=20,
    n_surr=100,
    n_iter_iaaft=500,
    seed=456
):
    rng = np.random.default_rng(seed)

    transforms = {
        "linear_AR1_original": lambda x: x,
        "static_square": lambda x: x**2,
        "static_exp": lambda x: np.exp(0.5*x),
        "static_tanh": lambda x: np.tanh(x),
    }

    all_rows = []

    for rep in tqdm(range(n_reps), desc="Transformaciones estáticas"):
        x = simulate_ar1(
            N=N,
            seed=rng.integers(0, 2**32 - 1)
        )

        for name, f in transforms.items():
            y = standardize(f(x))

            df = null_test_iaaft(
                y,
                n_surr=n_surr,
                n_iter_iaaft=n_iter_iaaft,
                seed=rng.integers(0, 2**32 - 1)
            )

            df["experiment"] = "static_transform_control"
            df["transform"] = name
            df["rep"] = rep
            df["N"] = N

            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)

df_static = experiment_static_transforms(
    N=4000,
    n_reps=20,
    n_surr=100,
    n_iter_iaaft=500,
    seed=456
)

df_static.to_csv("control_transformaciones_estaticas_IAAFT.csv", index=False)
df_static.head()

def simulate_mixed_linear_nonlinear(
    N=4000,
    transient=1000,
    a=0.55,
    b=1.5,
    epsilon=0.0,
    sigma=1.0,
    seed=None
):
    """
    Mezcla:
    
    x_{t+1} = (1-eps) L(x_t) + eps F(x_t) + sigma eta_t

    L(x) = a x
    F(x) = a x + b sin(x)

    equivalente a:
    x_{t+1} = a x_t + eps*b*sin(x_t) + sigma eta_t
    """
    rng = np.random.default_rng(seed)

    total = N + transient
    x = np.zeros(total)
    x[0] = rng.normal()

    for t in range(total - 1):
        L = a*x[t]
        F = a*x[t] + b*np.sin(x[t])

        x[t+1] = (1 - epsilon)*L + epsilon*F + sigma*rng.normal()

    return standardize(x[transient:])


def experiment_mixture_sweep(
    epsilons=np.linspace(0, 1, 11),
    N=4000,
    n_reps=20,
    n_surr=100,
    n_iter_iaaft=500,
    seed=789
):
    rng = np.random.default_rng(seed)

    all_rows = []

    for eps in tqdm(epsilons, desc="Barrido epsilon"):
        for rep in range(n_reps):
            x = simulate_mixed_linear_nonlinear(
                N=N,
                epsilon=eps,
                seed=rng.integers(0, 2**32 - 1)
            )

            df = null_test_iaaft(
                x,
                n_surr=n_surr,
                n_iter_iaaft=n_iter_iaaft,
                seed=rng.integers(0, 2**32 - 1)
            )

            df["experiment"] = "mixed_linear_nonlinear"
            df["epsilon"] = eps
            df["rep"] = rep
            df["N"] = N

            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)

epsilons = np.linspace(0, 1, 11)

df_mix = experiment_mixture_sweep(
    epsilons=epsilons,
    N=4000,
    n_reps=20,
    n_surr=100,
    n_iter_iaaft=500,
    seed=789
)

df_mix.to_csv("sensibilidad_mezcla_lineal_nolineal_IAAFT.csv", index=False)
df_mix.head()


def plot_z_vs_parameter(
    df,
    parameter,
    title=None,
    use_abs=True
):
    ycol = "absZ" if use_abs else "Z"

    summary = (
        df
        .groupby(["measure", parameter])[ycol]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(7, 5))

    for measure in summary["measure"].unique():
        sub = summary[summary["measure"] == measure]

        plt.errorbar(
            sub[parameter],
            sub["mean"],
            yerr=sub["std"],
            marker="o",
            capsize=3,
            label=measure
        )

    plt.axhline(5, linestyle="--", color="k", alpha=0.5, label="|Z| = 5")

    plt.xlabel(parameter)
    plt.ylabel("|Z|" if use_abs else "Z")
    plt.legend()

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()


plot_z_vs_parameter(
    df_lambda,
    parameter="lambda",
    title="Sensibilidad a no linealidad dinámica",
    use_abs=True
)

plot_z_vs_parameter(
    df_mix,
    parameter="epsilon",
    title="Mezcla lineal-no lineal",
    use_abs=True
)



def plot_static_transform_control(df_static, use_abs=True):
    ycol = "absZ" if use_abs else "Z"

    summary = (
        df_static
        .groupby(["measure", "transform"])[ycol]
        .agg(["mean", "std"])
        .reset_index()
    )

    measures = summary["measure"].unique()

    for measure in measures:
        sub = summary[summary["measure"] == measure]

        plt.figure(figsize=(7, 4))

        plt.bar(
            sub["transform"],
            sub["mean"],
            yerr=sub["std"],
            capsize=4
        )

        plt.axhline(5, linestyle="--", color="k", alpha=0.5)

        plt.ylabel("|Z|" if use_abs else "Z")
        plt.title(f"Control estático: {measure}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()


plot_static_transform_control(df_static, use_abs=True)


def detection_threshold(df, parameter, z_threshold=5):
    summary = (
        df
        .groupby(["measure", parameter])["absZ"]
        .mean()
        .reset_index()
    )

    rows = []

    for measure in summary["measure"].unique():
        sub = summary[summary["measure"] == measure].sort_values(parameter)

        detected = sub[sub["absZ"] > z_threshold]

        if len(detected) == 0:
            threshold = np.nan
        else:
            threshold = detected[parameter].iloc[0]

        rows.append({
            "measure": measure,
            "parameter": parameter,
            "threshold": threshold,
            "z_threshold": z_threshold
        })

    return pd.DataFrame(rows)


threshold_lambda = detection_threshold(df_lambda, "lambda", z_threshold=5)
threshold_mix = detection_threshold(df_mix, "epsilon", z_threshold=5)

print(threshold_lambda)
print(threshold_mix)

