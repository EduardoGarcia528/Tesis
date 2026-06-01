# ============================================================
# Experimentos bivariantes de S_eff con mapas logísticos acoplados
# Modelo nulo: Pi Delta theta -> null="shuffle"
# ============================================================

from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mi_libreria as ml


# ============================================================
# 1. Configuración
# ============================================================

@dataclass(frozen=True)
class Config:
    n: int
    burn: int
    n_rep: int
    n_null: int
    eps_grid: tuple
    taus: tuple
    delays: tuple
    rx: float = 4.0
    ry: float = 4.0
    rz: float = 4.0
    seed: int = 20260529


# Corrida inicial para revisar comportamiento general.
PILOT = Config(
    n=12_000,
    burn=3_000,
    n_rep=3,
    n_null=30,
    eps_grid=(0.0, 0.05, 0.15, 0.30, 0.45),
    taus=tuple(range(0, 13)),
    delays=(1, 2, 5, 10),
)

# Corrida más robusta, para ejecutar después de inspeccionar el piloto.
ROBUST = Config(
    n=50_000,
    burn=5_000,
    n_rep=30,
    n_null=200,
    eps_grid=tuple(np.round(np.linspace(0.0, 0.50, 11), 3)),
    taus=tuple(range(0, 16)),
    delays=(1, 2, 5, 10),
)


# ============================================================
# 2. Utilidades
# ============================================================

def logistic_map(x, r=4.0):
    """Mapa logístico."""
    return r * x * (1.0 - x)

def evaluate_driver_response(
    driver,
    response,
    cfg,
    rng_null,
    experiment,
    driver_name,
    response_name,
    repetition,
    simulation_seed,
    epsilon=np.nan,
    delay_true=np.nan,
    expected_tau=np.nan,
    rx=np.nan,
    ry=np.nan,
    extra_label="",
):
    """
    Evalúa dependencia retardada driver -> response.

    Debido a la convención interna del índice, para buscar que
    driver_t antecede a response_{t+tau}, se llama:

        indice_S_eff_fast(response, driver, tau=tau, ...)
    """
    return evaluate_curve(
        x=response,
        y=driver,
        cfg=cfg,
        rng_null=rng_null,
        experiment=experiment,
        direction=f"{driver_name}_drives_{response_name}",
        repetition=repetition,
        simulation_seed=simulation_seed,
        epsilon=epsilon,
        delay_true=delay_true,
        expected_tau=expected_tau,
        rx=rx,
        ry=ry,
        extra_label=extra_label,
    )


def _crop(series, cfg, max_delay):
    """Elimina la historia inicial y el transitorio."""
    start = max_delay + cfg.burn
    end = start + cfg.n
    return series[start:end]


def _as_float(value):
    """Convierte la salida del índice a float escalar."""
    return float(np.asarray(value).squeeze())


# ============================================================
# 3. Simuladores
# ============================================================

def simulate_independent(cfg, rng, rx=None, ry=None):
    """
    Dos mapas logísticos independientes:
        x_{t+1} = f_x(x_t)
        y_{t+1} = f_y(y_t)
    """
    rx = cfg.rx if rx is None else rx
    ry = cfg.ry if ry is None else ry

    total = cfg.burn + cfg.n + 1

    x = np.empty(total)
    y = np.empty(total)

    x[0] = rng.random()
    y[0] = rng.random()

    for t in range(total - 1):
        x[t + 1] = logistic_map(x[t], rx)
        y[t + 1] = logistic_map(y[t], ry)

    return x[cfg.burn: cfg.burn + cfg.n], y[cfg.burn: cfg.burn + cfg.n]


def simulate_unidirectional(cfg, eps, delay, rng, rx=None, ry=None):
    r"""
    Acoplamiento unidireccional x -> y:

        x_{t+1} = f_x(x_t)

        y_{t+1} = (1-eps) f_y(y_t) + eps x_{t+1-delay}

    Si delay=d, y_t contiene información de x_{t-d}.
    """
    rx = cfg.rx if rx is None else rx
    ry = cfg.ry if ry is None else ry

    max_delay = int(delay)
    total = max_delay + cfg.burn + cfg.n + 1

    x = rng.random(total)
    y = rng.random(total)

    for t in range(max_delay, total - 1):
        x[t + 1] = logistic_map(x[t], rx)

        y_free = logistic_map(y[t], ry)
        driver = x[t + 1 - delay]

        y[t + 1] = (1.0 - eps) * y_free + eps * driver

    return _crop(x, cfg, max_delay), _crop(y, cfg, max_delay)


def simulate_common_driver(
    cfg,
    eps,
    delay_x,
    delay_y,
    rng,
    rx=None,
    ry=None,
    rz=None,
):
    r"""
    Fuente común z -> x, z -> y:

        z_{t+1} = f_z(z_t)

        x_{t+1} = (1-eps) f_x(x_t) + eps z_{t+1-delay_x}

        y_{t+1} = (1-eps) f_y(y_t) + eps z_{t+1-delay_y}

    Si delay_y > delay_x, se espera una relación x -> y
    alrededor de tau = delay_y - delay_x.
    """
    rx = cfg.rx if rx is None else rx
    ry = cfg.ry if ry is None else ry
    rz = cfg.rz if rz is None else rz

    max_delay = max(int(delay_x), int(delay_y))
    total = max_delay + cfg.burn + cfg.n + 1

    x = rng.random(total)
    y = rng.random(total)
    z = rng.random(total)

    for t in range(max_delay, total - 1):
        z[t + 1] = logistic_map(z[t], rz)

        x_free = logistic_map(x[t], rx)
        y_free = logistic_map(y[t], ry)

        zx = z[t + 1 - delay_x]
        zy = z[t + 1 - delay_y]

        x[t + 1] = (1.0 - eps) * x_free + eps * zx
        y[t + 1] = (1.0 - eps) * y_free + eps * zy

    return _crop(x, cfg, max_delay), _crop(y, cfg, max_delay)


def simulate_bidirectional(cfg, eps, delay, rng, rx=None, ry=None):
    r"""
    Acoplamiento bidireccional:

    Para delay = 0:

        x_{t+1} = (1-eps) f_x(x_t) + eps f_y(y_t)
        y_{t+1} = (1-eps) f_y(y_t) + eps f_x(x_t)

    Para delay > 0:

        x_{t+1} = (1-eps) f_x(x_t) + eps y_{t+1-delay}
        y_{t+1} = (1-eps) f_y(y_t) + eps x_{t+1-delay}
    """
    rx = cfg.rx if rx is None else rx
    ry = cfg.ry if ry is None else ry

    max_delay = int(delay)
    total = max_delay + cfg.burn + cfg.n + 1

    x = rng.random(total)
    y = rng.random(total)

    for t in range(max_delay, total - 1):
        fx = logistic_map(x[t], rx)
        fy = logistic_map(y[t], ry)

        if delay == 0:
            x[t + 1] = (1.0 - eps) * fx + eps * fy
            y[t + 1] = (1.0 - eps) * fy + eps * fx
        else:
            x[t + 1] = (1.0 - eps) * fx + eps * y[t + 1 - delay]
            y[t + 1] = (1.0 - eps) * fy + eps * x[t + 1 - delay]

    return _crop(x, cfg, max_delay), _crop(y, cfg, max_delay)


# ============================================================
# 4. Cálculo de S_eff y modelo nulo
# ============================================================

def indice_S_observado(x, y, tau):
    """
    S_{Delta theta} observado.
    """
    value = ml.indice_S_eff_fast(
        x,
        y,
        tau=tau,
        delta=True,
    )
    return _as_float(value)


def indice_S_null(x, y, tau):
    r"""
    S_{Delta theta}^{Pi Delta theta}.
    """
    value = ml.indice_S_eff_fast(
        x,
        y,
        tau=tau,
        delta=True,
        null="shuffle",
    )
    return _as_float(value)


def evaluate_tau(x, y, tau, n_null, rng_null):
    """
    Calcula:
        S_obs
        media y desviación estándar del nulo
        DeltaS = S_obs - mu_null
        Z-score
        p-values empíricos laterales
    """
    s_obs = indice_S_observado(x, y, tau)

    null_values = np.empty(n_null, dtype=float)

    for b in range(n_null):
        # Útil si el shuffle interno depende del estado global de numpy.
        np.random.seed(int(rng_null.integers(0, 2**32 - 1)))
        null_values[b] = indice_S_null(x, y, tau)

    mu_null = np.mean(null_values)
    sigma_null = np.std(null_values, ddof=1)

    delta_s = s_obs - mu_null

    if sigma_null > 0:
        z_score = delta_s / sigma_null
    else:
        z_score = np.nan

    p_less = (1 + np.sum(null_values <= s_obs)) / (n_null + 1)
    p_greater = (1 + np.sum(null_values >= s_obs)) / (n_null + 1)
    p_two_sided = min(1.0, 2.0 * min(p_less, p_greater))

    return {
        "tau": int(tau),
        "S_obs": s_obs,
        "mu_null": mu_null,
        "sigma_null": sigma_null,
        "DeltaS": delta_s,
        "Z": z_score,
        "p_less": p_less,
        "p_greater": p_greater,
        "p_two_sided": p_two_sided,
    }


def evaluate_curve(
    x,
    y,
    cfg,
    rng_null,
    experiment,
    direction,
    repetition,
    simulation_seed,
    epsilon=np.nan,
    delay_true=np.nan,
    expected_tau=np.nan,
    rx=np.nan,
    ry=np.nan,
    extra_label="",
):
    """
    Evalúa todos los taus para una realización de un experimento.
    """
    rows = []

    for tau in cfg.taus:
        stats = evaluate_tau(
            x=x,
            y=y,
            tau=tau,
            n_null=cfg.n_null,
            rng_null=rng_null,
        )

        row = {
            "experiment": experiment,
            "direction": direction,
            "repetition": repetition,
            "simulation_seed": simulation_seed,
            "epsilon": epsilon,
            "delay_true": delay_true,
            "expected_tau": expected_tau,
            "rx": rx,
            "ry": ry,
            "extra_label": extra_label,
        }
        row.update(stats)
        rows.append(row)

    return rows


# ============================================================
# 5. Batería completa de experimentos
# ============================================================

def run_all_experiments(cfg, output_dir="resultados_S_bivariante"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng_sim = np.random.default_rng(cfg.seed)
    rng_null = np.random.default_rng(cfg.seed + 10_000)

    rows = []

    # --------------------------------------------------------
    # A. Mapas independientes
    # --------------------------------------------------------
    print("A. Mapas independientes")

    for rep in range(cfg.n_rep):
        sim_seed = int(rng_sim.integers(0, 2**32 - 1))
        rng_local = np.random.default_rng(sim_seed)

        x, y = simulate_independent(cfg, rng_local)

        rows.extend(
            evaluate_curve(
                x=x,
                y=y,
                cfg=cfg,
                rng_null=rng_null,
                experiment="A_independent",
                direction="x_to_y",
                repetition=rep,
                simulation_seed=sim_seed,
                epsilon=0.0,
                delay_true=np.nan,
                expected_tau=np.nan,
                rx=cfg.rx,
                ry=cfg.ry,
            )
        )

    # --------------------------------------------------------
    # B. Acoplamiento unidireccional instantáneo
    # --------------------------------------------------------
    print("B. Acoplamiento unidireccional instantaneo")

    for eps in cfg.eps_grid:
        for rep in range(cfg.n_rep):
            sim_seed = int(rng_sim.integers(0, 2**32 - 1))
            rng_local = np.random.default_rng(sim_seed)

            x, y = simulate_unidirectional(
                cfg=cfg,
                eps=eps,
                delay=0,
                rng=rng_local,
            )

            rows.extend(
                evaluate_curve(
                    x=x,
                    y=y,
                    cfg=cfg,
                    rng_null=rng_null,
                    experiment="B_unidirectional_instant",
                    direction="x_to_y",
                    repetition=rep,
                    simulation_seed=sim_seed,
                    epsilon=eps,
                    delay_true=0,
                    expected_tau=0,
                    rx=cfg.rx,
                    ry=cfg.ry,
                )
            )

    # --------------------------------------------------------
    # C. Acoplamiento unidireccional retardado
    # --------------------------------------------------------
    print("C. Acoplamiento unidireccional retardado")

    for delay in cfg.delays:
        for eps in cfg.eps_grid:
            for rep in range(cfg.n_rep):
                sim_seed = int(rng_sim.integers(0, 2**32 - 1))
                rng_local = np.random.default_rng(sim_seed)

                x, y = simulate_unidirectional(
                    cfg=cfg,
                    eps=eps,
                    delay=delay,
                    rng=rng_local,
                )

                # Dirección física correcta: x impulsa a y.
                rows.extend(
                    evaluate_driver_response(
                        driver=x,
                        response=y,
                        cfg=cfg,
                        rng_null=rng_null,
                        experiment="C_unidirectional_delayed",
                        driver_name="x",
                        response_name="y",
                        repetition=rep,
                        simulation_seed=sim_seed,
                        epsilon=eps,
                        delay_true=delay,
                        expected_tau=delay,
                        rx=cfg.rx,
                        ry=cfg.ry,
                    )
                )

                # Control de dirección opuesta: y impulsa a x.
                rows.extend(
                    evaluate_driver_response(
                        driver=y,
                        response=x,
                        cfg=cfg,
                        rng_null=rng_null,
                        experiment="C_unidirectional_delayed",
                        driver_name="y",
                        response_name="x",
                        repetition=rep,
                        simulation_seed=sim_seed,
                        epsilon=eps,
                        delay_true=delay,
                        expected_tau=np.nan,
                        rx=cfg.ry,
                        ry=cfg.rx,
                    )
                )

    # --------------------------------------------------------
    # D. Fuente común retardada
    # --------------------------------------------------------
    print("D. Fuente comun retardada")

    delay_x = 0

    for delay_y in cfg.delays:
        expected_tau = delay_y - delay_x

        for eps in cfg.eps_grid:
            for rep in range(cfg.n_rep):
                sim_seed = int(rng_sim.integers(0, 2**32 - 1))
                rng_local = np.random.default_rng(sim_seed)

                x, y = simulate_common_driver(
                    cfg=cfg,
                    eps=eps,
                    delay_x=delay_x,
                    delay_y=delay_y,
                    rng=rng_local,
                )

                rows.extend(
                    evaluate_driver_response(
                        driver=x,
                        response=y,
                        cfg=cfg,
                        rng_null=rng_null,
                        experiment="D_common_driver",
                        driver_name="x",
                        response_name="y",
                        repetition=rep,
                        simulation_seed=sim_seed,
                        epsilon=eps,
                        delay_true=delay_y,
                        expected_tau=expected_tau,
                        rx=cfg.rx,
                        ry=cfg.ry,
                        extra_label=f"delay_x={delay_x}",
                    )
                )

    # --------------------------------------------------------
    # E. Acoplamiento bidireccional
    # --------------------------------------------------------
    print("E. Acoplamiento bidireccional")

    bidirectional_delays = (0,) + cfg.delays

    for delay in bidirectional_delays:
        for eps in cfg.eps_grid:
            for rep in range(cfg.n_rep):
                sim_seed = int(rng_sim.integers(0, 2**32 - 1))
                rng_local = np.random.default_rng(sim_seed)

                x, y = simulate_bidirectional(
                    cfg=cfg,
                    eps=eps,
                    delay=delay,
                    rng=rng_local,
                )

                rows.extend(
                    evaluate_curve(
                        x=x,
                        y=y,
                        cfg=cfg,
                        rng_null=rng_null,
                        experiment="E_bidirectional",
                        direction="x_to_y",
                        repetition=rep,
                        simulation_seed=sim_seed,
                        epsilon=eps,
                        delay_true=delay,
                        expected_tau=delay,
                        rx=cfg.rx,
                        ry=cfg.ry,
                    )
                )

    # --------------------------------------------------------
    # F. Acoplamiento retardado con parámetros distintos
    # --------------------------------------------------------
    print("F. Acoplamiento retardado con parametros distintos")

    rx_mismatch = 4.0
    ry_mismatch = 3.99

    for delay in cfg.delays:
        for eps in cfg.eps_grid:
            for rep in range(cfg.n_rep):
                sim_seed = int(rng_sim.integers(0, 2**32 - 1))
                rng_local = np.random.default_rng(sim_seed)

                x, y = simulate_unidirectional(
                    cfg=cfg,
                    eps=eps,
                    delay=delay,
                    rng=rng_local,
                    rx=rx_mismatch,
                    ry=ry_mismatch,
                )

                rows.extend(
                    evaluate_driver_response(
                        driver=x,
                        response=y,
                        cfg=cfg,
                        rng_null=rng_null,
                        experiment="F_unidirectional_mismatch",
                        driver_name="x",
                        response_name="y",
                        repetition=rep,
                        simulation_seed=sim_seed,
                        epsilon=eps,
                        delay_true=delay,
                        expected_tau=delay,
                        rx=rx_mismatch,
                        ry=ry_mismatch,
                    )
                )

    raw = pd.DataFrame(rows)

    raw_path = output_dir / "S_bivariante_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarize_results(raw)
    summary_path = output_dir / "S_bivariante_summary.csv"
    summary.to_csv(summary_path, index=False)

    detection_mode = calibrate_detection_sign(summary)

    detected_delays = recover_delays(
        summary=summary,
        detection_mode=detection_mode,
    )

    detected_path = output_dir / "S_bivariante_detected_delays.csv"
    detected_delays.to_csv(detected_path, index=False)

    make_all_plots(
        summary=summary,
        detected_delays=detected_delays,
        detection_mode=detection_mode,
        output_dir=output_dir,
    )

    print("\nArchivos guardados:")
    print(raw_path)
    print(summary_path)
    print(detected_path)
    print(f"\nModo de deteccion calibrado: {detection_mode}")

    return raw, summary, detected_delays, detection_mode


# ============================================================
# 6. Resumen estadístico
# ============================================================

def summarize_results(raw):
    group_cols = [
        "experiment",
        "direction",
        "epsilon",
        "delay_true",
        "expected_tau",
        "tau",
        "rx",
        "ry",
        "extra_label",
    ]

    summary = (
        raw
        .groupby(group_cols, dropna=False)
        .agg(
            n_rep=("Z", "count"),
            S_obs_mean=("S_obs", "mean"),
            S_obs_std=("S_obs", "std"),
            mu_null_mean=("mu_null", "mean"),
            sigma_null_mean=("sigma_null", "mean"),
            DeltaS_mean=("DeltaS", "mean"),
            DeltaS_std=("DeltaS", "std"),
            Z_mean=("Z", "mean"),
            Z_median=("Z", "median"),
            Z_std=("Z", "std"),
            p_less_mean=("p_less", "mean"),
            p_greater_mean=("p_greater", "mean"),
            p_two_sided_mean=("p_two_sided", "mean"),
            frac_p_less_005=("p_less", lambda x: float(np.mean(x < 0.05))),
            frac_p_greater_005=("p_greater", lambda x: float(np.mean(x < 0.05))),
        )
        .reset_index()
    )

    summary["Z_sem"] = summary["Z_std"] / np.sqrt(summary["n_rep"])

    return summary


# ============================================================
# 7. Calibración del signo de detección
# ============================================================

def calibrate_detection_sign(summary):
    """
    Determina si la dependencia conocida produce mínimos o máximos de Z.

    Se usa el experimento B, que tiene acoplamiento instantáneo conocido.
    La referencia es Z(tau=0) para epsilon > 0.
    """
    ref = summary[
        (summary["experiment"] == "B_unidirectional_instant")
        & (summary["direction"] == "x_to_y")
        & (summary["tau"] == 0)
        & (summary["epsilon"] > 0)
    ].copy()

    if ref.empty:
        raise ValueError("No hay datos suficientes para calibrar el signo.")

    z_reference = ref["Z_mean"].median()

    if z_reference < 0:
        return "minimum"
    else:
        return "maximum"


def recover_delays(summary, detection_mode):
    """
    Recupera tau_hat a partir del extremo de Z_mean.

    Se aplica solamente a configuraciones con expected_tau definido.
    """
    valid = summary[
        summary["expected_tau"].notna()
        & (summary["epsilon"] > 0)
    ].copy()

    group_cols = [
        "experiment",
        "direction",
        "epsilon",
        "delay_true",
        "expected_tau",
        "rx",
        "ry",
        "extra_label",
    ]

    detected = []

    for keys, df in valid.groupby(group_cols, dropna=False):
        df = df.sort_values("tau")

        if detection_mode == "minimum":
            selected = df.loc[df["Z_mean"].idxmin()]
        else:
            selected = df.loc[df["Z_mean"].idxmax()]

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "tau_hat": int(selected["tau"]),
                "Z_extreme": selected["Z_mean"],
                "tau_error": int(selected["tau"] - selected["expected_tau"]),
                "correct_detection": bool(
                    selected["tau"] == selected["expected_tau"]
                ),
            }
        )

        detected.append(row)

    return pd.DataFrame(detected)


# ============================================================
# 8. Gráficas
# ============================================================

def _subset_summary(summary, experiment, direction="x_to_y", delay_true=None):
    df = summary[
        (summary["experiment"] == experiment)
        & (summary["direction"] == direction)
    ].copy()

    if delay_true is not None:
        df = df[df["delay_true"] == delay_true]

    return df


def plot_independent_control(summary, output_dir):
    df = _subset_summary(summary, "A_independent")

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        df["tau"],
        df["Z_mean"],
        yerr=df["Z_sem"],
        marker="o",
        capsize=3,
    )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$Z(\tau)$")
    ax.set_title("Control independiente")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "A_independent_Z_tau.png", dpi=250)
    plt.close(fig)


def plot_heatmap(summary, experiment, direction, delay_true, output_dir):
    df = _subset_summary(
        summary=summary,
        experiment=experiment,
        direction=direction,
        delay_true=delay_true,
    )

    if df.empty:
        return

    pivot = df.pivot_table(
        index="epsilon",
        columns="tau",
        values="Z_mean",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    tau_values = pivot.columns.to_numpy(dtype=float)
    eps_values = pivot.index.to_numpy(dtype=float)
    z_values = pivot.to_numpy(dtype=float)

    max_abs = np.nanmax(np.abs(z_values))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.pcolormesh(
        tau_values,
        eps_values,
        z_values,
        shading="nearest",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    if np.isfinite(delay_true):
        ax.axvline(delay_true, linestyle="--", linewidth=1.5)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(
        f"{experiment} | direction={direction} | delay={delay_true}"
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$Z(\tau,\varepsilon)$")

    fig.tight_layout()

    filename = (
        f"heatmap_{experiment}_{direction}_delay_{delay_true}.png"
        .replace(".", "p")
    )
    fig.savefig(output_dir / filename, dpi=250)
    plt.close(fig)


def plot_detected_delays(detected_delays, experiment, output_dir):
    df = detected_delays[
        (detected_delays["experiment"] == experiment)
        & (detected_delays["direction"] == "x_to_y")
    ].copy()

    if df.empty:
        return

    for delay_true in sorted(df["delay_true"].dropna().unique()):
        local = df[df["delay_true"] == delay_true].sort_values("epsilon")

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            local["epsilon"],
            local["tau_hat"],
            marker="o",
            label=r"$\widehat{\tau}$",
        )

        ax.axhline(
            local["expected_tau"].iloc[0],
            linestyle="--",
            linewidth=1.5,
            label=r"$\tau_{\mathrm{real}}$",
        )

        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"Retardo detectado $\widehat{\tau}$")
        ax.set_title(f"{experiment} | delay={delay_true}")
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()

        filename = (
            f"detected_delay_{experiment}_delay_{delay_true}.png"
            .replace(".", "p")
        )
        fig.savefig(output_dir / filename, dpi=250)
        plt.close(fig)


def make_all_plots(summary, detected_delays, detection_mode, output_dir):
    output_dir = Path(output_dir)

    plot_independent_control(summary, output_dir)

    # Instantáneo.
    plot_heatmap(
        summary=summary,
        experiment="B_unidirectional_instant",
        direction="x_to_y",
        delay_true=0,
        output_dir=output_dir,
    )

    # Retardado unidireccional, dirección correcta e invertida.
    for delay in sorted(
        summary.loc[
            summary["experiment"] == "C_unidirectional_delayed",
            "delay_true",
        ].dropna().unique()
    ):
        plot_heatmap(
            summary=summary,
            experiment="C_unidirectional_delayed",
            direction="x_to_y",
            delay_true=delay,
            output_dir=output_dir,
        )

        plot_heatmap(
            summary=summary,
            experiment="C_unidirectional_delayed",
            direction="y_to_x",
            delay_true=delay,
            output_dir=output_dir,
        )

    # Fuente común.
    for delay in sorted(
        summary.loc[
            summary["experiment"] == "D_common_driver",
            "delay_true",
        ].dropna().unique()
    ):
        plot_heatmap(
            summary=summary,
            experiment="D_common_driver",
            direction="x_to_y",
            delay_true=delay,
            output_dir=output_dir,
        )

    # Bidireccional.
    for delay in sorted(
        summary.loc[
            summary["experiment"] == "E_bidirectional",
            "delay_true",
        ].dropna().unique()
    ):
        plot_heatmap(
            summary=summary,
            experiment="E_bidirectional",
            direction="x_to_y",
            delay_true=delay,
            output_dir=output_dir,
        )

    # Parámetros distintos.
    for delay in sorted(
        summary.loc[
            summary["experiment"] == "F_unidirectional_mismatch",
            "delay_true",
        ].dropna().unique()
    ):
        plot_heatmap(
            summary=summary,
            experiment="F_unidirectional_mismatch",
            direction="x_to_y",
            delay_true=delay,
            output_dir=output_dir,
        )

    for experiment in [
        "B_unidirectional_instant",
        "C_unidirectional_delayed",
        "D_common_driver",
        "E_bidirectional",
        "F_unidirectional_mismatch",
    ]:
        plot_detected_delays(
            detected_delays=detected_delays,
            experiment=experiment,
            output_dir=output_dir,
        )

    with open(output_dir / "detection_mode.txt", "w", encoding="utf-8") as file:
        file.write(f"Detection mode calibrated from instantaneous coupling: {detection_mode}\n")


# ============================================================
# 9. Ejecución
# ============================================================

if __name__ == "__main__":

    # Primera corrida: inspección general.
    cfg = ROBUST

    raw, summary, detected_delays, detection_mode = run_all_experiments(
        cfg=cfg,
        output_dir="resultados_S_bivariante_ROBUST",
    )

    print("\nResumen de detección:")
    print(
        detected_delays[
            [
                "experiment",
                "direction",
                "epsilon",
                "delay_true",
                "expected_tau",
                "tau_hat",
                "Z_extreme",
                "tau_error",
                "correct_detection",
            ]
        ].head(30)
    )

    # Para la corrida robusta, sustituir:
    #
    # cfg = ROBUST
    # raw, summary, detected_delays, detection_mode = run_all_experiments(
    #     cfg=cfg,
    #     output_dir="resultados_S_bivariante_ROBUST",
    # )