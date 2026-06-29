using MelodicSymmetry
using NPZ
using CSV
using DataFrames
using Random
using Statistics

# ============================================================
# Configuración
# ============================================================

MELODY_DIR = "melodies"
OUTPUT_CSV = "h_orbit_shuffle_m3_m7.csv"

MS = 3:7
N_MELODIES = 23
N_SURR = 300

CIRCULAR = false
BASE = 2
SEED = 1234

Random.seed!(SEED)

# ============================================================
# Utilidades
# ============================================================

function load_melody_npy(path::String)
    x = npzread(path)
    x = vec(x)

    if eltype(x) <: AbstractFloat
        x = x[.!isnan.(x)]
        return Int.(round.(x))
    else
        return Int.(x)
    end
end


function n_windows(L::Int, m::Int; circular::Bool=false)
    if circular
        return L
    else
        return max(L - m + 1, 0)
    end
end


function normalize_entropy(H::Real, N::Int, K::Int; base::Real=2)
    K_eff = min(N, K)

    if K_eff <= 1
        return NaN
    end

    return H / log(base, K_eff)
end


function compute_H(seq::Vector{Int}, m::Int;
                   circular::Bool=false,
                   base::Real=2,
                   normalize::Bool=false)

    L = length(seq)
    N = n_windows(L, m; circular=circular)

    if N == 0
        return NaN
    end

    H = orbit_entropy(
        seq,
        m;
        method = :mle,
        base = base,
        circular = circular,
        deployment_efficiency = false
    )

    H_orbit = H.orbit

    if normalize
        K = Int(total_orbits(m))
        return normalize_entropy(H_orbit, N, K; base=base)
    else
        return H_orbit
    end
end


function empirical_pvalues(obs::Real, null_values::Vector{Float64})
    B = length(null_values)

    p_greater = (count(x -> x >= obs, null_values) + 1) / (B + 1)
    p_less    = (count(x -> x <= obs, null_values) + 1) / (B + 1)
    p_two     = min(1.0, 2 * min(p_greater, p_less))

    return p_greater, p_less, p_two
end


function compute_shuffle_null(seq::Vector{Int}, m::Int;
                              n_surr::Int=200,
                              circular::Bool=false,
                              base::Real=2,
                              normalize::Bool=false)

    H_obs = compute_H(
        seq,
        m;
        circular=circular,
        base=base,
        normalize=normalize
    )

    H_null = Float64[]

    for s in 1:n_surr
        seq_surr = shuffle(seq)

        Hs = compute_H(
            seq_surr,
            m;
            circular=circular,
            base=base,
            normalize=normalize
        )

        push!(H_null, Hs)
    end

    mu_null = mean(H_null)
    sigma_null = std(H_null)

    Z = sigma_null > 0 ? (H_obs - mu_null) / sigma_null : NaN

    p_greater, p_less, p_two = empirical_pvalues(H_obs, H_null)

    return (
        H_obs = H_obs,
        mu_null = mu_null,
        sigma_null = sigma_null,
        Z = Z,
        p_greater = p_greater,
        p_less = p_less,
        p_two = p_two
    )
end


# ============================================================
# Cálculo completo
# ============================================================

rows = []

for piece_id in 1:N_MELODIES
    path = joinpath(MELODY_DIR, string(piece_id) * ".npy")
    seq = load_melody_npy(path)

    println("Procesando melodía ", piece_id, " | L = ", length(seq))

    for m in MS
        println("  m = ", m)

        result = compute_shuffle_null(
            seq,
            m;
            n_surr=N_SURR,
            circular=CIRCULAR,
            base=BASE,
            normalize=false
        )

        L = length(seq)
        N = n_windows(L, m; circular=CIRCULAR)
        K = Int(total_orbits(m))

        push!(rows, (
            piece_id = piece_id,
            m = m,
            L = L,
            N = N,
            K = K,
            circular = CIRCULAR,
            base = BASE,
            null_model = "shuffle",
            n_surr = N_SURR,
            H_obs = result.H_obs,
            mu_null = result.mu_null,
            sigma_null = result.sigma_null,
            Z = result.Z,
            p_greater = result.p_greater,
            p_less = result.p_less,
            p_two = result.p_two
        ))
    end
end

df = DataFrame(rows)
CSV.write(OUTPUT_CSV, df)

println("Archivo guardado en: ", OUTPUT_CSV)
println(df)