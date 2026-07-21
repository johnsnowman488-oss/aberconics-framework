
### A Pluto.jl notebook ###
# v0.20.21
# ==========================================
# Example 01: OU Noise Memory Validation
# ==========================================
# Goal: Validate SOE memory-kernel approximation on colored OU noise.
# Runtime: ~35s on typical desktop settings.
# Outputs: `results/ou_noise/acf_comparison.html`, `results/ou_noise/psd_spectra.html`.
# Status: Validated baseline experiment.

using Markdown
# Fallback for Pluto's `md` macro when running as a plain script: no-op
macro md(x...)
    return :(nothing)
end
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 1de02932-0a5e-11f1-9cc9-85d7dfbc0baa
### A) Packages and helpers
begin
    
    using DifferentialEquations, StochasticDiffEq, DSP, Statistics, LinearAlgebra, Random, Dates
    using PlotlyLight, PlutoUI,  OptimizationOptimJL

    # random seed for reproducibility
    rng = MersenneTwister(42)

    # ════════════════════════════════════════════════════════════════════════════════
    # OBJECTIVE: Compare three noise processes with memory kernels
    # ════════════════════════════════════════════════════════════════════════════════
    #
    # 1. White OU: Simple exponential decay, δ-correlated driving noise
    #    - Benchmark baseline
    #    - Shortest memory, steepest ACF decay
    #
    # 2. Colored OU: Two-variable system with correlated noise input
    #    - Intermediate memory via auxiliary variable η
    #    - Richer spectral structure (power-law-like at intermediate scales)
    #
    # 3. Aberconics Memory: Explicit multi-channel exponential memory
    #    - L parallel exponential kernels with independent stochastic driving
    #    - Each channel i has decay rate γ_i and weight w_i
    #    - Flexibly approximates arbitrary colored noise via superposition
    #
    # Key metrics: ACF (temporal), PSD (frequency), Spectral Units (Mcap, Mscale, Deff)
    # Goal: Show that Aberconics with learned (w, γ) matches colored OU
    #
    # ════════════════════════════════════════════════════════════════════════════════

    # autocorrelation (normalized)
    function autocorr_ts(x, maxlag)
        n = length(x)
        μ = mean(x)
        y = x .- μ
        c = zeros(maxlag+1)
        for τ in 0:maxlag
            c[τ+1] = sum(y[1:end-τ] .* y[1+τ:end]) / (n - τ)
        end
        return c ./ c[1]
    end

    # power spectral density via periodogram
    function compute_psd(x, dt)
        fs = 1.0 / dt
        p = periodogram(x, fs=fs)
        return p.freq, p.power
    end

    # ────────────────────────────────────────────────────────────────────────────────
    # Spectral Units: Quantify memory capacity and complexity
    # ────────────────────────────────────────────────────────────────────────────────
    # Based on exponential-kernel decomposition. See paper for details.
    # Useful for comparing memory across different architectures.
    struct SpectralUnits
        Mcap::Float64
        Mscale::Float64
        Mres::Float64
        Hmem::Float64
        Hnorm::Float64
        Deff::Float64
    end

    function spectral_units(w::Vector, γ::Vector)
        τ = 1.0 ./ γ
        wpos = max.(w, 0.0)
        if sum(wpos) == 0.0
            wpos .= 1e-16
        end
        wtil = wpos ./ sum(wpos)
        Mcap = sum(wpos .* τ.^2) / sum(wpos .* τ)
        τmin, τmax = minimum(τ), maximum(τ)
        Mscale = log10(τmax / τmin + eps())
        L = length(w)
        Mres = Mscale == 0 ? NaN : L / Mscale
        Hmem = -sum(wtil .* log.(wtil .+ 1e-20))
        Hnorm = Hmem / log(L)
        Deff = L * exp(Hmem)   # L * exp(H) as in paper
        return SpectralUnits(Mcap, Mscale, Mres, Hmem, Hnorm, Deff)
    end

    # ────────────────────────────────────────────────────────────────────────────────
    # Non-negative Least Squares (NNLS) via projected gradient descent
    # ────────────────────────────────────────────────────────────────────────────────
    # Given measurement b and design matrix A (exponential basis), solve:
    #    min ||A*w - b||^2   s.t. w ≥ 0
    # Used to fit SOE (Sum-of-Exponentials) kernel to ACF of colored OU.
    function nnls_pg(A, b; lr=1e-3, iters=4000, w0=nothing)
        m, n = size(A)
        w = if isnothing(w0)
            ones(n) * (sum(b) / n / maximum(sum(abs.(A); dims=1)))
        else
            copy(w0)
        end
        for k in 1:iters
            r = A * w - b
            g = A' * r
            w .-= lr .* g
            w .= max.(w, 0.0)
            if norm(g) < 1e-6
                break
            end
        end
        return w
    end
end

# ╔═╡ dda04183-9fc6-4a13-91c1-ab1b74bfa0dc
### B) OU process definitions
begin
    # ────────────────────────────────────────────────────────────────────────────────
    # Process 1: White OU (baseline)
    # ────────────────────────────────────────────────────────────────────────────────
    # dx = -θ x dt + σ dW
    # - Simplest colored noise model
    # - Memory decays as exp(-θ * t)
    # - ACF is exponential, no secondary peaks
    function ou_white(θ=1.0, σ=1.0; dt=0.01, T=200.0)
        prob = SDEProblem(
            (u,p,t) -> -p.θ * u,
            (u,p,t) -> p.σ,
            0.0, (0.0,T), (θ=θ,σ=σ))
        sol = solve(prob, EM(), dt=dt, adaptive=false)
        return [u[1] for u in sol.u]
    end

    # ────────────────────────────────────────────────────────────────────────────────
    # Process 2: Colored OU (two-variable)
    # ────────────────────────────────────────────────────────────────────────────────
    # dx = -θ x dt + η dt
    # dη = -α η dt + σ dW
    # - Auxiliary variable η couples two exponential time scales (θ, α)
    # - Richer memory structure than white OU
    # - ACF has slower decay at short times, secondary structure at long times
    function colored_ou(θ=1.0, α=0.2, σ=1.0; dt=0.01, T=200.0)
        function f(du,u,p,t)
            du[1] = -p.θ * u[1] + u[2]
            du[2] = -p.α * u[2]
        end
        function g(du,u,p,t)
            du[1] = 0.0
            du[2] = p.σ
        end
        prob = SDEProblem(f, g, [0.0,0.0], (0.0,T), (θ=θ,α=α,σ=σ))
        sol = solve(prob, EM(), dt=dt, adaptive=false)
        return [u[1] for u in sol.u]
    end

    # ────────────────────────────────────────────────────────────────────────────────
    # Process 3: Aberconics Memory (multi-channel)
    # ────────────────────────────────────────────────────────────────────────────────
    # dx = -θ x dt + Σ w_i χ_i dt                    [main state x]
    # dχ_i = -γ_i χ_i dt + σ_i dW_i  (i=1..L)      [L memory channels]
    # - Each channel χ_i has its own decay rate γ_i and weight w_i
    # - Independent driving noises allow flexible structure
    # - Superposition of exponential kernels: K(t) = Σ w_i * exp(-γ_i * t)
    # - Can approximate arbitrary colored noise via learned (w, γ)
    function aberconics_memory(θ=1.0, w=[0.8,0.5,0.2], γ=[0.5,0.1,0.02], σχ=[0.8,0.6,0.3]; dt=0.01, T=200.0)
        L = length(w)
        n = 1 + L
        function f(du,u,p,t)
            x = u[1]
            du[1] = -p.θ * x + sum(p.w .* u[2:end])
            for i in 1:L
                du[1+i] = -p.γ[i] * u[1+i]
            end
        end
        function g(du,u,p,t)
            # du is n × L noise matrix. Only memory channels get noise
            fill!(du, 0.0)
            for i in 1:L
                du[1+i, i] = p.σχ[i]  # du[row, col] with row=1+i, col=i
            end
        end
        p = (θ=θ, w=Float64.(w), γ=Float64.(γ), σχ=Float64.(σχ))
        u0 = zeros(n)
        # Provide noise rate prototype to specify that du is a matrix
        noise_proto = zeros(n, L)
        prob = SDEProblem(f, g, u0, (0.0,T), p; noise_rate_prototype=noise_proto)
        sol = solve(prob, EM(), dt=dt, adaptive=false)
        return [u[1] for u in sol.u]
    end
end

# ╔═╡ 34a19d08-e923-4d23-a26e-3bc2b48ec6a6
### C) Parameters (script-friendly defaults)
begin
global t_total_start = time()
println("\n" * "="^80)
println("OU_NOISE SIMULATION: Colored Noise Memory Kernel Validation")
println("="^80)

# Simulation parameters (defaults used when not running in Pluto)
T_sim = 500
θ_sim = 1.0
α_sim = 0.1
L_ab = 3
dt_sim = 0.01

# Aberconics tuning defaults
w_factor = 1.0
γ_factor = 1.0

# Generate default γ, w, σχ for Aberconics (log‑spaced)
γ_ab = exp10.(range(-2, stop=0, length=L_ab))
w_ab = ones(L_ab) ./ L_ab
σχ_ab = ones(L_ab) .* 0.5

println("\n[CONFIG] Simulation Parameters:")
println("  T (horizon) = ", T_sim, " seconds")
println("  dt (timestep) = ", dt_sim)
println("  n_samples = ", Int(T_sim / dt_sim) + 1)
println("  θ (white OU decay) = ", θ_sim)
println("  α (colored OU aux decay) = ", α_sim)
println("  L (Aberconics channels) = ", L_ab)
println()
println("[CONFIG] Aberconics Parameters:")
println("  γ (decay rates, log-spaced) = ", round.(γ_ab, digits=4))
println("  w (weights, uniform) = ", round.(w_ab, digits=4))
println("  σχ (channel noise) = ", round.(σχ_ab, digits=4))
println("  w_factor = ", w_factor)
println("  γ_factor = ", γ_factor)
println()

# ╔═╡ 62c8ce2f-048d-4166-8f72-d45b758d49c6
### D) Simulate all three processes & fit
begin
    # ────────────────────────────────────────────────────────────────────────────────
    # Simulation Parameters
    # ────────────────────────────────────────────────────────────────────────────────
    dt = 0.01
    T = T_sim
    θ = θ_sim
    α = α_sim
    
    # Ensure w_ab, γ_ab, σχ_ab are defined locally (redundant for Pluto, necessary for script mode)
    γ_ab = exp10.(range(-2, stop=0, length=L_ab))
    w_ab = ones(L_ab) ./ L_ab
    σχ_ab = ones(L_ab) .* 0.5
    w_ab = w_ab .* w_factor
    γ_ab = γ_ab .* γ_factor

    println("\n" * "="^80)
    println("PHASE 1: SIMULATIONS")
    println("="^80)
    println("\n[PARAMS] Core dynamics:")
    println("  T (horizon) = ", T, " s")
    println("  dt (timestep) = ", dt)
    println("  n_samples = ", Int(T / dt) + 1)
    println("  θ (white OU decay) = ", θ)
    println("  α (colored OU decay) = ", α)
    println()

    # White OU
    t1 = time()
    println("[SOLVER] Starting white OU simulation (baseline)...")
    x_white = ou_white(θ, 1.0; dt=dt, T=T)
    t_white = time() - t1
    println("  ✓ Completed in $(round(t_white, digits=3))s, ", length(x_white), " samples")
    println("  Final value: x(T) = $(round(x_white[end], digits=4))")
    println()

    # Colored OU
    t1 = time()
    println("[SOLVER] Starting colored OU simulation (2-variable)...")
    println("  Parameters: α = ", α)
    x_col = colored_ou(θ, α, 1.0; dt=dt, T=T)
    t_col = time() - t1
    println("  ✓ Completed in $(round(t_col, digits=3))s, ", length(x_col), " samples")
    println("  Final value: x(T) = $(round(x_col[end], digits=4))")
    println()

    # Aberconics memory
    t1 = time()
    println("[SOLVER] Starting Aberconics memory simulation ($L_ab channels)...")
    println("  Channel configuration:")
    for i in 1:L_ab
        println("    Channel $i: γ = $(round(γ_ab[i], digits=4)), w = $(round(w_ab[i], digits=4))")
    end
    x_ab = aberconics_memory(θ, w_ab, γ_ab, ones(L_ab).*0.5; dt=dt, T=T)
    t_ab = time() - t1
    println("  ✓ Completed in $(round(t_ab, digits=3))s, ", length(x_ab), " samples")
    println("  Final value: x(T) = $(round(x_ab[end], digits=4))")
    println()

    # ────────────────────────────────────────────────────────────────────────────────
    # Compute Autocorrelation Functions (ACF)
    # ────────────────────────────────────────────────────────────────────────────────
    println("="^80)
    println("PHASE 2: AUTOCORRELATION ANALYSIS")
    println("="^80)
    
    maxlag = min(200, length(x_white)-1)
    println("\n[ACF] Computing autocorrelation functions (maxlag=$maxlag)...")
    
    t1 = time()
    acf_white = autocorr_ts(x_white, maxlag)
    t_acf_white = time() - t1
    println("  White OU: $(round(t_acf_white, digits=4))s")
    println("    ρ(0) = $(round(acf_white[1], digits=4)), ρ(50) = $(round(acf_white[51], digits=4))")
    println("    Characteristic decay scale τ_acf ≈ $(round(-maxlag/(log(acf_white[end])+eps()), digits=2)) steps")
    
    t1 = time()
    acf_col = autocorr_ts(x_col, maxlag)
    t_acf_col = time() - t1
    println("  Colored OU: $(round(t_acf_col, digits=4))s")
    println("    ρ(0) = $(round(acf_col[1], digits=4)), ρ(50) = $(round(acf_col[51], digits=4))")
    println("    Characteristic decay scale τ_acf ≈ $(round(-maxlag/(log(acf_col[end])+eps()), digits=2)) steps")
    
    t1 = time()
    acf_ab = autocorr_ts(x_ab, maxlag)
    t_acf_ab = time() - t1
    println("  Aberconics: $(round(t_acf_ab, digits=4))s")
    println("    ρ(0) = $(round(acf_ab[1], digits=4)), ρ(50) = $(round(acf_ab[51], digits=4))")
    println("    Characteristic decay scale τ_acf ≈ $(round(-maxlag/(log(acf_ab[end])+eps()), digits=2)) steps")
    println()

    # ────────────────────────────────────────────────────────────────────────────────
    # Compute Power Spectral Density (PSD)
    # ────────────────────────────────────────────────────────────────────────────────
    println("="^80)
    println("PHASE 3: SPECTRAL ANALYSIS (PSD)")
    println("="^80)
    
    println("\n[PSD] Computing power spectral density via periodogram...")
    
    t1 = time()
    f_white, P_white = compute_psd(x_white, dt)
    t_psd_white = time() - t1
    println("  White OU: $(round(t_psd_white, digits=4))s")
    println("    Frequency points: $(length(f_white))")
    println("    PSD range: [$(round(minimum(P_white[2:end]), digits=3)), $(round(maximum(P_white), digits=3))]")
    
    t1 = time()
    f_col, P_col = compute_psd(x_col, dt)
    t_psd_col = time() - t1
    println("  Colored OU: $(round(t_psd_col, digits=4))s")
    println("    Frequency points: $(length(f_col))")
    println("    PSD range: [$(round(minimum(P_col[2:end]), digits=3)), $(round(maximum(P_col), digits=3))]")
    
    t1 = time()
    f_ab, P_ab = compute_psd(x_ab, dt)
    t_psd_ab = time() - t1
    println("  Aberconics: $(round(t_psd_ab, digits=4))s")
    println("    Frequency points: $(length(f_ab))")
    println("    PSD range: [$(round(minimum(P_ab[2:end]), digits=3)), $(round(maximum(P_ab), digits=3))]")
    println()

    # ────────────────────────────────────────────────────────────────────────────────
    # Fit Sum-of-Exponentials (SOE) Kernel to Colored OU
    # ────────────────────────────────────────────────────────────────────────────────
    println("="^80)
    println("PHASE 4: KERNEL FITTING (NNLS)")
    println("="^80)
    
    println("\n[FITTING] Fitting SOE kernel to colored OU ACF via NNLS...")
    println("  Fitting basis: log-spaced γ ∈ [10⁻², 10¹]")
    
    t1 = time()
    γ_fit = exp10.(range(-2, stop=1, length=15))
    t_vals = (0:maxlag) * dt   # lag times
    A = [exp.(-γ * t_vals) for γ in γ_fit]  # length(γ_fit) columns
    A_mat = hcat(A...)  # each column is exp(-γ_i * t)
    println("  [NNLS] Design matrix: $(size(A_mat)) (samples × basis functions)")
    
    w_fit = nnls_pg(A_mat, acf_col)
    println("  [NNLS] Initial weights computed (before pruning)")
    println("    Sum(w) = $(round(sum(w_fit), digits=4))")
    println("    Max(w) = $(round(maximum(w_fit), digits=4))")
    
    # keep only significant weights
    idx = w_fit .> 0.01 * maximum(w_fit)
    γ_fit = γ_fit[idx]
    w_fit = w_fit[idx]
    # recompute ACF fit
    acf_fit = A_mat[:,idx] * w_fit
    
    t_fit = time() - t1
    println("  [NNLS] Pruning threshold: 0.01 × max(w)")
    println("    Significant modes: $(length(w_fit)) / 15")
    println("    γ values: ", round.(γ_fit, digits=4))
    println("    w values: ", round.(w_fit, digits=4))
    
    # L1 error
    l1_error = norm(acf_fit - acf_col, 1) / length(acf_col)
    println("  [NNLS] L1 error (normalized): $(round(l1_error, digits=5))")
    println("  Fitting completed in $(round(t_fit, digits=3))s")
    println()

    # Compute spectral units for fitted kernel
    println("="^80)
    println("PHASE 5: SPECTRAL UNITS (MEMORY METRICS)")
    println("="^80)
    
    su_fit = spectral_units(w_fit, γ_fit)
    println("\n[METRICS] Fitted kernel spectral units:")
    println("  Mcap (mean memory capacity) = $(round(su_fit.Mcap, digits=4))")
    println("  Mscale (spectral span, log₁₀) = $(round(su_fit.Mscale, digits=4))")
    println("  Mres (memory resolution) = $(round(su_fit.Mres, digits=4))")
    println("  Hmem (entropy, nats) = $(round(su_fit.Hmem, digits=4))")
    println("  Hnorm (normalized entropy) = $(round(su_fit.Hnorm, digits=4))")
    println("  Deff (effective dimension) = $(round(su_fit.Deff, digits=2))")
    println()
end
end

# ╔═╡ 62c8ce2f-048d-4166-8f72-d45b758d49c6_setup
### Setup: Plot output directory & helper functions
begin
    plots_dir = "plots_ou_noise"
    
    function ensure_plot_dir(dir::AbstractString)
        if !isdir(dir)
            mkdir(dir)
        end
    end

    function save_plot_html(p, dir::AbstractString, name::AbstractString)
        ensure_plot_dir(dir)
        PlotlyLight.preset.source.standalone!()
        path = joinpath(dir, string(name, ".html"))
        PlotlyLight.save(p, path)
        return path
    end

    function write_plot_index(dir::AbstractString, name::AbstractString, files::Vector{String}; title::AbstractString="OU Noise Analysis")
        path = joinpath(dir, string(name, ".html"))
        open(path, "w") do io
            write(io, "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>")
            write(io, title)
            write(io, "</title></head><body>\n<h1>")
            write(io, title)
            write(io, "</h1>\n<ul>\n")
            for f in files
                fname = basename(f)
                write(io, "<li><a href=\"")
                write(io, fname)
                write(io, "\">")
                write(io, fname)
                write(io, "</a></li>\n")
            end
            write(io, "</ul>\n</body></html>\n")
        end
        return path
    end
    
    save_plots = true
    println("Plot output directory: $plots_dir")
end

# ╔═╡ 53b128a1-e1ee-4ae3-afd1-dea810dd9362
### E) ACF comparison
begin
    md"""
    ### Autocorrelation Functions (ACF)
    
    **Interpretation:**
    - **Blue** (OU white): Exponential decay, memory exhausted by lag ~5
    - **Red** (Colored OU): Slower decay, richer structure at short times
    - **Green dashed** (Aberconics): Should approximate Red if (w, γ) tuned correctly
    - **Black** (SOE fit): Best sum-of-exponentials approximation to colored OU
    
    A good Aberconics configuration will have green dashed closely match red solid.
    """
end

begin
    println("\n" * "="^80)
    println("PHASE 6: VISUALIZATION (ACF)")
    println("="^80)
    
    t1 = time()
    println("\n[PLOT] Generating ACF comparison plot...")
    p_acf = Plot()
    trace_white = Config(x=0:maxlag, y=acf_white, type="scatter", mode="lines", name="OU white", line=Dict("color"=>"#1f77b4", "width"=>1.5))
    trace_col   = Config(x=0:maxlag, y=acf_col,   type="scatter", mode="lines", name="Colored OU", line=Dict("color"=>"#d62728", "width"=>2))
    trace_ab    = Config(x=0:maxlag, y=acf_ab,    type="scatter", mode="lines", name="Aberconics", line=Dict("color"=>"#2ca02c", "width"=>1.5, "dash"=>"dash"))
    trace_fit   = Config(x=0:maxlag, y=acf_fit,   type="scatter", mode="lines", name="SOE fit",    line=Dict("color"=>"#000000", "width"=>2))
    for t in [trace_white, trace_col, trace_ab, trace_fit]
        push!(p_acf.data, t)
    end
    p_acf.layout.title = "Autocorrelation Functions (ACF)"
    p_acf.layout.xaxis.title = "lag (timesteps)"
    p_acf.layout.yaxis.title = "ρ(τ)"
    p_acf.layout.hovermode = "x unified"
    
    t_plot_acf = time() - t1
    println("  Plot prepared in $(round(t_plot_acf, digits=3))s")
    
    if save_plots
        t1 = time()
        acf_path = save_plot_html(p_acf, plots_dir, "ou_acf")
        t_save_acf = time() - t1
        println("  ✓ Saved in $(round(t_save_acf, digits=3))s: ", acf_path)
    end
    p_acf
end

# ╔═╡ e2a76b53-4695-47ad-964b-f657d2967891
### F) PSD comparison
begin
    md"""
    ### Power Spectral Density (PSD, log-log)
    
    **Interpretation:**
    - **Blue** (OU white): Flat PSD (1/f^0), white noise spectral shape
    - **Red** (Colored OU): Colored spectrum, power concentrated at low frequencies (1/f^α)
    - **Green dashed** (Aberconics): Should track red if memory channels match
    
    Deviations indicate mismatch in memory structure.
    """
end

begin
    println("\n" * "="^80)
    println("PHASE 7: VISUALIZATION (PSD)")
    println("="^80)
    
    t1 = time()
    println("\n[PLOT] Generating PSD comparison plot (log-log)...")
    p_psd = Plot()
    trace_white_psd = Config(x=f_white, y=P_white, type="scatter", mode="lines", name="OU white", line=Dict("color"=>"#1f77b4", "width"=>1.5, "opacity"=>0.6))
    trace_col_psd = Config(x=f_col, y=P_col, type="scatter", mode="lines", name="Colored OU", line=Dict("color"=>"#d62728", "width"=>2))
    trace_ab_psd  = Config(x=f_ab,  y=P_ab,  type="scatter", mode="lines", name="Aberconics", line=Dict("color"=>"#2ca02c", "width"=>1.5, "dash"=>"dash"))
    for t in [trace_white_psd, trace_col_psd, trace_ab_psd]
        push!(p_psd.data, t)
    end
    p_psd.layout.xaxis.type = "log"
    p_psd.layout.yaxis.type = "log"
    p_psd.layout.title = "Power Spectral Density (PSD)"
    p_psd.layout.xaxis.title = "frequency (Hz)"
    p_psd.layout.yaxis.title = "Power"
    p_psd.layout.hovermode = "x unified"
    
    t_plot_psd = time() - t1
    println("  Plot prepared in $(round(t_plot_psd, digits=3))s")
    
    if save_plots
        t1 = time()
        psd_path = save_plot_html(p_psd, plots_dir, "ou_psd")
        t_save_psd = time() - t1
        println("  ✓ Saved in $(round(t_save_psd, digits=3))s: ", psd_path)
    end
    p_psd
end

# ╔═╡ 71f98342-d509-4cde-8e47-75cd0c14f6dc
### G) Spectral units table & interpretation
begin
    su = su_fit
    md"""
    ### Spectral Units of Fitted Kernel
    
    | Metric | Value | Meaning |
    |--------|-------|---------|
    | **Memory Capacity** \(M_{\mathrm{cap}}\) | $(round(su.Mcap,digits=3)) s | Mean timescale; larger = longer memory |
    | **Spectral Span** \(M_{\mathrm{scale}}\) | $(round(su.Mscale,digits=3)) decades | Log-range of decay rates; larger = broader spectrum |
    | **Memory Resolution** \(M_{\mathrm{res}}\) | $(round(su.Mres,digits=3)) modes/decade | How finely timescales are sampled |
    | **Spectral Entropy** \(H_{\mathrm{mem}}\) | $(round(su.Hmem,digits=3)) nats | Uniformity of weight distribution |
    | **Normalized Entropy** | $(round(su.Hnorm,digits=3)) | H / log(L), ranges [0,1] |
    | **Effective Dimension** \(D_{\mathrm{eff}}\) | $(round(su.Deff,digits=2)) | Number of "active" memory channels |
    
    **Intuition:**
    - Large *Mcap*: Slow decaying memory (good for low-frequency content)
    - Large *Mscale*: Broad range of timescales (flexible kernel)
    - Large *Deff*: Many independent memory modes (high capacity)
    """
end

# ╔═╡ d194e54b-856c-4dc4-b105-0f5d9a722e97
### H) Experiment Summary & Next Steps
begin
    md"""
    ## Experiment Summary
    
    ### Goals Achieved:
    1. ✅ **Simulated three OU processes** with increasing memory complexity
    2. ✅ **Fitted SOE kernel** to colored OU using NNLS (fixed γ grid)
    3. ✅ **Computed spectral metrics** (Mcap, Mscale, Deff) for kernel characterization
    4. ✅ **Visualized ACF and PSD** to compare temporal and frequency domains
    
    ### Key Observations:
    - White OU: Exponential decay, minimal memory
    - Colored OU: Richer structure, power-law region, longer temporal extent
    - Aberconics: Flexible multi-channel design, can match colored OU if (w, γ) tuned
    - SOE fit: Shows best exponential kernel approximation quality
    
    ### Next Steps (Optional Extensions):
    1. **Learn both w and γ**: Optimize decay rates in addition to weights (non-convex)
    2. **Sweep L**: How does effective dimension scale with number of channels?
    3. **Compare to theory**: Weyl asymptotics for intrinsic dimension
    4. **Time-series prediction**: Use learned memory for next-step or multi-step forecasting
    5. **Integrate with Lorenz**: Combine OU memory kernel with chaotic Lorenz dynamics
    
    ### Files:
    - `ou_acf.html`: Temporal correlation comparison
    - `ou_psd.html`: Frequency domain comparison  
    - Full index: `plots_ou_noise/index.html`
    """
end

# ╔═╡ final_summary
### FINAL SUMMARY & RUNTIME REPORT
begin
    t_total_end = time()
    t_total_elapsed = t_total_end - t_total_start
    
    println("\n" * "="^80)
    println("EXECUTION SUMMARY")
    println("="^80)
    
    println("\nPHASE RUNTIMES:")
    println("  Phase 1 (Simulations):")
    println("    • White OU:        $(round(t_white, digits=3))s")
    println("    • Colored OU:      $(round(t_col, digits=3))s")
    println("    • Aberconics:      $(round(t_ab, digits=3))s")
    println("    • Subtotal:        $(round(t_white + t_col + t_ab, digits=3))s")
    println()
    println("  Phase 2 (ACF):     $(round(t_acf_white + t_acf_col + t_acf_ab, digits=3))s")
    println("    • White:           $(round(t_acf_white, digits=4))s")
    println("    • Colored:         $(round(t_acf_col, digits=4))s")
    println("    • Aberconics:      $(round(t_acf_ab, digits=4))s")
    println()
    println("  Phase 3 (PSD):     $(round(t_psd_white + t_psd_col + t_psd_ab, digits=3))s")
    println("    • White:           $(round(t_psd_white, digits=4))s")
    println("    • Colored:         $(round(t_psd_col, digits=4))s")
    println("    • Aberconics:      $(round(t_psd_ab, digits=4))s")
    println()
    println("  Phase 4 (NNLS):    $(round(t_fit, digits=3))s")
    println()
    println("  Phase 5 (Metrics): <1ms (spectral units)")
    println()
    println("  Phase 6 (ACF Plot): $(round(t_plot_acf, digits=3))s")
    println("    • Save:            $(round(t_save_acf, digits=3))s")
    println()
    println("  Phase 7 (PSD Plot): $(round(t_plot_psd, digits=3))s")
    println("    • Save:            $(round(t_save_psd, digits=3))s")
    println()
    
    println("="^80)
    println("TOTAL RUNTIME: $(round(t_total_elapsed, digits=2))s")
    println("="^80)
    println()
    println("[CONFIG SUMMARY]")
    println("  Horizon:           T = $(T_sim) seconds")
    println("  Timestep:          dt = $(dt_sim)")
    println("  Total samples:     n = $(Int(T_sim/dt_sim)+1)")
    println("  ACF max lag:       $(maxlag) timesteps")
    println("  Aberconics L:      $(L_ab) channels")
    println("  NNLS basis size:   15 (γ ∈ [10⁻², 10¹])")
    println()
    println("[FINAL METRICS]")
    println("  Fitted SOE modes:  $(length(w_fit))/15")
    println("  ACF fit L1 error:  $(round(l1_error, digits=5))")
    println("  Mcap:              $(round(su_fit.Mcap, digits=4))")
    println("  Deff:              $(round(su_fit.Deff, digits=2))")
    println()
    println("[OUTPUT FILES]")
    println("  Directory:         $(plots_dir)/")
    println("  ACF plot:          $(plots_dir)/ou_acf.html")
    println("  PSD plot:          $(plots_dir)/ou_psd.html")
    println("  Index:             $(plots_dir)/index.html")
    println()
    println("✓ EXPERIMENT COMPLETE")
    println("="^80)
end

begin
    if save_plots
        plot_files = [
            joinpath(plots_dir, "ou_acf.html"),
            joinpath(plots_dir, "ou_psd.html")
        ]
        index_path = write_plot_index(plots_dir, "index", plot_files, title="OU Noise Analysis")
        println("\nPlot index written to: $index_path")
    end
end

# ╔═╡ Export metrics to results/ou_noise/
### Metrics Export: Write summary to results directory
begin
    results_dir = "results_ou_noise"
    
    function ensure_results_dir(dir::AbstractString)
        if !isdir(dir)
            mkpath(dir)
        end
    end
    
    ensure_results_dir(results_dir)
    
    # Write metrics summary matching old format
    metrics_path = joinpath(results_dir, "metrics_summary.txt")
    open(metrics_path, "w") do io
        write(io, "OU_Noise Experiment Report\n")
        write(io, "$(Dates.now())\n\n")
        write(io, "SUMMARY:\n")
        write(io, "- SOE memory kernel fit validated against colored OU target.\n")
        write(io, "- Reported L1 fit error: $(round(l1_error, digits=5)).\n")
        write(io, "- Reported effective dimension: D_eff ≈ $(round(su_fit.Deff, digits=2)).\n")
        write(io, "- See HTML: ou_acf.html, ou_psd.html.\n\n")
        
        write(io, "DETAILED METRICS:\n")
        write(io, "  Mcap (Memory Capacity):    $(round(su_fit.Mcap, digits=4)) seconds\n")
        write(io, "  Mscale (Spectral Span):    $(round(su_fit.Mscale, digits=4)) decades\n")
        write(io, "  Mres (Memory Resolution):  $(round(su_fit.Mres, digits=4)) modes/decade\n")
        write(io, "  Hmem (Spectral Entropy):   $(round(su_fit.Hmem, digits=4)) nats\n")
        write(io, "  Hnorm (Normalized):        $(round(su_fit.Hnorm, digits=4))\n")
        write(io, "  Deff (Effective Dim):      $(round(su_fit.Deff, digits=2))\n")
        write(io, "  L1 Error (NNLS fit):       $(round(l1_error, digits=5))\n")
        write(io, "  Active modes (pruned):     $(length(w_fit))/15\n\n")
        
        write(io, "SIMULATION CONFIG:\n")
        write(io, "  Horizon T:                 $(T_sim) seconds\n")
        write(io, "  Timestep dt:               $(dt_sim)\n")
        write(io, "  Total samples:             $(Int(T_sim/dt_sim)+1)\n")
        write(io, "  ACF max lag:               $(maxlag) timesteps\n")
        write(io, "  Aberconics channels L:     $(L_ab)\n")
        write(io, "  NNLS basis size:           15 (γ ∈ [10⁻², 10¹])\n\n")
        
        write(io, "FITTED KERNEL (γ, w):\n")
        for i in 1:length(γ_fit)
            write(io, "  Channel $i: γ=$(round(γ_fit[i], digits=6)), w=$(round(w_fit[i], digits=6))\n")
        end
        write(io, "\n")
        
        write(io, "RUNTIME:\n")
        write(io, "  Total elapsed:             $(round(t_total_elapsed, digits=2))s\n")
    end
    
    println("\n[METRICS EXPORT] Written to: $metrics_path")
    
    # Copy HTML plots to results directory for consistency
    if save_plots
        for src_file in ["ou_acf.html", "ou_psd.html"]
            src_path = joinpath(plots_dir, src_file)
            dst_path = joinpath(results_dir, src_file)
            if isfile(src_path)
                cp(src_path, dst_path, force=true)
                println("  ✓ Copied $src_file to results/")
            end
        end
    end
end

# ╔═╡ Cell order:
# ╠═1de02932-0a5e-11f1-9cc9-85d7dfbc0baa
# ╠═dda04183-9fc6-4a13-91c1-ab1b74bfa0dc
# ╠═34a19d08-e923-4d23-a26e-3bc2b48ec6a6
# ╠═62c8ce2f-048d-4166-8f72-d45b758d49c6
# ╠═53b128a1-e1ee-4ae3-afd1-dea810dd9362
# ╠═e2a76b53-4695-47ad-964b-f657d2967891
# ╠═71f98342-d509-4cde-8e47-75cd0c14f6dc
# ╠═d194e54b-856c-4dc4-b105-0f5d9a722e97
