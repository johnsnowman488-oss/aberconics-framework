### A Pluto.jl notebook ###
# v0.20.21
# ==========================================
# Example 02: Lorenz '63 Chaos Suppression
# ==========================================
# Goal: Optimize memory channels to reduce/suppress chaos (Lyapunov-focused).
# Runtime: ~2-3 minutes in quick mode (machine-dependent).
# Outputs: `results/lorenz_63/baseline_vs_optimized.html`, `results/lorenz_63/phase_space_3d.html`.
# Status: Primary validated testbed.

using Markdown
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

# ╔═╡ 17094440-07b4-11f1-9c28-0b2ea2cb1ad8
begin
    import Pkg
   
    using DifferentialEquations, LinearAlgebra, Statistics, Random
    using DiffEqCallbacks
    using PlotlyLight, PlutoUI
    using Optimization, OptimizationOptimJL
    println("Packages loaded.")
end

# ╔═╡ e09cf67e-1306-48dc-bf09-f7bdc0cf57b4
function lorenz_memory!(du, u, p, t)
    σ, β, ρ0, γs, ws = p.σ, p.β, p.ρ0, p.γs, p.ws
    x, y, z = u[1], u[2], u[3]
    chis = u[4:end]
    
    # memory feedback
    ρ_t = ρ0 + sum(ws .* chis)
    
    # Lorenz
    du[1] = σ * (y - x)
    du[2] = x * (ρ_t - z) - y
    du[3] = x * y - β * z
    
    # memory channels (input-driven by x)
    for (i, γ) in enumerate(γs)
        du[3 + i] = -γ * chis[i] + x
    end
end

# Parameter container as a mutable struct (to allow modification in callbacks)
mutable struct LorenzParams
    σ::Float64
    β::Float64
    ρ0::Float64
    γs::Vector{Float64}
    ws::Vector{Float64}
end

# ╔═╡ d5749525-471d-48fb-9489-e6f9a7020253
function lorenz_memory_variational!(dY, Y, p, t)
    σ, β, ρ0, γs, ws = p.σ, p.β, p.ρ0, p.γs, p.ws
    L = length(γs)
    n = 3 + L

    # Base state
    x, y, z = Y[1], Y[2], Y[3]
    chis = @view Y[4:n]
    ρ_t = ρ0 + dot(ws, chis)

    dY[1] = σ * (y - x)
    dY[2] = x * (ρ_t - z) - y
    dY[3] = x * y - β * z
    for i in 1:L
        dY[3 + i] = -γs[i] * chis[i] + x
    end

    # Tangent state
    δx, δy, δz = Y[n + 1], Y[n + 2], Y[n + 3]
    δchis = @view Y[n + 4:2n]
    dδchis = @view dY[n + 4:2n]

    mem_tangent = 0.0
    for i in 1:L
        mem_tangent += ws[i] * δchis[i]
    end

    dY[n + 1] = -σ * δx + σ * δy
    dY[n + 2] = (ρ_t - z) * δx - δy - x * δz + x * mem_tangent
    dY[n + 3] = y * δx + x * δy - β * δz
    for i in 1:L
        dδchis[i] = -γs[i] * δchis[i] + δx
    end
end

function solver_candidates(hint::Symbol)
    if hint == :stiff
        return [Rosenbrock23(), Rodas5P(), Tsit5()]
    elseif hint == :explicit
        return [Tsit5(), AutoTsit5(Rosenbrock23()), Rosenbrock23(), Rodas5P()]
    else
        return [AutoTsit5(Rosenbrock23()), Rodas5P(), Rosenbrock23(), Tsit5()]
    end
end

function solve_with_fallback(prob; hint::Symbol=:auto, saveat=nothing, reltol=1e-6, abstol=1e-8, save_everystep=false, callback=nothing, maxiters=600_000, solver_warnings=false, trace_fallback=false)
    attempts = String[]
    kwargs = (reltol=reltol, abstol=abstol, save_everystep=save_everystep, maxiters=maxiters)
    if saveat !== nothing
        kwargs = merge(kwargs, (saveat=saveat,))
    end
    if callback !== nothing
        kwargs = merge(kwargs, (callback=callback,))
    end

    for alg in solver_candidates(hint)
        try
            sol = solve(prob, alg; kwargs...)
            if SciMLBase.successful_retcode(sol.retcode)
                return sol
            else
                push!(attempts, "$(nameof(typeof(alg))): retcode=$(sol.retcode)")
            end
        catch err
            push!(attempts, "$(nameof(typeof(alg))): $(typeof(err))")
        end
    end

    if trace_fallback
        println("Solver fallback trace: ", join(attempts, " | "))
    end
    error("All solver attempts failed for hint=$(hint).")
end

function estimate_lyapunov(sol, p; transient=50.0, dt_renorm=1.0, seed=1234, solver_hint=:stiff)
    n = 3 + length(p.γs)
    rng = MersenneTwister(seed)
    δ0 = randn(rng, n)
    δ0 ./= norm(δ0)

    Y0 = vcat(Array(sol.u[1]), δ0)
    prob = ODEProblem(lorenz_memory_variational!, Y0, (sol.t[1], sol.t[end]), p)

    acc = Ref(0.0)
    count = Ref(0)

    cb = PeriodicCallback((integrator) -> begin
        δ = @view integrator.u[n + 1:2n]
        nδ = norm(δ)
        if nδ > 0
            if integrator.t > transient
                acc[] += log(nδ)
                count[] += 1
            end
            δ ./= nδ
        end
    end, dt_renorm; save_positions=(false, false))

    solve_with_fallback(
        prob;
        hint=solver_hint,
        reltol=1e-7,
        abstol=1e-9,
        save_everystep=false,
        callback=cb,
        maxiters=800_000,
        solver_warnings=false
    )
    return count[] > 0 ? acc[] / (count[] * dt_renorm) : NaN
end

# Direct single-pass Lyapunov solver for optimization (Rodas4P only, no fallback)
function lyapunov_direct(γs, ws, u0; ρ0=28.0, tspan=(0.0, 120.0), transient=30.0, dt_renorm=1.0, seed=1234)
    p = LorenzParams(10.0, 8/3, ρ0, γs, ws)
    L = length(γs)
    n = 3 + L
    
    # Run base trajectory
    u0v = collect(u0)
    base_prob = ODEProblem(lorenz_memory!, u0v, tspan, p)
    base_sol = nothing
    try
        base_sol = solve(base_prob, Rodas5P(); reltol=1e-6, abstol=1e-8, save_everystep=false, maxiters=800_000)
        if !SciMLBase.successful_retcode(base_sol.retcode)
            return NaN
        end
    catch
        return NaN
    end
    
    if base_sol === nothing
        return NaN
    end
    
    # Variational solve
    rng = MersenneTwister(seed)
    δ0 = randn(rng, n)
    δ0 ./= norm(δ0)
    Y0 = vcat(Array(base_sol.u[1]), δ0)
    var_prob = ODEProblem(lorenz_memory_variational!, Y0, (base_sol.t[1], base_sol.t[end]), p)
    
    acc = Ref(0.0)
    count = Ref(0)
    cb = PeriodicCallback((integrator) -> begin
        δ = @view integrator.u[n + 1:2n]
        nδ = norm(δ)
        if nδ > 0
            if integrator.t > transient
                acc[] += log(nδ)
                count[] += 1
            end
            δ ./= nδ
        end
    end, dt_renorm; save_positions=(false, false))
    
    try
        solve(var_prob, Rodas5P(); reltol=1e-7, abstol=1e-9, save_everystep=false, callback=cb, maxiters=800_000)
    catch
        return NaN
    end
    
    return count[] > 0 ? acc[] / (count[] * dt_renorm) : NaN
end

# ╔═╡ 0aef1012-938b-4f16-9d47-c7c12ff8448b
function run_lorenz(γs, ws; ρ0=28.0, σ=10.0, β=8/3, 
                    tspan=(0.0, 200.0), saveat=0.05, u0=nothing, solver_hint=:auto, solver_trace=false)
    p = LorenzParams(σ, β, ρ0, γs, ws)
    u0v = u0 === nothing ? [1.0, 1.0, 1.0, zeros(length(γs))...] : collect(u0)
    @assert length(u0v) == 3 + length(γs) "u0 length must be 3 + length(γs)"
    prob = ODEProblem(lorenz_memory!, u0v, tspan, p)
    sol = solve_with_fallback(
        prob;
        hint=solver_hint,
        saveat=saveat,
        reltol=1e-6,
        abstol=1e-8,
        save_everystep=false,
        maxiters=800_000,
        solver_warnings=false,
        trace_fallback=solver_trace
    )
    return sol, p
end

function rho_stats(sol, p)
    L = length(p.γs)
    ρvals = Float64[]
    for u in sol.u
        chis = @view u[4:3+L]
        push!(ρvals, p.ρ0 + dot(p.ws, chis))
    end
    return (
        mean = mean(ρvals),
        std = std(ρvals),
        min = minimum(ρvals),
        max = maximum(ρvals)
    )
end

function lyapunov_calibration(; tspan=(0.0, 150.0), saveat=0.1, transient=40.0)
    γs_ref = [0.2, 0.05, 0.01]
    ws_ref = [0.0, 0.0, 0.0]
    sol_ref, p_ref = run_lorenz(γs_ref, ws_ref; tspan=tspan, saveat=saveat)
    return estimate_lyapunov(sol_ref, p_ref; transient=transient)
end

function ensure_plot_dir(dir::AbstractString)
    mkpath(dir)
    return dir
end

function save_plot_html(p, dir::AbstractString, name::AbstractString)
    ensure_plot_dir(dir)
    PlotlyLight.preset.source.standalone!()
    path = joinpath(dir, string(name, ".html"))
    PlotlyLight.save(p, path)
    return path
end

function write_plot_index(dir::AbstractString, name::AbstractString, files::Vector{String}; title::AbstractString="Lorenz '63 Plots")
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

# ╔═╡ cb0ff187-a888-468f-82ae-faee1adb036c
begin
    # Initial guess
    L = 3
    γ1_default = 0.2
    γ2_default = 0.05
    γ3_default = 0.01
    w1_default = 0.5
    w2_default = 0.2
    w3_default = 0.1
    
    @bind γ1 PlutoUI.Slider(0.001:0.001:1.0; default=γ1_default, show_value=true)
    @bind γ2 PlutoUI.Slider(0.001:0.001:1.0; default=γ2_default, show_value=true)
    @bind γ3 PlutoUI.Slider(0.001:0.001:1.0; default=γ3_default, show_value=true)
    @bind w1 PlutoUI.Slider(0.0:0.01:2.0; default=w1_default, show_value=true)
    @bind w2 PlutoUI.Slider(0.0:0.01:2.0; default=w2_default, show_value=true)
    @bind w3 PlutoUI.Slider(0.0:0.01:2.0; default=w3_default, show_value=true)
    
    γs = [γ1, γ2, γ3]
    ws = [w1, w2, w3]
    save_html_plots = true
    plots_dir = ensure_plot_dir(joinpath(pwd(), "plots_lorenz63"))
    quick_mode = true
    print_progress = true
    progress_every = 5
    md"**Memory parameters** – adjust to see effect on chaos."
end

# ╔═╡ a3ba21f6-e4c9-411b-b527-210e86e14b68
begin
    sol, p_sim = run_lorenz(γs, ws; tspan=(0.0, 200.0), saveat=0.1)
    λ = estimate_lyapunov(sol, p_sim; transient=50.0)
    ρ_base = rho_stats(sol, p_sim)
    lyap_now = round(λ, digits=5)
    rho_mean = round(ρ_base.mean, digits=4)
    rho_std = round(ρ_base.std, digits=4)
    rho_min = round(ρ_base.min, digits=4)
    rho_max = round(ρ_base.max, digits=4)
    println("Current Lyapunov exponent: ", lyap_now)
    println("rho(t) stats: mean=", rho_mean, ", std=", rho_std, ", min=", rho_min, ", max=", rho_max)
end

# ╔═╡ 2d4f8e31-8c3d-49dd-b231-6680bfe95754
begin
    # Extract data
    t = sol.t
    x = [u[1] for u in sol.u]
    
    # Create PlotlyLight figure
    plt_ts = PlotlyLight.Plot()
    trace_x = PlotlyLight.Config(
        x = t, y = x,
        type = "scatter", mode = "lines",
        name = "x(t)",
        line = Dict("color" => "blue", "width" => 1.5)
    )
    push!(plt_ts.data, trace_x)
    
    # Add memory channels if you want
    colors = ["red", "green", "purple"]
    for i in 1:L
        chi = [u[3 + i] for u in sol.u]
        trace_chi = PlotlyLight.Config(
            x = t, y = chi,
            type = "scatter", mode = "lines",
            name = "χ$(i)",
            line = Dict("color" => colors[i], "width" => 1, "dash" => "dot")
        )
        push!(plt_ts.data, trace_chi)
    end
    
    plt_ts.layout.title = "Lorenz + Memory (γ = $(round.(γs,digits=3)), w = $(round.(ws,digits=3)))"
    plt_ts.layout.xaxis.title = "time"
    plt_ts.layout.yaxis.title = "amplitude"
    plt_ts.layout.showlegend = true
    if save_html_plots
        ts_path = save_plot_html(plt_ts, plots_dir, "lorenz_timeseries")
        println("Saved plot: $(ts_path)")
    end
    
    # Optional display hook for older/newer PlotlyLight versions.
    if isdefined(PlotlyLight, :force!)
        PlotlyLight.force!(plt_ts)
    end
    plt_ts
end

# ╔═╡ 78e3f8e5-b9b2-4f0f-9fb2-9b3764651f45
begin
    x_all = [u[1] for u in sol.u]
    y_all = [u[2] for u in sol.u]
    z_all = [u[3] for u in sol.u]
    
    plt_3d = PlotlyLight.Plot()
    trace_3d = PlotlyLight.Config(
        x = x_all, y = y_all, z = z_all,
        type = "scatter3d", mode = "lines",
        line = Dict("color" => "blue", "width" => 2),
        name = "attractor"
    )
    push!(plt_3d.data, trace_3d)
    
    plt_3d.layout.title = "Phase Portrait"
    plt_3d.layout.scene = Dict(
        "xaxis_title" => "x",
        "yaxis_title" => "y",
        "zaxis_title" => "z"
    )
    if save_html_plots
        phase_path = save_plot_html(plt_3d, plots_dir, "lorenz_phase3d")
        println("Saved plot: $(phase_path)")
    end
    
    if isdefined(PlotlyLight, :force!)
        PlotlyLight.force!(plt_3d)
    end
    plt_3d
end

# ╔═╡ 7f3f7ae1-03c3-4838-b38e-0f6756b6857e
begin
    # Pack parameters in log space
    function pack(γs, ws)
        L = length(γs)
        @assert issorted(γs, rev=true) "γs must be in descending order for packing."
        dγ = zeros(L)
        dγ[L] = max(γs[L], 1e-12)
        for i in (L-1):-1:1
            dγ[i] = max(γs[i] - γs[i+1], 1e-12)
        end
        vcat(log.(dγ), log.(ws))
    end
    function unpack(θ)
        L = length(θ) ÷ 2
        dγ = exp.(θ[1:L])
        γs = similar(dγ)
        γs[L] = dγ[L]
        for i in (L-1):-1:1
            γs[i] = γs[i+1] + dγ[i]
        end
        ws = exp.(θ[L+1:2L])
        return γs, ws
    end
    
    θ0 = pack(γs, ws)
    t_opt_start = time()
    eval_counter = Ref(0)
    # Evaluation budget controls (can be tuned for quick tests)
    eval_limit = Ref(200)
    best_loss = Ref(Inf)
    best_theta = Ref(copy(θ0))

    function default_u0_set(L)
        [
            [1.0, 1.0, 1.0, zeros(L)...],
            [2.0, 3.0, 4.0, zeros(L)...],
            [-1.5, 1.0, 2.0, zeros(L)...]
        ]
    end
    
    function mean_lyapunov_for_params(γs, ws; tspan=(0.0, 120.0), saveat=0.1, transient=30.0, u0_set=default_u0_set(length(γs)), solver_hint=:stiff, opt_mode=false)
        λs = fill(0.0, length(u0_set))
        if opt_mode
            # Optimization mode: use fast Rodas4P-only solver, parallelized with threading
            Threads.@threads for i in eachindex(u0_set)
                λs[i] = lyapunov_direct(γs, ws, u0_set[i]; tspan=tspan, transient=transient, seed=1234 + i)
                if !isfinite(λs[i])
                    λs[i] = 1.0  # penalty for failed eval
                end
            end
        else
            # Validation mode: use robust solver fallback, serial evaluation
            for (i, u0) in enumerate(u0_set)
                λ_i = try
                    sol_i, p_i = run_lorenz(γs, ws; tspan=tspan, saveat=saveat, u0=u0, solver_hint=solver_hint)
                    val = estimate_lyapunov(sol_i, p_i; transient=transient, seed=1234 + i, solver_hint=solver_hint)
                    isfinite(val) ? val : 1.0
                catch
                    1.0
                end
                λs[i] = λ_i
            end
        end
        return mean(λs), λs
    end

    # Loss function with train horizon and physical regularization.
    function loss(θ; tspan=(0.0, 120.0), transient=30.0)
        γs, ws = unpack(θ)
        if any(!isfinite, γs) || any(!isfinite, ws)
            return 1e3
        end
        λ_mean, _ = mean_lyapunov_for_params(γs, ws; tspan=tspan, transient=transient, solver_hint=:stiff, opt_mode=true)
        if !isfinite(λ_mean)
            return 1e3
        end
        reg_w = 1e-3 * sum(ws)
        reg_mass = 5e-4 * sum(ws ./ γs)
        total = λ_mean + reg_w + reg_mass
        eval_counter[] += 1
        # keep track of best seen parameters so we can recover if we force-stop
        if total < best_loss[]
            best_loss[] = total
            best_theta[] = copy(θ)
        end
        if print_progress && (eval_counter[] % progress_every == 0)
            elapsed = round(time() - t_opt_start, digits=1)
            println("[opt] eval=$(eval_counter[]) loss=$(round(total, digits=5)) λ_mean=$(round(λ_mean, digits=5)) elapsed_s=$(elapsed)")
        end
        # Hard evaluation cap: throw to abort optimization when budget exceeded
        if eval_counter[] >= eval_limit[]
            println("[opt] eval limit reached: $(eval_counter[]) >= $(eval_limit[]) - aborting optimization")
            throw(ErrorException("EVAL_LIMIT_REACHED"))
        end
        return total
    end

    stage1_tspan = quick_mode ? (0.0, 80.0) : (0.0, 120.0)
    stage1_transient = quick_mode ? 20.0 : 30.0
    stage1_iters = quick_mode ? 4 : 12
    stage2_tspan = quick_mode ? (0.0, 140.0) : (0.0, 200.0)
    stage2_transient = quick_mode ? 35.0 : 50.0
    stage2_iters = quick_mode ? 3 : 8
    train_u0 = quick_mode ? default_u0_set(L)[1:1] : default_u0_set(L)
    val_u0 = default_u0_set(L)

    println("Optimization mode: ", quick_mode ? "quick" : "full")
    println("  stage1: tspan=$(stage1_tspan), transient=$(stage1_transient), maxiters=$(stage1_iters)")
    println("  stage2: tspan=$(stage2_tspan), transient=$(stage2_transient), maxiters=$(stage2_iters)")
    println("  train_u0_count: ", length(train_u0), " IC(s)")
    suggested = (stage1_iters + stage2_iters) * 10 * length(train_u0)
    println("  expected_loss_evals: ~", suggested, " total")
    # set a sensible default evaluation cap for quick testing
    eval_limit[] = 200
    println("  eval_limit set to: ", eval_limit[])
    
    # Stage 1: coarse optimization on shorter horizon.
    optf1 = OptimizationFunction((θ, _) -> loss(θ; tspan=stage1_tspan, transient=stage1_transient), Optimization.AutoFiniteDiff())
    optprob1 = OptimizationProblem(optf1, θ0)
    res1 = nothing
    try
        res1 = solve(optprob1, BFGS(), maxiters=stage1_iters)
    catch err
        if occursin("EVAL_LIMIT_REACHED", string(err))
            println("[opt] Stage1 stopped early due to eval limit; using best parameters seen so far.")
            res1 = (u = copy(best_theta[]),)
        else
            rethrow(err)
        end
    end
    # ensure best_theta reflects the stage1 result (fallback if solve returned nothing)
    if res1 === nothing
        res1 = (u = copy(best_theta[]),)
    end
    best_theta[] = copy(res1.u)

    # Stage 2: refinement on longer horizon.
    optf2 = OptimizationFunction((θ, _) -> loss(θ; tspan=stage2_tspan, transient=stage2_transient), Optimization.AutoFiniteDiff())
    optprob2 = OptimizationProblem(optf2, res1.u)
    res = nothing
    try
        res = solve(optprob2, BFGS(), maxiters=stage2_iters)
    catch err
        if occursin("EVAL_LIMIT_REACHED", string(err))
            println("[opt] Stage2 stopped early due to eval limit; using best parameters seen so far.")
            res = (u = copy(best_theta[]),)
        else
            rethrow(err)
        end
    end
    if res === nothing
        res = (u = copy(best_theta[]),)
    end

    γ_opt, w_opt = unpack(res.u)

    λ_train_mean, λ_train_all = mean_lyapunov_for_params(γ_opt, w_opt; tspan=stage2_tspan, transient=stage2_transient, u0_set=train_u0, solver_hint=:stiff)
    λ_val_mean, λ_val_all = mean_lyapunov_for_params(γ_opt, w_opt; tspan=(0.0, 260.0), transient=70.0, u0_set=val_u0, solver_hint=:stiff)

    println("Optimized parameters:")
    println("  gamma = ", round.(γ_opt, digits=4))
    println("  w = ", round.(w_opt, digits=4))
    println("Train Lyapunov (mean): ", round(λ_train_mean, digits=5), " from ", round.(λ_train_all, digits=5))
    println("Validation Lyapunov (mean): ", round(λ_val_mean, digits=5), " from ", round.(λ_val_all, digits=5))
end

# ╔═╡ 9da79db0-1a64-4463-a051-f1291437d55c
begin
    sol_opt, p_opt = run_lorenz(γ_opt, w_opt; tspan=(0.0, 200.0), saveat=0.1)
    λ_opt = estimate_lyapunov(sol_opt, p_opt; transient=50.0)
    ρ_opt = rho_stats(sol_opt, p_opt)
    
    plt_compare = PlotlyLight.Plot()
    trace_base = PlotlyLight.Config(
        x = sol.t, y = [u[1] for u in sol.u],
        type = "scatter", mode = "lines",
        name = "Baseline (λ=$(round(λ,digits=3)))",
        line = Dict("color" => "gray", "width" => 1)
    )
    trace_opt = PlotlyLight.Config(
        x = sol_opt.t, y = [u[1] for u in sol_opt.u],
        type = "scatter", mode = "lines",
        name = "Optimised (λ=$(round(λ_opt,digits=3)))",
        line = Dict("color" => "red", "width" => 1.5)
    )
    push!(plt_compare.data, trace_base)
    push!(plt_compare.data, trace_opt)
    plt_compare.layout.title = "Baseline vs Optimised Memory"
    plt_compare.layout.xaxis.title = "time"
    plt_compare.layout.yaxis.title = "x(t)"
    if save_html_plots
        compare_path = save_plot_html(plt_compare, plots_dir, "lorenz_baseline_vs_optimized")
        files = String[]
        for fname in ("lorenz_timeseries.html", "lorenz_phase3d.html", "lorenz_baseline_vs_optimized.html")
            fpath = joinpath(plots_dir, fname)
            if isfile(fpath)
                push!(files, fpath)
            end
        end
        index_path = write_plot_index(plots_dir, "index", files)
        println("Saved plot: $(compare_path)")
        println("Plot index: $(index_path)")
    end
    if isdefined(PlotlyLight, :force!)
        PlotlyLight.force!(plt_compare)
    end
    rho_opt_mean = round(ρ_opt.mean, digits=4)
    rho_opt_std = round(ρ_opt.std, digits=4)
    rho_opt_min = round(ρ_opt.min, digits=4)
    rho_opt_max = round(ρ_opt.max, digits=4)
    println("Optimized run rho(t) stats: mean=", rho_opt_mean, ", std=", rho_opt_std, ", min=", rho_opt_min, ", max=", rho_opt_max)
    plt_compare
end

# ╔═╡ Cell order:
# ╠═17094440-07b4-11f1-9c28-0b2ea2cb1ad8
# ╠═e09cf67e-1306-48dc-bf09-f7bdc0cf57b4
# ╠═d5749525-471d-48fb-9489-e6f9a7020253
# ╠═0aef1012-938b-4f16-9d47-c7c12ff8448b
# ╠═cb0ff187-a888-468f-82ae-faee1adb036c
# ╠═a3ba21f6-e4c9-411b-b527-210e86e14b68
# ╠═2d4f8e31-8c3d-49dd-b231-6680bfe95754
# ╠═78e3f8e5-b9b2-4f0f-9fb2-9b3764651f45
# ╠═7f3f7ae1-03c3-4838-b38e-0f6756b6857e
# ╠═9da79db0-1a64-4463-a051-f1291437d55c
