using DelayDiffEq, DDEProblemLibrary, DiffEqDevTools, DiffEqCallbacks
using Test

const prob = prob_dde_constant_1delay_scalar
const alg = MethodOfSteps(Tsit5(); constrained = false)

# continuous callback
@testset "continuous" begin
    cb = ContinuousCallback(
        (u, t, integrator) -> t - 2.6, # Event when event_f(t,u,k) == 0
        integrator -> (integrator.u = -integrator.u)
    )

    sol1 = solve(prob, alg, callback = cb)
    ts = findall(x -> x ≈ 2.6, sol1.t)
    @test length(ts) == 2
    @test sol1.u[ts[1]] == -sol1.u[ts[2]]
    @test sol1(
        2.6;
        continuity = :right
    ) ≈ -sol1(2.6; continuity = :left) atol = 1.0e-5

    # fails on 32bit?!
    # see https://github.com/SciML/DelayDiffEq.jl/pull/180
    if Int === Int64
        sol2 = solve(prob, alg, callback = cb, dtmax = 0.01)
        ts = findall(x -> x ≈ 2.6, sol2.t)
        @test length(ts) == 2
        @test sol2.u[ts[1]] == -sol2.u[ts[2]]
        @test sol2(2.6; continuity = :right) ≈
            -sol2(2.6; continuity = :left)

        sol3 = appxtrue(sol1, sol2)
        @test sol3.errors[:L2] < 1.5e-2
        @test sol3.errors[:L∞] < 3.5e-2
    end
end

# discrete callback
@testset "discrete" begin
    # Automatic absolute tolerances
    cb = AutoAbstol()

    sol1 = solve(prob, alg, callback = cb)
    sol2 = solve(prob, alg, callback = cb, dtmax = 0.01)
    sol3 = appxtrue(sol1, sol2)

    @test sol3.errors[:L2] < 1.4e-3
    @test sol3.errors[:L∞] < 4.1e-3

    # Terminate early
    cb = DiscreteCallback((u, t, integrator) -> t == 4, terminate!)
    sol = @test_logs solve(prob, alg; callback = cb, tstops = [4.0])
    @test sol.t[end] == 4
end

@testset "save discontinuity" begin
    f(du, u, h, p, t) = (du .= 0)
    prob = DDEProblem(f, [0.0], nothing, (0.0, 1.0))

    condition(u, t, integrator) = t == 0.5
    global iter
    function affect!(integrator)
        integrator.u[1] += 100
        global iter = integrator.iter
    end
    cb = DiscreteCallback(condition, affect!)
    sol = solve(prob, MethodOfSteps(Tsit5()), callback = cb, tstops = [0.5])
    @test sol.t[iter + 1] == sol.t[iter + 2]
    @test sol.u[iter + 1] == [0.0]
    @test sol.u[iter + 2] != [0.0]
end

# Issue #341: PeriodicCallback support for DDEIntegrator
@testset "periodic callback" begin
    # Simple DDE with constant delay
    function f(du, u, h, p, t)
        du[1] = -u[1] + h(p, t - 1.0)[1]
    end
    h(p, t) = [1.0]
    prob = DDEProblem(f, [1.0], h, (0.0, 10.0); constant_lags = [1.0])

    # Count how many times the callback is triggered
    counter = Ref(0)
    function affect!(integrator)
        counter[] += 1
    end
    cb = PeriodicCallback(affect!, 1.0)

    sol = solve(prob, MethodOfSteps(Tsit5()), callback = cb)
    @test sol.retcode == ReturnCode.Success
    # Should be triggered approximately 10 times (at t=1,2,3,...,10)
    @test counter[] >= 9
end

# Issue https://github.com/SciML/DifferentialEquations.jl/issues/1124
# PresetTimeCallback should not be skipped due to floating-point accumulation
# in propagated tstops from constant lags
@testset "PresetTimeCallback with constant lags" begin
    # This test reproduces the issue where PresetTimeCallback is skipped
    # when constant lag multiples coincide with preset times due to FP error.
    # With delays = [0.2, 4], t=9.0 = 45*0.2 and the propagated tstop
    # accumulates to 8.999999999999996 due to repeated t + lag additions.

    function f(du, u, h, p, t)
        du[1] = -u[1] + h(p, t - p[1])[1]
    end
    h(p, t) = [1.0]

    # Test with delays that cause the issue: 0.2 is a problematic lag
    # because 9.0 = 45 * 0.2 and FP accumulation lands just below 9.0
    delays = [0.2, 4.0]
    tspan = (0.0, 12.0)

    # Track which preset times were actually hit
    callback_times = Float64[]
    preset_times = [2.0, 3.0, 9.0, 10.0]

    function record_callback!(integrator)
        push!(callback_times, integrator.t)
    end

    cb = PresetTimeCallback(preset_times, record_callback!)
    prob = DDEProblem(f, [1.0], h, tspan, [delays[1]]; constant_lags = delays)
    sol = solve(prob, MethodOfSteps(Tsit5()), callback = cb)

    @test sol.retcode == ReturnCode.Success
    # All preset times should have been hit
    @test length(callback_times) == length(preset_times)
    for pt in preset_times
        @test any(ct -> isapprox(ct, pt; atol = 1.0e-10), callback_times)
    end
end
