using Pkg 
using PrettyTables
using Revise
using Plots
using Random
using ProgressMeter
using JLD2
using Distributions
using Parameters
using StatsBase

# model denotes whether our model is SABR, Heston, 32 or custom
model = "H"

# Choose the constants below in such a way that kappaSABR, kappaHeston or kappa32 returns the constant kappa
# That is, these constants should satisfy the middle range conditions
beta = 1 # 0 <= beta <= 1
mu = 2 # mu is in the reals
ro = -0.05 # -1 <= ro <= 1
kappa = 1 # kappa >= 0
theta = 1 # theta > 0
v = 1 # v > 0 

# Returns value of b function at state (st, vt)
function b(st, vt)

    if model == "S" # SABR
        return [0, 0]
    elseif model == "H" # Heston
        return [mu*st, v-theta*vt]
    elseif model == "3" # 32
        return [mu*st, v*vt-theta*vt^2]
    else
        return [st + vt, vt] # b1(0,v) >= 0 for all v >= 0 and b2(0) >= 0
    end
end

# Returns value of sigma function at state (st, vt)
function sigma(st, vt)

    sig = zeros(Float64, 2, 2)
    
    if model == "S" # SABR
        sig[1,1] = sqrt(1-ro^2)*vt*st^beta
        sig[1,2] = ro*vt*st^beta 
        sig[2,2] = kappaS(st)*vt 
        return sig
    elseif model == "H" # Heston
        sig[1,1] = sqrt(1-ro^2)*sqrt(vt)*st
        sig[1,2] = ro*sqrt(vt)*st
        sig[2,2] = kappaH(st)*vt^(3/2)
        return sig
    elseif model == "3" # 32
        sig[1,1] = sqrt(1-ro^2)*sqrt(vt)*st
        sig[1,2] = ro*sqrt(vt)*st
        sig[2,2] = kappa3(st, vt)*vt^(3/2)
        return sig
    else
        sig[1,1] = st # sig11(0,v) = 0 for all v >= 0
        sig[1,2] = st # sig12(0,v) = 0 for all v >= 0
        sig[2,2] = vt # sig22(0) = 0 
        return sig
    end
end

# Returns value of a function at state (st, vt)
function a(st, vt)
    sig = sigma(st,vt)
    return sig*transpose(sig)
end

# Returns the value if it is positive and 0 otherwise
function pos(value)
    return max(0, value)
end

# Returns -value if value is negative and 0 otherwise
function neg(value)
    return -min(0, value)
end

# replaces constant kappa with a function of st and vt for the SABR volatility model
function kappaS(st)
    if (0 < st <= (kappa*abs(ro))^(1/beta))
        return (st^beta)/abs(ro)
    elseif ((kappa*abs(ro))^(1/beta) < st <= (kappa/abs(ro))^(1/beta))
        return kappa
    elseif (st > (kappa/abs(ro))^(1/beta))
        return abs(ro)*st^beta
    else 
        return kappa
    end
end

# replaces constant kappa with a function of st and vt for the Heston volatility model
function kappaH(st)
    if (0 < st <= kappa*abs(ro))
        return st/abs(ro)
    elseif (kappa*abs(ro) < st <= kappa/abs(ro))
        return kappa
    elseif (st > kappa/abs(ro))
        return abs(ro)*st
    else 
        return kappa
    end
end

# replaces constant kappa with a function of st and vt for the 32 volatility model
function kappa3(st, vt) 
    if (0 < st <= kappa*abs(ro)*vt)
        return st/(abs(ro)*vt)
    elseif (kappa*abs(ro)*vt < st <= (kappa*vt)/abs(ro))
        return kappa
    elseif (st > kappa*vt/abs(ro))
        return abs(ro)*st/vt
    else 
        return kappa
    end
end

# Calculates birth and death rates where c1 corresponds to alpha, c2 to delta, c3 to beta, and c4 to kappa
# If c1 = c2 = c3 = c4 = 1/2 then we do four double operations approximation
# If c1 = 1, c2 = 0, c3 = 1, c4 = 0 then we do two double operations approximation
function bdRates(st, vt, N, c1, c2, c3, c4)

    sig = sigma(st, vt)
    rates = Float64[]
    
    push!(rates, N*pos(b(st, vt)[1]) + N^2*(sig[1,1]^2 + sig[1,2]^2)/2 - N^2*(c1*pos(sig[1,2]) + c3*neg(sig[1,2]))*sig[2,2])
    push!(rates, N*pos(b(st, vt)[2]) + N^2*(sig[2,2]^2)/2 - N^2*(c1*pos(sig[1,2]) - c3*neg(sig[1,2]))*sig[2,2])
    push!(rates, N*neg(b(st, vt)[1]) + N^2*(sig[1,1]^2 + sig[1,2]^2)/2 - N^2*(c1*pos(sig[1,2]) + c3*neg(sig[1,2]))*sig[2,2])
    push!(rates, N*neg(b(st, vt)[2]) + N^2*(sig[2,2]^2)/2 - N^2*(c1*pos(sig[1,2]) - c3*neg(sig[1,2]))*sig[2,2])
    push!(rates, N^2*c1*pos(sig[1,2])*sig[2,2])
    push!(rates, N^2*c2*pos(sig[1,2])*sig[2,2])
    push!(rates, N^2*c4*neg(sig[1,2])*sig[2,2])
    push!(rates, N^2*c3*neg(sig[1,2])*sig[2,2]) # or other way around with c3 and c4?

    return rates
end

# Simulates from time 0 to T the birth death process with magnification level N that starts with population (st, vt), 
# and with constant weights c1, c2, c3, c4 corresponding to alpha, delta, beta, kappa respectively
function simulator(st, vt, N, T, c1, c2, c3, c4)

    stateSequence= Tuple{Float64, Float64}[(st, vt)] # records state changes
    times = Float64[0.0] # records times when state changes

    while times[end] < T # up to time T

        rate = 0 # total of all rates
        rates = bdRates(st, vt, N, c1, c2, c3, c4)
        
        for i ∈ 1:length(rates)
            rate += rates[i]
        end

        probs = Float64[] # relative probability of each of the possible changes in the state

        for i ∈ 1:length(rates)
            push!(probs, rates[i]/rate)
        end

        cumProbs = cumsum(probs)
        r = rand()
        change = 1

        # determine which state change occurs next time
        while r >= cumProbs[change] && change < length(cumProbs)
            change += 1
        end

        state = (st, vt)

        if change == 1
            state = (st+1/N, vt)
        elseif change == 2
            state = (st, vt+1/N)
        elseif change == 3 && st - 1/N >= 0
            state = (st-1/N, vt)
        elseif change == 4 && vt - 1/N >= 0
            state = (st, vt-1/N)
        elseif change == 5
            state = (st+1/N, vt+1/N)
        elseif change == 6 && st - 1/N >= 0 && vt - 1/N >= 0
            state = (st-1/N, vt-1/N)
        elseif change == 7 && st - 1/N >= 0
            state = (st-1/N, vt+1/N)
        elseif change == 8 && vt - 1/N >= 0
            state = (st+1/N, vt-1/N)
        end

        push!(stateSequence, state)
        st = state[1]
        vt = state[2]

        time = rand(Exponential(1/rate)) # get interarrival time for state change
        push!(times, times[end] + time)
    end

    return stateSequence, times
end

# simulate an approximate Markov Chain solution to the SDE
stateSequence, times = simulator(15, 2, 10, 2, 1, 0, 1, 0) 
priceSequence = Float64[]
volSequence = Float64[]

for i ∈ 1:length(stateSequence)
    push!(priceSequence, stateSequence[i][1])
    push!(volSequence, stateSequence[i][2])
end

# print(stateSequence)
# print(times)

# plot stock price vs volatility simulation
plot(times, priceSequence, label = "Price", xlabel = "Time", ylabel = "Price / Volatility", title = "Price vs Volatility")
plot!(times, volSequence, label = "Volatility")



