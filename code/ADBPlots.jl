module ADBPlots

export plot, plot_timeseries, plot_timeseries!, vary_smoothing_of_timeseries
export bootstrap_histogram, bootstrap_histogram!

import ADBUtilities: @colon, @extract_from, @more, @dbg, @work, @getfor, keys_ordered_by_values
import Regressors: LoessRegressor, predict
import TreeCollections: DataTable
import Validation: bootstrap_resample_of_mean
import Plots.plot
import StatsBase
using Plots, StatPlots

"""
## `plot(d)`

Produces a bar plot of the dictionary `d`, whose values are required
to be of `Real` type. In the plot the keys of the dictionary are ordered
by the corresponding values.

"""
function plot{S,T<:Real}(d::Dict{S,T}; params...)
    x = String[]
    y = T[]
    for k in keys_ordered_by_values(d)
        push!(x, string(k))
        push!(y, d[k])
    end
    return bar(x,y; params...)
end
function plot!{S,T<:Real}(d::Dict{S,T}; params...)
    x = String[]
    y = T[]
    for k in keys_ordered_by_values(d)
        push!(x, string(k))
        push!(y, d[k])
    end
    return bar!(x,y; params...)
end
"""

This function plots timeseries. The first input array represents time
and the second input array represents the values to be plotted. An
optional parameter called `smoothing` bewteen 0 and 1 determines the
smoothing of a curve fitted to the data using a LOESS scheme. The
function has an exclamation mark form.

"""
function plot_timeseries(x, y; smoothing=0.4)

   n_patterns = length(x)

   imin = indmin(x)
   imax = indmax(x)
 
   scatter(x, y; markersize=1)
   rgs = LoessRegressor(x, y, 1:n_patterns; span=smoothing)
   xp = collect(linspace(x[imin], x[imax], 500))
   yp = predict(rgs, xp)

   return plot!(xp,yp)
end

"""

This function plots timeseries. The first input array represents time
and the second input array represents the values to be plotted. An
optional parameter called `smoothing` bewteem 0 and 1 determines the
smoothing of a curve fitted to the data using a LOESS scheme. The
function has an non-exclamation mark form.

"""
function plot_timeseries!(x, y; smoothing=0.4)

    n_patterns = length(x)

    imin = indmin(x)
    imax = indmax(x)
 
    scatter!(x,y; markersize=1)
    rgs = LoessRegressor(x, y, 1:n_patterns; span=smoothing)
    xp = collect(linspace(x[imin], x[imax], 500))
    yp = predict(rgs, xp)

    return plot!(xp,yp)
end


function vary_smoothing_of_timeseries(t, y; min=0.1, max=0.9, n=800, verbose=true)
    if length(t) != length(y)
        throw(Base.error("Arrays must have the same length."))
    end
    yc = convert(Array{Float64}, y)
    n_patterns = length(t)
    imin = indmin(t)
    imax = indmax(t)
    inner_bag = []
    for i in 1:n_patterns
        if i != imin && i != imax
            push!(inner_bag, i)
        end
    end
    bag=[imin, imax]
    append!(bag, StatsBase.sample(inner_bag, n_patterns - 2, replace=false))
    n_patterns = length(t)
    n_train = round(Int, 0.7*n_patterns) 
    bag_train = bag[1:n_train]
    bag_test = bag[(n_train + 1):end]
    return plot(vary_float_parameter(LoessRegressor, t, yc, bag_train, bag_test; 
        to_be_varied=:span, min=min, max=max, n=n, verbose=verbose))
end

function bootstrap_histogram(v; parameters...)
    params = Dict(parameters)
    @extract_from params label string(now().instant.periods.value)
    bootstrap = bootstrap_resample_of_mean(v)
    p = histogram(bootstrap; alpha = 0.5, normalized=true, label=label, params...)
    return density!(bootstrap; label="", linewidth=2, color=:black)
end


function bootstrap_histogram!(v; parameters...)
    params = Dict(parameters)
    @extract_from params label string(now().instant.periods.value)
    bootstrap = bootstrap_resample_of_mean(v)
    p = histogram!(bootstrap; alpha = 0.5, normalized=true, label=label, params...)
    return density!(bootstrap; label=label, linewidth=2, label="", color=:black)
end

end # of module
