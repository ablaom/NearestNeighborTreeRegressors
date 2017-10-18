__precompile__()
module Validation

export split_bag
export test_mean_less_than_zero, test_mean_is_zero, bootstrap_analysis_of_mean
export cross_validation_error, cross_validation_errors, compare_on_test_set, bootstrap_resample_of_mean
export cv_error, cv_errors
export vary_float_parameter, vary_int_parameter
export vary_smoothing_of_timeseries
export @cv_errors_versus_ξ, @cv_error_versus_ξ, @error_versus_ξ, @bootstrap_cv_errors_versus_ξ
export learning_curve, compete

import ADBUtilities: @colon, @extract_from, argmin, argmax, keys_ordered_by_values, @dbg
import HypothesisTests: OneSampleTTest, pvalue
import Distributions
import ScikitLearnBase.BaseRegressor
import Regressors: fit!, predict, LoessRegressor, add!, rms_error, rmsl_error
import TreeCollections.DataTable


"""
# split_bag(bag, percentages...)

Assumes (but does not check) that `collect(bag)` has integer
eltype. Then splits bag into a tuple of `Vector{Int}` objects whose
lengths are given by the corresponding `percentages` of
`length(bag)`. The last percentage should not actually be provided, as
it is inferred from the preceding ones. So, for example,

    julia> split_bag(1:1000, 20, 70)
    (1:200, 201:900, 901:1000)

"""
function split_bag(bag, percentages...)
    bag = collect(bag)
    bags = []
    if sum(percentages) >= 100
        throw(DomainError)
    end
    n_patterns = length(bag)
    first = 1
    for p in percentages
        n = round(Int, p*n_patterns/100)
        n == 0 ? (Base.warn("Bag with only one element"); n = 1) : nothing
        push!(bags, bag[first:(first + n - 1)])
        first = first + n
    end
    first > n_patterns ? (Base.warn("Last bag has only one element."); first = n_patterns) : nothing
    push!(bags, bag[first:n_patterns])
    return tuple(bags...)
end

                          
"""
Returns the p-value for the null hypothesis that the mean in a list
of numbers is less than or equal to zero. This is a bootstrap test and not recommended 
for sample sizes under 100.
"""
function test_mean_less_than_zero(y; n=0)
    N = length(y)

    if n == 0
        n = 1000*N
    end

    accept_count=0
    for i in 1:n
        simulated_mean = mean(Distributions.sample(y, N))
        if simulated_mean <= 0.0
            accept_count += 1
        end
    end
    return accept_count/float(n)
end

""" 

Returns the p-value for the null hypothesis that the mean in a list of
numbers is equal to zero. This is a bootstrap test. (Have tested on
normally distributed data, for sample sizes above 100, by comparing
with
[t-test](http://juliastats.github.io/HypothesisTests.jl/stable/parametric.html#t-test-1))

"""
    
function test_mean_is_zero(y; n::Int=0)
    N = length(y)

    if n == 0
        n = 1000*N
    end

    subzero_count=0
    for i in 1:n
        simulated_mean = mean(Distributions.sample(y, N))
        if simulated_mean < 0.0
            subzero_count += 1
        end
#        @dbg i simulated_mean subzero_count
    end
    p = subzero_count/n
    if p <= 0.5
        return 2*p
    else
        return 2*(1-p)
    end
end

""" 
bootstrap_analysis_of_mean(v; c=0.95, n=10^6, n_bins = 20)

# Arguments
* v ... a vector of numerical data
* c ... confidence level

# Return value
(mu, interval, historgram) with:

* mu                                  ... mean value of elements of v
* interval                            ... bootstrap confidence interval
* histogram=(boundaries, frequencies) ... histogram of means from  with-replacement random resamplings from v (the number of resamplings being `n'). The elements of `frequencies` are normalized. Here `boundaries` are the bin boundaries. 
"""
function bootstrap_analysis_of_mean(v; c=0.95, n=10^6, n_bins = 30)
    n_samples = length(v)
    mu = mean(v)
    v = v - mu
    simulated_means = Array(Float64, n)

    for i in 1:n
        pseudo_sample = Distributions.sample(v, n_samples, replace=true)
        simulated_means[i]=mean(pseudo_sample)
    end
    alpha = (1-c)/2
    interval = (mu + quantile(simulated_means, alpha), mu + quantile(simulated_means, 1-alpha))
    boundaries, frequencies = hist(mu + float(simulated_means), 20)
    frequencies = frequencies/n 
    return mu, interval, (boundaries, frequencies)
end

"""
bootstrap_resample_of_mean(v; n=10^6)

Returns a vector of `n` estimates of the mean of the distribution generating the samples in `v` from `n` bootstrap resamples of `v`.
"""
function bootstrap_resample_of_mean(v; n=10^6)
    n_samples = length(v)
    mu = mean(v)
    simulated_means = Array(Float64, n)

    for i in 1:n
        pseudo_sample = Distributions.sample(v, n_samples, replace=true)
        simulated_means[i]=mean(pseudo_sample)
    end
    return simulated_means
end

"""
## `function compete(e0, e1; alpha=0.05)`

Given paired samples `e0` and `e1`, we test the
null-hypothesis that the underlying distributions have the same mean,
using the significance level `alpha`. Normality of the underlying
distributions is assumed and a two-sided t-test applied. 

### Return value

- '0' if the null is rejected and `e0` has the smaller mean (M1 "wins")
- '1' if the null is rejected and `e1` has the smaller mean (M0 "wins")
- 'D' if the null is accepted ("draw")

"""

function compete(e0, e1; alpha=0.05)
    t = OneSampleTTest(e1-e0)
    if pvalue(t) > alpha
        return 'D'
    end
    if mean(e0) < mean(e1)
        return '0'
    end
    return '1'
end



"""
# `cross_validation_errors(rgs::BaseRegressor, X, y, bag;`
#  `n_folds=9, scoring_function=rms_error, randomized=false, verbose=true, parallel=false)`)

Return a list of cross-validation root-mean-squared errors for
patterns `X` and responses `y`, considering only rows with indexes in
`bag`. The `bag` is initially randomized when an optional parameter
`randomized` is set to `true`.

"""
function cross_validation_errors(rgs::BaseRegressor, X, y, bag;
                                 n_folds::Int=9, scoring_function::Function=rms_error, randomized::Bool=false, parallel::Bool=false, verbose::Bool=true)

    # convert bag, possibly an iterator with no append method, to an array:
    bag = collect(bag)
    n_samples = length(bag)
    if randomized
        bag = Distributions.sample(bag, n_samples, replace=false)
    end    
    k = floor(Int,n_samples/n_folds)

    # function to return the error for the fold with `test_bag=bag[f:s]`:
    function err(f, s)
        test_bag = bag[f:s]
        train_bag = append!(bag[1:(f - 1)], bag[(s + 1):end])
        fit!(rgs, X, y, train_bag, verbose=verbose)
        return scoring_function(rgs, X, y, test_bag)
    end

    firsts = 1:k:((n_folds - 1)*k + 1) # itr of first test_bag index
    seconds = k:k:(n_folds*k)          # itr of ending test_bag index

    if parallel && nworkers() > 1
        if verbose
            println("Distributing cross-validation computation among $(nworkers()) workers.")
        end
        return @parallel vcat for n in 1:n_folds
            [err(firsts[n], seconds[n])]
        end
    else
        errors = Array(Float64, n_folds)
        for n in 1:n_folds
            verbose ? print("fold $n,") : nothing
            errors[n] = err(firsts[n], seconds[n])
        end
        verbose ? println() : nothing
        return errors
    end
    
end

cross_validation_errors(rgs::BaseRegressor, X, y;
      parameters...) = cross_validation_errors(rgs, X, y, 1:length(y); parameters...)

cross_validation_error(rgs::BaseRegressor, X, y, bag;
      parameters...) =  mean(cross_validation_errors(rgs, X, y, bag; parameters...))

cross_validation_error(rgs::BaseRegressor, X, y;
      parameters...) = cross_validation_error(rgs, X, y, 1:length(y); parameters...)

const cv_error = cross_validation_error
const cv_errors = cross_validation_errors
                
"""
Return the p-value for the null hypothesis, `new_regressor is no
better than old regressor` based on test set. Here we say new is
*better* than old if it's predictions are closer to true more often
in the general population
"""
function compare_on_test_set(old::BaseRegressor, new::BaseRegressor, X, y, bag)
    n = length(bag)
    new_wins = sum(abs(predict(new, X, bag) - y[bag]) .< abs(predict(old, X, bag) - y[bag]))
    mu = new_wins/float(n)
    p = Distributions.cdf(Distributions.Binomial(n,mu),n/2)
    return p
end

"""
## `function vary_float_parameter(RegressorType, X, y, bag_train, bag_test;`
## `to_be_varied=?, min=?, max=?, n=20, verbose=true, scoring_function=rms_error, parameters...)`

Returns `(u,v)` where `u = linspace(min,max, n)` (converted to an
array) and `v` is a length `n` vector of rms errors when
`RegressorType` is intantiated (trained) on the data `X`,`y` using row
indices in `bag_train` and tested on the rows with indices in
`bag_test`. The keyword arguments with questionmarks are compulsory.

### Arguments

* RegressorType ... a subtype of BaseRegressor
* X , y ... patterns and responses
* to_be_varied ... name of RegressorType parameter to be varied,  prefixed with `:` to make it a symbol
* min ... minimum value for the varying parameter
* max ... maximum value for the varying parameter
* n ... number of values of the varying parameter
* parameters ... keyword parameters for instantiating RegressorType, *with the parameter to be varied omitted*

### Example

In the following example `RegressorType` only has one keyword parameter:

    vary_float_parameter(NearestNeighbourTreeRegressor, X, y, bag_train, bag_test;
                         to_be_varied=:lambda, min=1, max=100, n=20, verbose=true)`

"""
function vary_float_parameter(RegressorType, X, y, bag_train, bag_test; parameters...)
    parameters = Dict(parameters)
    s = parameters[:to_be_varied] # the symbol of the parameter to be varied
    min = parameters[:min]
    max = parameters[:max]
    delete!(parameters, :to_be_varied)
    delete!(parameters, :min)
    delete!(parameters, :max)
    delete!(parameters, :verbose)
    if :n in keys(parameters)
        n = parameters[:n]
        delete!(parameters, :n)
    else
        n = 20
    end
    if :verbose in keys(parameters)
        verbose = parameters[:verbose]
        delete!(parameters, :verbose)
    else
        verbose = true
    end
    if :scoring_function in keys(parameters)
        scoring_function = parameters[:scoring_function]
        delete!(parameters, :scoring_function)
    else
        scoring_function = rms_error
    end
    errors = Array(Float64, n)
    range=linspace(min, max, n)
    i=1
    for value in range
        parameters[s]=value
        if verbose == true
            println("Training and testing with parameter equal to $(value)")
        end
        rgs = RegressorType(X, y, bag_train; parameters...)
        er = scoring_function(rgs, X, y, bag_test)
        println()
        println("Cross-validation error=$er")
        errors[i]= er
        i += 1
    end

    return Array{Float64,1}(range), errors
end

"""
## `function vary_int_parameter(RegressorType, X, y, bag_train, bag_test;`
## `to_be_varied=?, min=0, max=10, step=1, values=[min:step:max], scoring_function=rms_error, parameters...)`

Returns `(u,v)` where `u = min:step:max` (converted to an array) and
`v` is a corresponding vector of rms errors when `RegressorType` is
intantiated (trained) on the data `X,y` using row indices in
`bag_train` and tested on the rows with indices in `bag_test`, and the
`to_be_varied` parameter takes on values in `u`. The keyword arguments
with questionmarks are compulsory.

### Arguments

* RegressorType ... a subtype of BaseRegressor
* X , y ... patterns and responses
* to_be_varied ... name of RegressorType parameter to be varied,  prefixed with `:` to make it a symbol
* min ... minimum value for the varying parameter
* max ... maximum value for the varying parameter
* step size for increments in the parameter being varied.
* values ... alternatively, a specific iterable of values to be used
* parameters ... keyword parameters for instantiating RegressorType, *with the parameter to be varied omitted*

### Example

    vary_int_parameter(RandomForestRegressor, X, y, bag_train, bag_test;
                     to_be_varied=:max_features, min=1, max=20, step=2, n_trees=100)

"""
function vary_int_parameter(RegressorType, X, y, bag_train, bag_test; parameters...)
    parameters = Dict(parameters)
    s = parameters[:to_be_varied] # the symbol of the parameter to be varied
    delete!(parameters, :to_be_varied)
    @extract_from parameters min 0
    @extract_from parameters max 10
    @extract_from parameters step 1
    @extract_from parameters values collect(min:step:max)
    @extract_from parameters scoring_function rms_error

    n = length(values)
    errors = Array(Float64, n)
    i=1
    for value in values
        parameters[s]=value
        println("Training and testing with parameter equal to $(value)")
        rgs = RegressorType(X, y, bag_train; parameters...)
        er = scoring_function(rgs, X, y, bag_test)
        println()
        println("Error on test = $er")
        errors[i]= er
        i += 1
    end

    return collect(values), errors
end

macro error_versus_ξ(regressor_code, X, y, bag_train, bag_test, range)
    xi = :ξ
    quote
        n = length($(esc(range)))
        i = 1
        errors = Array(Float64, n)
        for val in $(esc(range))
            println("Training and testing with parameter equal to ", val)
            $(esc(xi)) = val
            er = rms_error(fit!($(esc(regressor_code)), $(esc(X)), $(esc(y)), $(esc(bag_train))),
                           $(esc(X)), $(esc(y)), $(esc(bag_test)))
            #            er = scoring_function(rgs, X, y, bag_test)
            println("Error on test set provided=$er \n")
            errors[i]= er
            i += 1
        end
        collect($(esc(range))), errors
    end
end


macro cv_error_versus_ξ(regressor_code, X, y, bag, range, n_folds...)

    xi = :ξ
    k = isempty(n_folds) ? 6 : n_folds[1]
    quote
        n = length($(esc(range)))
        i = 1
        errors = Array(Float64, n)
        for val in $(esc(range))
            println("Training and testing with parameter equal to ", val)
            $(esc(xi)) = val
            er = cross_validation_error($(esc(regressor_code)), $(esc(X)), $(esc(y)),
                                        $(esc(bag)), n_folds=$(esc(k)))
            println("Cross-validation error = $er \n")
            errors[i]= er
            i += 1
        end
        println($k,"-fold cross-validation used.")
        collect($(esc(range))), errors
    end
end


macro cv_errors_versus_ξ(regressor_code, X, y, bag, range, n_folds...)

    xi = :ξ
    k = isempty(n_folds) ? 6 : n_folds[1]
    quote
        n = length($(esc(range)))
        i = 1
        errors = Array(Float64, (n,3)) # for bootstrap C.I. and mean 
        for val in $(esc(range))
            println("Training and testing with parameter equal to ", val)
            $(esc(xi)) = val
            er = cross_validation_errors($(esc(regressor_code)), $(esc(X)), $(esc(y)),
                                         $(esc(bag)), n_folds=$(esc(k)))
            boot = bootstrap_resample_of_mean(er)
            e5 = quantile(boot, 0.05)
            mu = mean(er)
            e95 = quantile(boot, 0.95)
            println("Mean of cross validation errors = $mu \n")
            errors[i,1] = e5
            errors[i,2] = mu
            errors[i,3] = e95
            i += 1
        end
        println($k,"-fold cross-validation used.")
        collect($(esc(range))), errors
    end

end



macro bootstrap_cv_errors_versus_ξ(regressor_code, X, y, bag, range, bootstrap_size, n_folds...)

    xi = :ξ
    k = isempty(n_folds) ? 6 : n_folds[1]
    quote
        n = length($(esc(range)))
        i = 1
        errors = Array(Float64, (n,bootstrap_size)) # for bootstrap C.I. and mean 
        for val in $(esc(range))
            println("Training and testing with parameter equal to ", val)
            $(esc(xi)) = val
            er = cross_validation_errors($(esc(regressor_code)), $(esc(X)), $(esc(y)),
                                         $(esc(bag)), n_folds=$(esc(k)))
            boot = bootstrap_resample_of_mean(er; n=bootstrap_size)
            mu = mean(er)
            println("Mean of cross validation errors = $mu \n")
            for j in 1:bootstrap_size
                errors[i,j] = boot[j]
            end
            i += 1
        end
        println($k,"-fold cross-validation used.")
        collect($(esc(range))), errors
    end
end

function learning_curve(rgs::BaseRegressor, X, y, train, validate, values; parameters...)
    parameters = Dict(parameters)
    @extract_from parameters restart true
    @extract_from parameters verbose true

    values = collect(values)
    sort!(values)
    
    if restart
        rgs.n_iter=0
    end
    
    n_iters = Float64[]
    errors = Float64[]

    filter!(x -> (x > rgs.n_iter), values)
    verbose ? println("Iteration values not yet reached: ", values) : nothing
    if isempty(values)
        return n_iters, errors
    end
    
    n_add = values[1] - rgs.n_iter
    if restart
        n_temp =rgs.n
        rgs.n = n_add
#        @dbg parameters
        fit!(rgs, X, y, train; verbose=verbose, parameters...)
        rgs.n = n_temp
    else
        add!(rgs, X, y, train; n=n_add, parameters...)
    end
    push!(n_iters, rgs.n_iter)
    push!(errors, rms_error(rgs, X, y, validate))

    filter!(x -> (x > rgs.n_iter), values)
    verbose ? println("Iteration values not yet reached: ", values) : nothing

    while !isempty(values)
        n_add = values[1] - rgs.n_iter
        add!(rgs, X, y, train; n=n_add, verbose=verbose, parameters...)
        push!(n_iters, rgs.n_iter)
        push!(errors, rms_error(rgs, X, y, validate))
        filter!(x -> (x > rgs.n_iter), values)
        verbose ? println("Iteration values not yet reached: ", values) : nothing
    end
    return n_iters, errors

end
                            
end # of module
