__precompile__()
# August 2017
module Regressors

export TreeRegressor, feature_importance
export BaggedRegressor, ConstantRegressor
export LoessRegressor, LinearRegressor #, LinearLogRegressor, LocallyConstantRegressor
export ElasticNetRegressor, StackedRegressor, coefs #, ElasticNetRegressorPath
export XGBoostRegressor, cv, parse_cv_output
export row, show, prefit!, get_params, set_params!, add!
export rms_error, rmsl_error, rms, @provide_parameter_interface

# extended:
export predict, fit!, clone, get_params, set_params!

import ADBUtilities: @colon, @extract_from, argmin, argmax, keys_ordered_by_values, @dbg
import TreeCollections: countmap, Small, DataTable, row, IntegerSet, Node, is_stump, is_leaf
import TreeCollections: has_left, has_right, make_leftchild!, make_rightchild!
import TreeCollections: unite!, child, FrameToTableScheme
import ScikitLearnBase
import ScikitLearnBase: BaseRegressor, @declare_hyperparameters, simple_get_params, simple_set_params!
import DataFrames: DataFrame, Formula, isna 
import Loess
import GLM
import Lasso
import XGBoost
import UnicodePlots
import Distributions

# to be extended:
import Base: show, showall
import ScikitLearnBase: predict, fit!, clone, get_params, set_params!


global const MAX_HEIGHT = 1000 # used in single tree regularization;
                             # MAX_HEIGHT = Inf ideal but slower

"""

This is identical to ScikitLearn.@declare_hyperparameters except that
it does not provide the clone method, which has performance issues.

"""
macro provide_parameter_interface(estimator_type, params)
    :(begin
        $ScikitLearnBase.get_params(estimator::$(esc(estimator_type));
                                    deep=true) =
            simple_get_params(estimator, $(esc(params)))
        $ScikitLearnBase.set_params!(estimator::$(esc(estimator_type));
                                    new_params...) =
            simple_set_params!(estimator, new_params;param_names=$(esc(params)))
    end)
end

# fit! methods in this module take a fourth non-keyword argument,
# `bag`, which is a list of pattern indices for which the fit is
# restricted (useful for Breimann-style bagging
# ensembles, cross-validation, etc). Since the ScikitLearn API uses
# bagless versions, here they are:

fit!{T<:BaseRegressor}(rgs::T, X, y; params...) = fit!(rgs,X,y,1:length(y); params...)

# The following are fall-back extensions of `predict` methods defined
# for particular regressors below, to broadcast versions:
# predict(rgs::BaseRegressor, X::DataTable) = [predict(rgs, X[i,:]) for i in 1:size(X, 1)]
# predict(rgs::BaseRegressor, X::DataTable, bag) = [predict(rgs, X[i,:]) for i in bag]
row(X::DataFrame, i) = [X[i,j] for j in 1:size(X, 2)]
row{T}(X::Array{T, 2}, i) = X[i,:]
# predict(rgs::BaseRegressor, X::DataFrame) = [predict(rgs, row(X,i)) for i in 1:size(X, 1)]
# predict(rgs::BaseRegressor, X::DataFrame, bag) = [predict(rgs, row(X,i)) for i in bag]
# predict{T<:Real}(rgs::BaseRegressor, X::Array{T, 2}) = [predict(rgs, X[i,:]) for i in 1:size(X, 1)]
# predict{T<:Real}(rgs::BaseRegressor, X::Array{T, 2}, bag) = [predict(rgs, X[i,:]) for i in bag]

showall(rgs::BaseRegressor) = showall(STDOUT, rgs)


##################
# Loss functions #
##################

function rms(y, yhat, bag=[])
    n = length(y)
    length(yhat) == n || throw(DimensionMismatch())
    if isempty(bag)
        bag = 1:n
    end
    ret = 0.0
    for i in bag
        ret += (y[i] - yhat[i])^2
    end
    return sqrt(ret/n)
end

function rmsl(y, yhat, bag=[])
    n = length(y)
    length(yhat) == n || throw(DimensionMismatch())
    if isempty(bag)
        bag = 1:n
    end
    ret = 0.0
    for i in bag
        ret += (log(y[i]) - log(yhat[i]))^2
    end
    return sqrt(ret/n)
end

###################
# Error functions #
###################

rms_error(rgs::BaseRegressor, X, y, bag) = rms(predict(rgs, X, bag), y[bag])
rms_error(rgs::BaseRegressor, X,y) = rms(predict(rgs, X), y)
rmsl_error(rgs::BaseRegressor, X, y, bag) = rmsl(predict(rgs, X, bag), y[bag])
rmsl_error(rgs::BaseRegressor, X,y) = rmsl(predict(rgs, X), y)

function rms_error(regressor::BaseRegressor, x::Vector, y, bag)

    warn("A depreciated version of rms_error is being used.")
    
    n_patterns = length(bag)
    if n_patterns == 0
        error("A bag cannot have zero length D")
    end
    total = 0.0
    for i in bag
        y_hat = predict(regressor, x[i])
        total += (y_hat - y[i])^2
    end
    return sqrt(total/n_patterns)
end

####################################################################
# Functions for updating means and variance via recursive formulas #
####################################################################

# (see
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# and http://tullo.ch/articles/speeding-up-decision-tree-training/)

"""
# `function mean_and_ss_after_add(mean, ss, n, x)`

Returns the mean, and the sum-of-square deviations from the mean, of
`n+1` numbers, given the corresponding two quantities for the first
`n` numbers (the inputs `mean` and `ss`) and the value of the `n+1`th
number, `x`.
"""
function mean_and_ss_after_add(mean, ss, n, x)
    n<0 ? throw(DomainError) : nothing
    mean_new = (n*mean + x)/(n + 1)
    ss_new = ss + (x - mean_new)*(x - mean)
    return mean_new, ss_new
end

"""
# `function mean_and_ss_after_omit(mean, ss, n, x)`

Given `n` numbers, their mean `mean` and sum-of-square deviations from
the mean `ss`, this function returns the new mean and corresponding
sum-of-square deviations of the same numbers when one of the numbers,
`x` is omitted from the list.

"""
function mean_and_ss_after_omit(mean, ss, n, x)
    n <= 1 ? throw(DomainError) : nothing
    mean_new = (n*mean - x)/(n-1)
    ss_new = ss - (x - mean)*(x - mean_new)
    return mean_new, ss_new
end

#################
# TreeRegressor #
#################

immutable NodeData
    feature::Int
    kind::Int8   # 0: root, 1: ordinal, 2: categorical, 3: leaf 
    r::Float64   # A *threshold*, *float repr. of integer subset*, or
                 # *prediction*, according to kind above
end

typealias RegressorNode Node{NodeData}

function should_split_left(pattern, node::RegressorNode)
    data = node.data
    j, r, kind = data.feature, data.r, data.kind
    if kind == 1     # ordinal
        return pattern[j] <= r 
    elseif kind == 2 # categorical
        small = round(Small, pattern[j])
        return small in round(IntegerSet, r)
    else
        throw(Base.error("Expecting an ordinal or categorical node here."))
    end
end


"""
## `mutable Regressors.TreeRegressor`

### Method calls

    TreeRegressor()          # unfitted regressor (default hyperparameters)
    TreeRegressor(X, y)      # regressor trained on all patterns, `X` input, `y` target
    TreeRegressor(X, y, bag) # regressor trained on patterns with indices in `bag`

In any call above keyword arguments can be supplied to change hyperparameters from defaults.

### Training in stages

    rgs = TreeRegressor()  # or with hyperparameters set by keyword arguments
    fit!(rgs, X, y)        # train `rgs` on all patterns
    fit!(rgs, X, y, bag)   # train on patterns in `bag`

Supplying `verbose=true` in either `fit!` method turns on reporting to STDOUT.

### Method argument types

  Argument  | Type              | Description           
------------|-------------------|-----------------------------------------------------------
 `X`        | `DataTable`       | Input patterns (one `DataTable` column per feature)
 `y`        | `Vector{Float64}` | Output responses
 `bag`      |  eltype `Int`     | Iterator returning row indices of patterns (rows) to be considered for training
                       
### Hyperparameters (static):

- `max_features=0`: Number of features randomly selected at each node to
                                  determine splitting criterion selection (integer).
                                  If 0 (default) then redefined as `n_features=length(X)`
- `min_patterns_split=2`: Minimum number of patterns at node to consider split (integer). 

- `penalty=0` (range, [0,1]): Float between 0 and 1. The gain afforded by new features
      is penalized by mulitplying by the factor `1 - penalty` before being
      compared with the gain afforded by previously selected features.

- `extreme=false`: If true then the split of each feature considered is uniformly random rather than optimal.                              
- `regularization=0.0` (range, [0,1)): regularization in which predictions 
    are a weighted sum of predictions at the leaf and its "nearest neighbours`
     as defined by the pattern. 

- `cutoff=0` (range, [`max_features`, `size(X, 2)`]): features with
       indices above `cutoff` are ignored completely. If zero then set to
       maximum feature index.

### Hyperparameters (dynamic, ie affected by fitting):

- `popularity_given_feature=Dict{Int,Int}()`: A dictionary keyed on
          feature index. Each feature whose index is a key is not
          penalised when considering new node splits (see `penalty`
          above). Whenever a feature with index `j` is chosen as the
          basis of a decision in a split, then the number of patterns
          affected by the decision is added to
          `popularity_given_feature[j]`. If `j` is not already a key,
          then it is first added and `popularity_given_feature[j]`
          initialized to zero.

### Post-fitted parameters:

- `names`: the names of the features

### Return value:

A `TreeRegressor` object. 

"""        
type TreeRegressor <: BaseRegressor
    
    # hyperparameters:
    max_features::Int
    min_patterns_split::Int
    penalty::Float64
    extreme::Bool
    regularization::Float64
    cutoff::Int 
    popularity_given_feature::Dict{Int, Int} # dynamic 
    
    # model:
    model::RegressorNode # The stump node of the decision tree
    
    # postfit-parameters:
    names::Vector{Symbol}
    fitted::Bool

    # default constructor initializes hyperparameters only:
    function TreeRegressor(max_features::Int, min_patterns_split::Int,
                           penalty::Float64, extreme::Bool, regularization::Float64,
                           cutoff::Int, popularity_given_feature::Dict{Int,Int})
        if cutoff !=0 && cutoff < max_features
            throw(Base.warn("cutoff < max_features"))
        end
        rgs = new(max_features, min_patterns_split, penalty, extreme, regularization,
                  cutoff, popularity_given_feature)
        rgs.fitted = false
        return rgs
    end
    
end

function show(stream::IO, rgs::TreeRegressor)
    tail(n) = "..."*string(n)[end-3:end]
    prefix = rgs.fitted ? "" : "unfitted "
    print(stream, prefix, "TreeRegressor@$(tail(hash(rgs)))")
end

function feature_importance(rgs::TreeRegressor)
    rgs.fitted || error("Operation not permitted on $rgs.")

    ret = Dict{Symbol, Float64}()
    kys = keys(rgs.popularity_given_feature)
    N = sum(rgs.popularity_given_feature[j] for j in kys)
    for j in kys
        ret[rgs.names[j]] = rgs.popularity_given_feature[j]/N
    end

    return ret
end

function showall(stream::IO, rgs::TreeRegressor)
    show(stream, rgs); println("\n  Hyperparameters:")
    display((get_params(rgs)))
    x = Symbol[]
    y = Float64[]
    if rgs.fitted
        imp_given_name = feature_importance(rgs)            
        for name in reverse(keys_ordered_by_values(imp_given_name))
            push!(x, name)
            push!(y, round(Int, 1000*imp_given_name[name])/1000)
        end
        UnicodePlots.show(stream, UnicodePlots.barplot(x,y,title="Feature importance"))
    end
end

# lazy keyword constructors:
## without fitting:
function TreeRegressor(;max_features::Int=0, min_patterns_split::Int=2,
              penalty::Float64=0.0, extreme::Bool=false,
                       regularization::Float64=0.0, cutoff::Int=0,
                       popularity_given_feature::Dict{Int,Int}=Dict{Int,Int}())
    return TreeRegressor(max_features, min_patterns_split, penalty, extreme, regularization,
                         cutoff, popularity_given_feature)
end

## with immediate fitting:
TreeRegressor(X::DataTable,
              y::Vector{Float64},
              bag; parameters...) = fit!(TreeRegressor(;parameters...), X, y, bag)

## with immediate fitting (bagless version):
TreeRegressor(X::DataTable,
              y::Vector{Float64}; parameters...) = fit!(TreeRegressor(;parameters...), X, y)

# provide `set_params!`, `get_params`, `clone`:
@provide_parameter_interface TreeRegressor [:max_features,
                                            :min_patterns_split, :penalty, :extreme,
                                            :regularization, :cutoff, :popularity_given_feature]

# but rewrite clone to be faster:
clone(rgs::TreeRegressor) = TreeRegressor(rgs.max_features, rgs.min_patterns_split,
                                          rgs.penalty, rgs.extreme, rgs.regularization,
                                          rgs.cutoff, copy(rgs.popularity_given_feature))

function fit!(rgs::TreeRegressor,
              X::DataTable,
              y::Vector{Float64},
              bag; verbose=false)
    
    n_patterns = length(y)
    size(X, 1) != n_patterns ? throw(BoundsError) : nothing
    n_features = size(X, 2)

    max_features = (rgs.max_features == 0 ? n_features : rgs.max_features)
    cutoff = (rgs.cutoff == 0? n_features : rgs.cutoff)

    rgs.names = X.names

    # The initial set of penalized feature indices:
    F = Set(keys(rgs.popularity_given_feature))

    # A vector representing the popularity of each feature:
    popularity = zeros(Float64, cutoff)
    for j in F
        popularity[j] = rgs.popularity_given_feature[j]
    end
    
    # Create a root node to get started. Its unique child is the true
    # stump of the decision tree:
    root = RegressorNode(NodeData(0, 0, 0.0)) 

    function split_on_categorical(j, bag, no_split_error)

        # For the case `j` is the index of a *categorical* feature.

        # Returns `(gain, left_values)` where gain is the lowering of
        # the error (sum of square deviations) obtained for the best
        # split based on sample inputs X[j] with corresponding target
        # values y; `left_values` is a floating point representation
        # of the set of values of x for splitting left in the optimal
        # split. If the feature is constant (takes on a single value)
        # then no split can improve the error and gain=0. 
    
        # Note that only inputs with indices in the iterator `bag` are
        # considered.
    
        if rgs.extreme
            vals = collect(Set([round(Small, X.raw[i,j]) for i in bag]))
            n_vals = length(vals)
            if n_vals == 1
                return 0.0, 0.0 # X[j] is constant in this bag, so no gain
            end
            n_select = rand(1:(n_vals - 1))
            left_selection = Distributions.sample(vals, n_select; replace=false)
            left_values_encoded = IntegerSet(left_selection) 
            left_count = 0
            right_count = 0
            left_mean = 0.0
            right_mean = 0.0
            for i in bag
                v = round(Small, X.raw[i,j])
                if v in left_selection
                    left_count += 1
                    left_mean += y[i]
                else
                    right_count += 1
                    right_mean += y[i]
                end
            end
            left_mean = left_mean/left_count
            right_mean = right_mean/right_count
            
            # Calcluate the split error, denoted `err` (the sum of squares of
            # deviations of target values from the relevant mean - namely
            # left or right
            err = 0.0
            for i in bag
                v = round(Small, X.raw[i,j])
                if v in left_selection
                    err += (left_mean - y[i])^2
                else
                    err += (right_mean - y[i])^2
                end
            end
            gain = no_split_error - err
            left_values = Float64(IntegerSet(left_selection))
            return gain, left_values
        end

        # Non-extreme case:
        
        # 1. Determine a Vector{Small} of values taken by `X[j]`,
        # called `values` and order it according to the mean values of
        # y:

        count = Dict{Small,Int}()   # counts of `X[j]` values, keyed on
                                    # values of `X[j]`.
        mu = Dict{Small,Float64}() # mean values of target `y` keyed on
                                    # values of `X[j]`, initially just the
                                    # unnormalized sum of `y` values.
        for i in bag
            value = round(Small, X.raw[i,j])
            if !haskey(count, value)
                count[value] = 1
                mu[value] = y[i]
            else
                count[value] += 1
                mu[value] += y[i]
            end
        end
        if length(count) == 1          # feature is constant for data in bag
            return 0.0, 0.0            # so no point in splitting;
                                       # second return value is irrelevant
        end
    
        # normalize sums to get the means:
        for v in keys(count)
            mu[v] = mu[v]/count[v]
        end
        vals = keys_ordered_by_values(mu) # vals is a Vector{Small}

        # 2. Let the "kth split" correspond to the left-values vals[1],
        # vals[2], ... vals[k] (so that the last split is no
        # split). 
        
        n_vals = length(vals)

        # do first split outside loop handling others:
        left_count = 0
        right_count = 0
        left_mean = 0.0
        right_mean = 0.0
        for i in bag
            v = round(Small, X.raw[i,j])
            if v == vals[1]
                left_count += 1
                left_mean += y[i]
            else
                right_count += 1
                right_mean += y[i]
            end
        end
        left_mean = left_mean/left_count
        right_mean = right_mean/right_count
        left_ss = 0.0
        right_ss = 0.0
        for i in bag
            v = round(Small, X.raw[i,j])
            if v == vals[1]
                left_ss += (left_mean - y[i])^2
            else
                right_ss += (right_mean - y[i])^2
            end
        end
        
        error = left_ss + right_ss
        position = 1 # updated if error is improved in loop below

        for k in 2:(n_vals - 1)

            # Update the means and sum-of-square deviations:
            for i in bag
                if round(Small, X.raw[i,j]) == vals[k]
                    left_mean, left_ss = mean_and_ss_after_add(left_mean, left_ss, left_count, y[i])
                    left_count += 1
                    right_mean, right_ss = mean_and_ss_after_omit(right_mean, right_ss, right_count, y[i])
                    right_count += -1
                end
            end

            # Calcluate the kth split error:
            err = left_ss + right_ss

            if err < error
                error = err
                position = k
            end
        end
    
        gain = no_split_error - error
        
        # println("pos error ",position," ", err)
        left_values = Float64(IntegerSet(vals[1:position]))
        return gain, left_values
        
    end # of function `split_on_categorical`

    function split_on_ordinal(j, bag, no_split_error)

        # For case `j` is the index of an *ordinal* feature.
    
        # Returns (gain, threshold) where `gain` is the lowering of
        # the error (sum of square deviations) obtained for the best split
        # based on sample inputs `X[j]` with corresponding target values
        # y; `threshold` is the maximum value the feature for a left split
        # in the best case. If the feature is constant (takes on a single
        # value) then no split can improve the error and gain =
        # 0.0.
        
        # Note that only inputs with indices in the iterator `bag` are
        # considered.

        # 1. Determine a Vector{Float64} of values taken by X[j], called `vals` below:
        val_set = Set{Float64}()
        for i in bag
            push!(val_set, X.raw[i,j])
        end
        vals = collect(val_set)
        sort!(vals)
        n_vals = length(vals)
        if n_vals == 1     # feature is constant for data in bag
            return 0.0, 0.0        # so no point in splitting; second
                                   # value irrelevant
        end

        if rgs.extreme
            min_val = minimum(vals)
            max_val = maximum(vals)
            threshold = min_val + rand()*(max_val - min_val)

            # Calculate the left and right mean values of target
            left_count = 0
            right_count = 0
            left_mean = 0.0
            right_mean = 0.0
            for i in bag
                v = X.raw[i,j]
#               println("threshold=$threshold; X.raw[i,j]=$(X.raw[i,j])")
                if v <= threshold
                    left_count += 1
                    left_mean += y[i]
                else
                    right_count += 1
                    right_mean += y[i]
                end
            end    
            left_mean = left_mean/left_count
            right_mean = right_mean/right_count

            # Calcluate the split error, denoted `err` (the sum of
            # squares of deviations of target values from the relevant
            # mean - namely left or right)
            err = 0.0
            for i in bag
                v = X.raw[i,j]
                if v <= threshold
                    err += (left_mean - y[i])^2
                else
                    err += (right_mean - y[i])^2
                end
            end    
            gain = no_split_error - err
            return  gain, threshold
        end
        
        # Non-extreme case:            

        # 2. Let the "jth split" correspond to threshold = vals[j], (so
        # that the last split is no split). The error for the jth split
        # will be error[j]. We calculate these errors now:
    
        
        # println("len = $(length(vals))")

        # we do the first split outside of the loop that considering the
        # others because we will use recursive formulas to update
        # means and sum-of-square deviations (for speed enhancement)

        # mean and ss (sum-of-square deviations) for first split:
        left_mean = 0.0 
        left_count = 0  
        right_mean = 0.0 
        right_count = 0  
        for i in bag
            if X.raw[i,j] <= vals[1]
                left_mean += y[i]
                left_count += 1
            else
                right_mean += y[i]
                right_count += 1
            end
        end
        left_mean = left_mean/left_count
        right_mean = right_mean/right_count
        left_ss = 0.0
        right_ss = 0.0
        for i in bag
            if X.raw[i,j] == vals[1]
                left_ss += (y[i] - left_mean)^2
            else
                right_ss += (y[i] - right_mean)^2
            end
        end

        # error for first split:
        error = left_ss + right_ss
        position = 1 # its position, to be updated if better split found
        
        for k in 2:(n_vals - 1)

            # Update the means and sum-of-square deviations:
            for i in bag
                x = X.raw[i,j]
                if x == vals[k] # (x > vals[k-1]) && (x <vals[k])
                    left_mean, left_ss = mean_and_ss_after_add(left_mean, left_ss, left_count, y[i])
                    left_count += 1
                    right_mean, right_ss = mean_and_ss_after_omit(right_mean, right_ss, right_count, y[i])
                    right_count += -1
                end
            end

            # Calcluate the kth split error:
            err = left_ss + right_ss

            # Note value and split position if there is improvement
            if err < error
                error = err
                position = k
            end
        end

        gain = no_split_error - error
        
        threshold = 0.5*(vals[position] + vals[position + 1])
        return gain, threshold
        
    end # of function `split_on_ordinal` (method 2)

    function attempt_split(bag, F, parent, gender, no_split_error)

        # Returns split_failed, bag_left, bag_right as follows:
      
        # Computes the error for the best split on each feature within
        # a random sample of features of size `max_features` (set in
        # `TreeRegressor()`) and, if splitting improves the error (postive
        # `gain`), create a new splitting node of appropriate type and
        # connect to `parent`. In evaluating what is the *best* feature,
        # those features not previously selected have their gains
        # penalized by muliplying by `rgs.penalty`.

        # `split_failed` is true if no new node is created.

        # `bag_left` and `bag_right` are splits of `bag` based on the
        # optimum splitting found, or are empty if `split_failed`.

        # Note that splitting is based only on patterns with indices in
        # `bag`.

        if max_features == n_features
            feature_sample_indices = collect(1:n_features)
        else
            feature_sample_indices = Distributions.sample(1:cutoff,
                                                          max_features; replace=false)
        end

        max_gain = -Inf # max gain so far
        opt_index = 0 # index of `feature_sample_indices`
                             # delivering feature index for max_gain
                             # (mythical at present)
        opt_crit = 0.0       # criterion delivering that max_gain

        # println("sample indices = $feature_sample_indices") 

        for i in 1:max_features
            j = feature_sample_indices[i]

            if X.scheme.is_ordinal[j]
                gain, crit = split_on_ordinal(j, bag, no_split_error)
            else
                gain, crit = split_on_categorical(j, bag, no_split_error)
            end

            if !(j in F) && rgs.penalty != 0.0
                gain = (1 - rgs.penalty)*gain
            end

            if gain > max_gain
                max_gain = gain
                opt_crit = crit
                opt_index = i
            end
        end
    
        # If no gain, return to calling function with `split_failed=true`:
        if max_gain == 0.0
            yvals = [y[i] for i in bag]
            return true, Int[], Int[]
        end

        # Otherwise, create a new node with a splitting criterion based on
        # the optimal feature and unite with `parent`:
        j = feature_sample_indices[opt_index] # feature with minimum error

        if X.scheme.is_ordinal[j]
            data = NodeData(j, 1, opt_crit)
        else
            data = NodeData(j, 2, opt_crit)
        end
        baby = RegressorNode(data) # new decision node
        unite!(baby, parent, gender)
    
        # Update the set of unpenalised features and the feature popularity vector:
        push!(F,j)
        popularity[j] += length(bag)
    
        # Split `bag` accordingly:
        bag_left  = Int[]
        bag_right = Int[]
        for i in bag
            if should_split_left(X[i,:], baby)
                push!(bag_left, i)
            else
                push!(bag_right, i)
            end
        end
        
        # Return the bag splits with decleration `split_failed=false`:
        return false, bag_left, bag_right 

    end # of function `attempt_split`

    function grow(bag,                  # Patterns to be considerd for splitting
                 F,                     # Current set of unpenalised features
                 parent::RegressorNode, # Node to which any child will be
                                        # connected in successful call to
                                        # `attempt_split` above.
                 gender)                # Determines how any child
                                        # will be connected to parent
                                        # (as left, right, or if this
                                        # is the first call to grow,
                                        # androgynous)

        # Recursive function to grow the decision tree. Has no return
        # value but generally alters 2nd and 3rd arguments.

        n_patterns = length(bag)
            
        # Compute mean of targets in bag:
        target_mean = 0.0
        for i in bag
            target_mean += y[i]
        end
        target_mean = target_mean/n_patterns

        # Do not split node if insufficient samples, but create and
        # connect leaf node with above target_mean as prediction:
        if n_patterns < rgs.min_patterns_split
        #   println("insufficient patterns")
            leaf = RegressorNode(NodeData(0, 3, target_mean))
            unite!(leaf, parent, gender)
        #   println("Leaf born, n_samples = $n_patterns, target prediction = $target_mean")
            return
        end

        # Find sum of square deviations for targets in bag:
        no_split_error = 0.0
        for i in bag
            no_split_error += (y[i] - target_mean)^2
        end

        # If the following is succesful, it creates a child and connects
        # it to `parent`.  In that case it also returns new bags according
        # to the optimal splitting criterion computed there (empty bags
        # otherwise) and updates `F` and `popularity`:
        split_failed, bag_left, bag_right = attempt_split(bag, F, parent, gender,
                                                          no_split_error)

        # If split makes no (significant) difference, then create and
        # connect prediction node:
        if split_failed
            leaf = RegressorNode(NodeData(0, 3, target_mean))
            unite!(leaf, parent, gender)
            # println("Leaf born, n_samples = $n_patterns, target prediction = $target_mean")
            return
        end
        
        # Otherwise continue growing branches left and right:
        baby = child(parent, gender)
        F_left =  copy(F)
        F_right = copy(F)
        grow(bag_left, F_left, baby, 1)   # grow a left branch
        grow(bag_right, F_right, baby, 2) # grow a right branch
        # F = union(F_left, F_right)

        return
    end

    grow(bag, F, root, 0) # 0 means `root` is to be androgynous, meaning
                          # `root.left = root.right`.

    for j in 1:cutoff
        pop = popularity[j]
        if pop != 0.0
            rgs.popularity_given_feature[j] = pop
        end
    end
    
    rgs.model = root.left
    rgs.fitted = true
    return rgs

end

function predict(node::RegressorNode, pattern::Vector)
    while !is_leaf(node)
        if should_split_left(pattern, node)
            node = node.left
        else
            node = node.right
        end
    end
    return node.data.r
end

function predict(rgs::TreeRegressor, X::DataTable, bag=[])

    size(X, 2) == length(rgs.names) || throw(DimensionMismatch)

    rgs.fitted || throw(Base.error("Attempting to predict using $rgs."))

    if isempty(bag)
        bag = 1:X.nrows
    end
    
    if rgs.regularization == 0.0
        return [predict(rgs.model, X[i,:]) for i in bag]
    else
        tree = rgs.model
        lambda = rgs.regularization

        ret = Array(Float64, length(bag))
        k = 1 # counter for index of `ret` (different from bag index)
        for i in bag

            pattern = X[i,:]
            
            # Pass pattern down tree from top (stump), recording branching on way
            branchings = Char[] # ['l','r','r', etc]
            node = tree
            while !is_leaf(node)
                if should_split_left(pattern, node)
                    push!(branchings, 'l')
                    node = node.left
                else
                    push!(branchings, 'r')
                    node = node.right
                end
            end
            depth = length(branchings)
            
            # Passing (partway) back up the tree, collect and sum predictions of
            # nearby leaves each weighted by "distance" away:
            prediction = node.data.r
            height = min(depth, MAX_HEIGHT) 
            for h in 1:height
                node = node.parent
                if branchings[depth + 1 - h] == 'l'
                    node = node.right
                else
                    node = node.left
                end
                prediction += (lambda^h)*predict(node, pattern)
                node = node.parent
            end
            
            # normalize the summed prediction:
            ret[k] = prediction*(1-lambda)/(1-lambda^(height+1))
            k += 1
            
        end
        return ret
        
    end    

end

#####################
# ConstantRegressor #
#####################

type ConstantRegressor <: BaseRegressor
    model::Float64 # the constant prediction
    fitted::Bool
    ConstantRegressor()=new()
end

function show(stream::IO, rgs::ConstantRegressor)
    rgs.fitted ? print(stream, "ConstantRegressor($(rgs.model))") :
                      print(stream, "unfitted ConstantRegressor()")
end

# lazy constructors:
## with immediate fitting:
ConstantRegressor(X, y, bag) = fit!(ConstantRegressor(), X, y, bag)

## with immediate fitting (bagless version):
ConstantRegressor(X, y) = fit!(ConstantRegressor(), X, y)

# provide `set_params!`, `get_params`, `clone` 
@declare_hyperparameters ConstantRegressor Vector{Symbol}()

function fit!(rgs::ConstantRegressor, X, y, bag; verbose=false)
    rgs.model = mean([y[i] for i in bag])
    rgs.fitted = true
    return rgs
end

function predict(rgs::ConstantRegressor, X, bag=[])
    rgs.fitted || error("Attempting to predict using $rgs.")
    if isempty(bag)
        bag = 1:X.nrows
    end
    return Float64[rgs.model for i in bag]
end

###################
# BaggedRegressor #
###################

"""

# `Regressors.BaggedRegressor(X, y, bag;`
# ` atom=TreeRegressor(), n=20, bagging=1.0, parallel=false, verbose=true)`

An ensemble regression model providing for pattern bagging and partial
pattern bagging in the training of individual regressors.

## Arguments:

  Argument  | Type              | Description           
------------|-------------------|-----------------------------------------------------------
 `X`        | `DataTable`       | Input patterns (one `DataTable` column per feature)
 `y`        | `Vector{Float64}` | Output responses
 `bag`      |  eltype `Int`     | Iterator returning row indices of patterns (rows) to be considered for training
                        
The argument `bag` can be omitted, in which case it defaults to all
pattern indices.  All three arguments can be omitted, in which case a
call to `fit!` is needed to train the model, as in:

    julia> rgs = BaggedRegressor(atom=TreeRegressor(max_features=3), n =100)
    julia> fit!(rgs, X, y) # or fit!(rgs, X, y, bag)
    julia> predict(rgs, X[1:1,:])
    12.34
    julia> full_predictions = predict(rgs, X, bag_test)
                        
The ensemble of models can be added to buy a call to `add!`, as in

    julia> add!(rgs, X, y, n=200)

If the keyword argument `n` is omitted, then `rgs.n` is used in its
place. The total number size of the ensemble is stored as
`rgs.n_iter`. See also `Validation.learning_curve`.

## Optional keyword arguments:

The `atom` is any regressor (object of type `BaseRegressor`) which is
cloned and trained `n` times to build the ensemble. The number of
parallel processors used if `parallel=true` is whatever a call to
`nworkers()` provides. Partial bagging is provided on a scale of 0 to
1 ("1" is standard bootstrap-bagging).

"""
type BaggedRegressor{T<:BaseRegressor} <: BaseRegressor

    # hyperparameters:
    atom::T
    n::Int # number of models created in call to fit!
    bagging::Float64 # between 0 and 1

    # model:
    model::Vector{T} # The stump node of the decision tree
    
    # postfit-parameters:
    n_iter::Int # extra models can be added with add!(rgs,X,y,n=100)
    
    # inner constructor initializes hyperparameters only:
    function BaggedRegressor(atom::T, n::Int, bagging::Float64)
        rgs = new(atom, n, bagging)
        rgs.n_iter = 0
        return rgs
    end

end

# Inner constructors BaggedRegressor{T}(...) must have {T} explicitly appearing, so:
BaggedRegressor{T}(atom::T, n::Int, bagging) = BaggedRegressor{T}(atom,n,bagging)

function show{T}(stream::IO, rgs::BaggedRegressor{T})
    tail(n) = "..."*string(n)[end-3:end]
    prefix = rgs.n_iter > 0 ? "" : "unfitted "
    print(stream, prefix, "BaggedRegressor{$T}@$(tail(hash(rgs)))")
end

function showall(stream::IO, rgs::BaggedRegressor)
    show(stream, rgs); println(stream, "\n Hyperparameters:")
    display(get_params(rgs))
    println(stream, "Current size of ensemble: \n  :n_iter => $(rgs.n_iter)")
end

# lazy keyword constructors:
## without fitting:
BaggedRegressor(;atom::BaseRegressor=TreeRegressor(), n::Int=20,
                bagging=1.0) = BaggedRegressor(atom, n, Float64(bagging))
                              
## with immediate fitting:
function BaggedRegressor(X, y, bag; params...)
    parameters = Dict(params)
    @extract_from parameters parallel false
    @extract_from parameters verbose true
    return fit!(BaggedRegressor(;parameters...), X, y, bag; parallel=parallel, verbose=verbose)
end

## with immediate fitting (bagless version):
BaggedRegressor(X, y; parameters...) = BaggedRegressor(X,y,1:length(y); parameters...)

# provide `set_params!`, `get_params`
@provide_parameter_interface BaggedRegressor [:atom, :n, :bagging]

clone(rgs::BaggedRegressor) = BaggedRegressor(rgs.atom, rgs.n, rgs.bagging)

# function +{T<:BaseRegressor}(rgs1::BaggedRegressor{T}, rgs2::BaggedRegressor{T})
#     return BaggedRegressor(vcat(rgs1.ensemble, rgs2.ensemble), rgs1.parameters)
# end
   
function fit!{T}(rgs::BaggedRegressor{T}, X, y, fullbag; parallel::Bool = false, verbose::Bool = true)

    n_patterns = length(fullbag)

    function ensemble(n)
        ensemble = Array(T, n)
    
        # initialize random number generator:
        srand((round(Int,time()*1000000)))

        if verbose
            println("")
            print("Now cloning and fitting regressor number: ")
        end
        for i in 1:n
            verbose ? (print("$i,"); flush(STDOUT)) : nothing
            indices = Set(1:n_patterns)
            bag = Int[]
            for k = 1:n_patterns
                j = Distributions.sample(collect(indices))
                push!(bag, fullbag[j])
                if rand() > rgs.bagging
                    delete!(indices, j)
                end
            end
            ensemble[i] = clone(rgs.atom)
            fit!(ensemble[i], X, y, bag)
        end
        verbose ? println() : nothing
        return ensemble
    end

    if nprocs() == 1 || parallel == false
        rgs.model = ensemble(rgs.n)
    else
        verbose ? println("Ensemble-building in parallel on $(nworkers()) processors.") : nothing
        chunk_size = div(rgs.n, nworkers())
        left_over = mod(rgs.n, nworkers())
        grand_ensemble =  @parallel (vcat) for i = 1:nworkers()
            if i != nworkers()
                ensemble(chunk_size)
            else
                ensemble(chunk_size + left_over)
            end
        end
        rgs.model = grand_ensemble
    end
    rgs.n_iter = rgs.n
    return rgs
end

fit!(rgs::BaggedRegressor, X, y; parameters...) = fit!(rgs, X, y, 1:length(y); parameters...)

function add!(rgs::BaggedRegressor, X, y, bag; parameters...)
    params = Dict(parameters)
    @extract_from params n rgs.n
    @extract_from params verbose true
    rgs_extra = clone(rgs)
    rgs_extra.n = n
    fit!(rgs_extra, X,y, bag; verbose=verbose, params...) # build size `n` ensemble for rgs_extra
    rgs.n_iter += n
    rgs.model = vcat(rgs.model, rgs_extra.model) # add its ensemble to that of rgs
    return rgs
end

add!(rgs::BaggedRegressor, X, y; parameters...) = add!(rgs, X, y, 1:length(y); parameters...)

function predict(rgs::BaggedRegressor, X, bag=[])
    rgs.n_iter > 0 || error("Attempting to predict using $rgs")                                     
    if isempty(bag)
        bag = 1:size(X, 1)
    end
    return mean([predict(m, X, bag) for m in rgs.model])
end

# type RandomForestRegressor <: BaseRegressor
#     forest::Vector{TreeRegressor}
#     error::Float64 # internally computed estimate of generalization error
# end

# function RandomForestRegressor(
#   X::DataTable,           # Input patterns, a list of 1D arrays, one per feature
#   y::Vector{Float64},     # output responses
#   fullbag;                # iterator returning training pattern indices
#   n_trees = 20,           # number of trees
#   max_features = "all",    # num features in random selection (integer or string; see below)
#   min_patterns_split = 2,  # min num patterns at node to consider split
#   verbose = true)
#     # This constructor fits a Random Forest model to the data supplied
#     # as input. An internal estimate of the root mean square error is
#     # supplied as the attribute `error'. Predictions are extraced
#     # using the 'predict' function:

#     #   > regressor = RandomForestRegressor(X, y, 1:length(y), 200, "log2", 2)
#     #   > regressor.error
#     #   2.334324
#     #   > predict(regressor, ['a', 'b', 1, 'f'])
#     #   126.4545

#     #  `max_features' can be "all", "sqrt", "log2" a positive integer
#     #  between 1 and n_features inclusively, or a float between 0 and
#     #  1 (representing proportion of n_features).
  
#     if n_trees <= 0
#         Base.error("The number of trees must be a positive integer.")
#     end
    
#     # initialize random number generator:
#     srand((round(Int,time()*1000000)))
    
#     n_features = length(X)
#     n_patterns = length(fullbag)
#     max_n_patterns = size(X)[1] # for allocating storage 

#     # recast max_features as integer as necessary
#     if typeof(max_features)<:AbstractString
#         if max_features == "all"
#             max_features = n_features
#         elseif max_features == "sqrt"
#             max_features = round(Int,sqrt(n_features))
#         elseif  max_features == "log2"
#             max_features = round(Int,log(2, n_features))
#         else
#             throw(DomainError())
#         end
#     elseif  typeof(max_features) == Float64
#         if max_features > 0 && max_features <= n_features
#             max_features = round(Int,max_features*n_features)
#         else
#             throw(DomainError())
#         end
#     elseif typeof(max_features) == Int
#         if max_features <= 0 || max_features > n_features
#             throw(DomainError())
#         end
#     else
#         throw(DomainError())
#     end

#     verbose ? begin
#         println("The total number of features in the sample is  $n_features.") : nothing
#         println("The number of features on which to base node splits will be $max_features")
#         println("The total number of samples in the bag provided is $n_patterns")
#         println("Minimum number of samples at a node before considering a split is $min_patterns_split")
#         println()
#         print("Now training tree number: ")
#         end : nothing
              
#     forest = Array(TreeRegressor, n_trees)
#     y_oob = zeros(Float64, max_n_patterns) # The out-of-bag estimate of output on each pattern
#     tally = zeros(Int, max_n_patterns)  # For keeping track of number of times
#                                         # each pattern got a contribution to its
#                                         # oob output estimate
  
#     for t in 1:n_trees
#         verbose ? print("$t ") : nothing
#         flush(STDOUT)

#         bag = Distributions.sample(fullbag, n_patterns; replace=true)
#         notbag = setdiff(fullbag, bag)
# #        println("fulbag=$fullbag bag=$bag notbag=$notbag")

#         # build a tree
#         tree = TreeRegressor(X, y, bag; max_features=max_features, min_patterns_split=min_patterns_split)
#         forest[t] = tree
        
#         # update the (unnormalized) oob estimates of outputs:
#         for i in notbag
#             y_oob[i] += predict(tree, X[i,:])
#             tally[i] += 1
#         end
#     end

#     # normalize the oob estimates of the ouputs:
#     for i in fullbag
#         if tally[i] != 0      # very likely
#             y_oob[i] = y_oob[i]/tally[i]
#         end
#     end

#     # calculate the mean square error based on oob estimates of output
#     error = 0.0
#     count = 0
#     for i in fullbag
#         if tally[i] != 0
#             count += 1
#             error += (y_oob[i] - y[i])^2
#         end
#     end
#     error = sqrt(error/count)
    
#     return RandomForestRegressor(forest, error)
    
# end

# function predict(regressor::RandomForestRegressor, pattern)
#     y_hat = 0.0
#     forest =  regressor.forest
#     for tree in forest
#         y_hat += predict(tree, pattern)
#     end
#     return y_hat/length(forest)
# end

##################
# LoessRegressor #
##################

type LoessRegressor <:BaseRegressor

    # hyperparameters:
    span::Float64
    degree::Int

    # model:
    model::Loess.LoessModel{Float64}

    # post-fit parameters:
    fitted::Bool

    # constructor:
    function LoessRegressor(span::Float64, degree::Int)
        rgs = new(span,degree)
        rgs.fitted = false
        return rgs
    end
end

function show(stream::IO, rgs::LoessRegressor)
    tail(n) = "..."*string(n)[end-3:end]
    prefix = (rgs.fitted ? "" : "unfitted ")
    print(stream, prefix, "LoessRegressor@$(tail(hash(rgs)))")
end

function showall(stream::IO, rgs::LoessRegressor)
    show(stream, rgs); println("\n Hyperparameters:")
    display(get_params(rgs))
end

# lazy keyword constructors:
## without fitting:
LoessRegressor(;span::Float64=0.1, degree::Int=2) = LoessRegressor(span, degree)
                                                   
## with immediate fitting:
LoessRegressor(x, y, bag; parameters...) = fit!(LoessRegressor(;parameters...), x, y, bag)

## with immediate fitting (bagless version):
LoessRegressor(x, y; parameters...) = LoessRegressor(x, y, 1:length(y); parameters...)

# provide `set_params!`, `get_params`, `clone`:
@declare_hyperparameters LoessRegressor [:span, :degree]

function fit!(rgs::LoessRegressor, x, y, bag; verbose=false)
    xx = Float64[x[i] for i in bag]
    yy = Float64[y[i] for i in bag]
    rgs.model = Loess.loess(xx,yy; span=rgs.span, degree=rgs.degree)
    rgs.fitted = true
    return rgs
end

fit!(rgs::LoessRegressor, x, y) = fit!(rgs, x, y, 1:length(y))

function predict(rgs::LoessRegressor, r::Real)
    rgs.fitted || error("Attempting to predict using $rgs.")
    r = convert(Float64, r)
    return Loess.predict(rgs.model, r)
end

function predict(rgs::LoessRegressor, x::Vector{Float64})
    return Float64[Loess.predict(rgs.model, r) for r in x]
end

##################
# LinearRegresor #
##################

# to do: this regressor takes as input X a DataTable but would more
# efficiently take a DataFrame as input.

type LinearRegressor <: BaseRegressor

    # hyperparameters: None
    
    # model:
    model

    # post-fit parameters:
    names::Vector{Symbol}
    fitted::Bool
    
    function LinearRegressor()
        rgs = new()
        rgs.fitted = false
        return rgs
    end
    
end

function show(stream::IO, rgs::LinearRegressor)
    tail(n) = "..."*string(n)[end-3:end]
    prefix = (rgs.fitted ? "" : "unfitted ")
    print(stream, prefix, "LinearRegressor@$(tail(hash(rgs)))")
end

function showall(stream::IO, rgs::LinearRegressor)
    show(stream, rgs); println()
    show(stream, rgs.model)
end

LinearRegressor(X, y, bag) = fit!(LinearRegressor(), X, y, bag)

## with immediate fitting (bagless version):
LinearRegressor(X, y) = LinearRegressor(X, y, 1:length(y))

# provide `set_params!`, `get_params`, `clone`:
@declare_hyperparameters LinearRegressor [:span, :degree]

function fit!(rgs::LinearRegressor, X::DataTable, y::Vector, bag; verbose=false)

    size(X,1) == length(y) || throw(DimensionMismatch())

    if sum(!X.scheme.is_ordinal) != 0
        throw(ArgumentError("You are attempting to fit a linear model to a DataTable with categoricals."*
                         " Consider one hot encoding."))
    end
    
    # drop rows without indices in bag:
    df = convert(DataFrame, X[bag,:])
    y = y[bag]
    
    # get names of input columns:
    rgs.names = X.names

    # combine input patterns and responses:
    df[:_target_] = y

    # create R-style formula:
    lhs = :_target_
    rhs = Expr(:call, :+, rgs.names...)
    form  = Formula(lhs, rhs)    

    # build the model:
    rgs.model = GLM.fit(GLM.LinearModel, form, df)

    rgs.fitted = true

    return rgs

end

function fit!(rgs::LinearRegressor, X::DataFrame, y::Vector, bag; verbose=false)

    size(X,1) == length(y) || throw(DimensionMismatch())

    element_types = [eltype(X[j]) for j in 1:size(X,2)]
    is_not_real = map(T -> !(T<:Real), element_types)
    if sum(is_not_real) > 0
        throw(ArgumentError("You are attempting to fit a"*
                            " linear model to a DataFrame with categoricals."*
                            " Consider one hot encoding."))
    end
    
    # drop rows without indices in bag:
    df = X[bag,:]
    y = y[bag]
    
    # get names of input columns:
    rgs.names = names(X)

    # combine input patterns and responses:
    df[:_target_] = y

    # create R-style formula:
    lhs = :_target_
    rhs = Expr(:call, :+, rgs.names...)
    form  = Formula(lhs, rhs)    

    # build the model:
    rgs.model = GLM.fit(GLM.LinearModel, form, df)

    rgs.fitted = true

    return rgs

end

function predict(rgs::LinearRegressor, X::DataTable, bag=[])

    rgs.fitted || error("Attempting to predict using $rgs.")

    # length checks:
    X.ncols == length(rgs.names) || throw(DimensionMismatch())

    if isempty(bag)
        XX = X
    else
        XX = X[bag,:]
    end

    # call GLM predict method:
    return GLM.predict(rgs.model, convert(DataFrame,XX))

end

function predict(rgs::LinearRegressor, X::DataFrame, bag=[])

    rgs.fitted || error("Attempting to predict using $rgs.")

    # length checks:
    size(X, 2) == length(rgs.names) || throw(DimensionMismatch())

    if isempty(bag)
        XX = X
    else
        XX = X[bag,:]
    end

    # call GLM predict method:
    return GLM.predict(rgs.model, XX)

end


#########################
# Elastic Net Regressor #
#########################

""" 
# `ElasticNetRegressor(lambda=0, alpha=1.0, standardize=false, max_n_coefs=0, criterion=:ceof)`

ScikitLearn style Wrapper for elastic net implementation at:

[1] http://lassojl.readthedocs.io/en/latest/lasso.html

Algorithm details: 

[2] Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization
paths for generalized linear models via coordinate descent. Journal of
Statistical Software, 33(1), 1.

## Example use

For input patterns, presented as a `DataFrame` or `DataTable` object
`X`, and corresponding output responses `y`:

    julia> rgs = ElasticNetRegressor(alpha=0.5)
    julia> fit!(rgs,X,y) # fit and optimize regularization parameter
    julia> cv_error(rgs, X,y)
    0.534345

    julia> rgs.lambdaopt
    0.005987423  # estimate of optimal value of lambda

    julia> rgs2 = ElasticNetRegressor(X, y, lambda = 0.006, alpha =0.2) # one line construct and train

## Feature extraction

From a trained object `rgs` one may obtain a ranking of the features used in training based on the absolute values of coefficients in the linear model obtained (generally choosing `alpha=1.0` for pure L1 regularization):

    julia> coefs(rgs)
    15×3 DataFrames.DataFrame
    │ Row │ feature │ name                   │ coef       │
    ├─────┼─────────┼────────────────────────┼────────────┤
    │ 1   │ 7       │ SaleType__CWD          │ 0.584603   │
    │ 2   │ 16      │ SaleCondition__Alloca  │ -0.307277  │
    │ 3   │ 11      │ SaleType__New          │ 0.274558   │
    │ 4   │ 19      │ SaleCondition__Partial │ 0.245157   │
    │ 5   │ 15      │ SaleCondition__AdjLand │ -0.221836  │
    │ 6   │ 18      │ SaleCondition__Normal  │ 0.107383   │

Or type `showall(rgs)` in the REPR for a graphical representation. 

## Hyperparameters:

- `lambda`: regularization parameter, denoted λ in [2]. If set to 0
  (default) then, upon fitting, a value is optimized as follows: For
  each training set in a cross-validation regime, the value of
  `lambda` minimizing the RMS error on the hold-out set is noted. The
  minimization is over a "regularization path" of `lambda` values
  (numbering `n_lambda`) which are log-distibuted from λmax down to
  lambda_min_ratio \* λmax. Here λmax is the smallest amount of regularization
  giving a null model.  The geometric mean of these individually
  optimized values is used to train the final model on all available
  data and is stored as the post-fit parameter `lambdaopt`. N.B. *The
  hyperparameter `lambda` is not updated (remains 0).*

- `n_lambdas`: number of `lambda` values tried in each
  cross-validation path if `lambda` is unspecified (initially set to
  0). Defaults to 100.

- `alpha`: denoted α in [2], taking values in (0,1]; measures degree of Lasso in the
  Ridge-Lasso mix and defaults to 1.0 (Lasso regression)

- `standardize`: when `true` input features are centred and rescaled to
  have mean 0 and std 1. Default: false

- `max_n_coefs`: maximum number of non-zero coefficients being
  admitted. If set to zero, then `max_n_coefs = min(size(X, 2),
  2*size(X, 1))` is used. If exceeded an error is thrown.

- `lambda_min_ratio`: see discussion of `lambda` above. If unspecified
  (or 0.0) then this is automatically selected.

- `criterion`: Early stopping criterion in building paths. If
  `criterion` takes on the value `:coef` then the model is considered
  to have converged if the the maximum absolute squared difference in
  coefficients between successive iterations drops below a certain
  tolerance. This is the criterion used by glmnet. Alternatively, if
  `criterion` takes on the value `:obj` then the model is considered
  to have converged if the the relative change in the Lasso/Elastic
  Net objective between successive iterations drops below a certain
  tolerance. This is the criterion used by GLM.jl. Defaults to `:coef`

## Post-fit parameters:

- `lambdaopt` the optimized value of the regularization parameter, or
  `lambda` if the latter is specified and non-zero.

- `intercept`: intercept of final linear model

- `coefs`:  coefficients for final linear model, stored as `SparseVector{Float64,Int64}`

- `loglambdaopt = log(lambdaopt)`

## If `lambda` is not specified or zero, then these parameters are also available:

- `loglambdaopt_stde`: standard error of cross-validation estimates of
  log(lambda) 

- `cv_rmse`: mean of cross-validation RMS errors 

- `cv_rmse_stde`: standard error of cross-validation RMS errors 


"""
type ElasticNetRegressor <: BaseRegressor

    # hyperparameters:
    lambda::Float64
    n_lambdas::Int
    alpha::Float64
    standardize::Bool
    max_n_coefs::Int
    criterion::Symbol
    lambda_min_ratio::Float64
    
    # model:
    intercept::Float64
    coefs::SparseVector{Float64,Int64}

    # post-fitted parameters
    lambdaopt::Float64
    loglambdaopt::Float64
    loglambdaopt_stde::Float64
    cv_rmse::Float64
    cv_rmse_stde::Float64
    names::Vector{Symbol}
    
    fitted::Bool
    
    function ElasticNetRegressor(lambda, n_lambdas::Int, alpha, standardize::Bool, 
                                 max_n_coefs::Int, criterion::Symbol, lambda_min_ratio::Float64)
        if alpha <= 0.0 || alpha > 1.0
            alpha == 0.0 ? throw(Base.error("alpha=0 dissallowed."*
                                            " Consier ridge regression instead.")) : nothing
            throw(DomainError)
        end
        rgs = new(lambda, n_lambdas, alpha, standardize, max_n_coefs, criterion, lambda_min_ratio)
        rgs.fitted = false
        
        return rgs
    end
    
end

@declare_hyperparameters ElasticNetRegressor [:lambda, :n_lambdas, :alpha,
                                              :standardize, :max_n_coefs, :criterion, :lambda_min_ratio]

"""
## function coefs(rgs::ElasticNetRegressor)

Returns a `DataFrame` with three columns:

column name | description
:-----------|:-------------------------------------------------
`:feature`  | index of a feature used to train `rgs`
`:name `    | name of that feature
`:coef`     | coefficient for that feature in the trained model

The rows are ordered by the absolute value of the coefficients. If
`rgs` is unfitted, an error is returned.

"""
function coefs(rgs::ElasticNetRegressor)
    rgs.fitted || error("You are attempting to extract model information from $rgs.")
    coef_given_feature = Dict{Int, Float64}()
    abs_coef_given_feature = Dict{Int, Float64}()
    v = rgs.coefs # SparseVector
    for k in eachindex(v.nzval)
        coef_given_feature[v.nzind[k]] = v.nzval[k]
        abs_coef_given_feature[v.nzind[k]] = abs(v.nzval[k])
    end
    df = DataFrame()
    df[:feature] = reverse(keys_ordered_by_values(abs_coef_given_feature))
    df[:name] = map(df[:feature]) do feature
        rgs.names[feature]
    end
    df[:coef] = map(df[:feature]) do feature
        coef_given_feature[feature]
    end
    return df
end

function show(stream::IO, rgs::ElasticNetRegressor)
    tail(n) = "..."*string(n)[end-3:end]
    prefix = (rgs.fitted ? "" : "unfitted ")
    print(stream, prefix, "ElasticNetRegressor@$(tail(hash(rgs)))")
end

function showall(stream::IO, rgs::ElasticNetRegressor)
    show(stream, rgs); println()
    println("Hyperparameters:")
    display(get_params(rgs))
    if rgs.lambda == 0.0
        println("Regularization is to be set automatically (lambda = 0.0)")
    end
    
    if rgs.fitted
        println("Post-fit parameters:")
        d = Dict{Symbol,Any}()
        d[:lambdaopt] = rgs.lambdaopt
        d[:loglambdaopt] = rgs.loglambdaopt
        d[:intercept] = rgs.intercept
        if rgs.lambda == 0.0
            d[:cv_rmse] = rgs.cv_rmse
            d[:cv_rmse_stde] = rgs.cv_rmse_stde
            d[:loglambdaopt_stde] = rgs.loglambdaopt_stde
        end
        display(d)
        if rgs.fitted
            # display the feature ranked by abs value of coef:
            x = Symbol[]
            y = Float64[]
            cinfo = coefs(rgs)
            for i in 1:size(cinfo, 1)
                name, coef = (cinfo[i, :name], cinfo[i, :coef])
                coef = floor(1000*coef)/1000
                if coef < 0
                    label = string(name, " (-)")
                else
                    label = string(name, " (+)")
                end
                push!(x, label)
                push!(y, abs(coef))
            end
            UnicodePlots.show(stream, UnicodePlots.barplot(x,y,title="Non-zero coefficients"))
        end
    end
end

# lazy keyword constructors:
## without fitting:
function ElasticNetRegressor(;lambda=0.0, n_lambdas::Int=100,
                             alpha=1.0, standardize::Bool=false, 
                             max_n_coefs::Int=0, criterion::Symbol=:coef, lambda_min_ratio=0.0)
    return ElasticNetRegressor(lambda, n_lambdas, alpha, standardize, max_n_coefs, criterion, lambda_min_ratio)
end

## with immediate fitting:
ElasticNetRegressor(X, y, bag; parameters...) = fit!(ElasticNetRegressor(;parameters...), X, y, bag)

## with immediate fitting (bagless version):
ElasticNetRegressor(X, y; parameters...) = fit!(ElasticNetRegressor(;parameters...), X, y)

"""

`getcol(A, j)` gets the jth column of a sparse matrix `A`, returned as
a sparse vector. If `j=0` then the last column is returned. If `j`
exceeds the number of columns of `A`, then a zero vector is returned.

"""
function getcol(A::SparseMatrixCSC{Float64,Int}, j::Int)
    if j == 0
        j = A.n
    end
    j > A.n ? (return SparseVector(A.m,Int[],Float64[])) : nothing
    I = Int[]     # indices for non-zero entries of col
    V = Float64[] # values of non-zero entries of col
    for k in A.colptr[j]:(A.colptr[j+1] - 1)
        push!(I, A.rowval[k])
        push!(V, A.nzval[k])
    end
    return SparseVector(A.m,I,V)
end

function lambda_max(X::DataTable, y, alpha)
    Xy = X.raw'*(y .- mean(y)) # dot products of feature columns with centred response
    λmax = abs(Xy[1])
    for i = 2:length(Xy)
        x = abs(Xy[i])
        if x > λmax
            λmax = x
        end
    end
    return λmax/(alpha*length(y))
end

function lambda_max(X::DataFrame, y, alpha)
    # dot products of feature columns with centred response:    
    Xy = [first(X[j]'*(y .- mean(y))) for j in 1:size(X,2)]
    λmax = abs(Xy[1])
    for i = 2:length(Xy)
        x = abs(Xy[i])
        if x > λmax
            λmax = x
        end
    end
    return λmax/(alpha*length(y))
end

function fit!(rgs::ElasticNetRegressor, X::DataTable, y, bag; verbose=false)

    const n_folds = 10

    size(X, 1) == length(y) || throw(DimensionMismatch())

    bag = collect(bag)
    rgs.names = X.names
    
    # initialize path hyperparameters:
    path = ElasticNetRegressorPath(Float64[], rgs.n_lambdas, rgs.alpha, rgs.standardize,
                                   rgs.max_n_coefs, rgs.criterion, rgs.lambda_min_ratio)

    # unpack the data to be used:
    if isempty(bag)
        XX = X
        yy = y
    else
        XX = X[bag,:]
        yy = y[bag]
    end
    newbag = 1:length(bag)

    if rgs.lambda == 0
    
        best_lambdas = Array(Float64, n_folds)
        errors = Array(Float64, n_folds)
        n_samples = length(bag)
        # if randomized
        #     bag = Distributions.sample(bag, n_samples, replace=false)
        # end    
        
        k = floor(Int,n_samples/n_folds)
        first = 1       # first test_bag index
        second = k
        # println("Optimizing regularization parameter using $(n_folds)-fold cross-validation. ")
        for n in 1:n_folds
            print("fold number = $n  ")
            test_bag = newbag[first:second]
            train_bag = vcat(newbag[1:first-1], newbag[second+1:end])
            first = first + k 
            second = second + k
            fit!(path, XX[train_bag,:], yy[train_bag])
            lambdas = path.postfit_lambdas
            ss_deviations = zeros(Float64, length(lambdas))
            for i in test_bag
                ss_deviations = ss_deviations + map(predict(path, row(XX,i))) do yhat
                    (yhat - yy[i])^2
                end
            end
            L = indmin(ss_deviations)
            if L in [0, length(path.postfit_lambdas)]
                Base.warn("Optimal value of lambda not found in search range.")
            end
            best_lambdas[n] = lambdas[L]
            # println("lambda = $(best_lambdas[n])")
            errors[n] = sqrt(ss_deviations[L]/length(test_bag))
        end
        
        loglambdas = log(best_lambdas)
        rgs.loglambdaopt = mean(loglambdas)
        rgs.loglambdaopt_stde = std(loglambdas)
        rgs.lambdaopt = exp(rgs.loglambdaopt)
        rgs.cv_rmse = mean(errors)
        rgs.cv_rmse_stde = std(errors)

        println("Optimal regularization, lambda = $(rgs.lambdaopt)")        

        # preparing for final train:
        lambda = rgs.lambdaopt
    else
        lambda = rgs.lambda 
    end
    
    # calculate path lambdas for final train
    λmax = lambda_max(XX, yy, rgs.alpha)
    λmax < rgs.lambda ? throw(Base.error("Something wrong here")) : nothing
    path.lambdas = exp.(linspace(log(λmax), log(lambda), rgs.n_lambdas))
    # make sure last element of lambdas is *exactly* lambda:
    path.lambdas[end]=lambda
    
    # final train on all the data:
    fit!(path, XX, yy)
    
    # check path ends at optimal lambda:
    if path.postfit_lambdas[end] != lambda
        Base.warn("Early stopping of path before required lambda reached.")
    end
        
    # record model
    rgs.intercept =  path.intercepts[end]
    rgs.coefs = getcol(path.coefs, 0) # 0 gets last column

    rgs.fitted = true

    return rgs
    
end

function fit!(rgs::ElasticNetRegressor, X::DataFrame, y, bag; verbose=false)

    const n_folds = 10

    if size(X, 1) != length(y)
        throw(DimensionMismatch("Number of input"*
                                " patterns and responses not matching."))
    end
    bag = collect(bag)
    rgs.names = names(X)
    
    # initialize path hyperparameters:
    path = ElasticNetRegressorPath(Float64[], rgs.n_lambdas, rgs.alpha, rgs.standardize,
                                   rgs.max_n_coefs, rgs.criterion, rgs.lambda_min_ratio)

    # unpack the data to be used:
    if isempty(bag)
        XX = X
        yy = y
    else
        XX = X[bag,:]
        yy = y[bag]
    end
    newbag = 1:length(bag)

    if rgs.lambda == 0
    
        best_lambdas = Array(Float64, n_folds)
        errors = Array(Float64, n_folds)
        n_samples = length(bag)
        # if randomized
        #     bag = Distributions.sample(bag, n_samples, replace=false)
        # end    
        
        k = floor(Int,n_samples/n_folds)
        first = 1       # first test_bag index
        second = k
        println("Optimizing regularization parameter using $(n_folds)-fold cross-validation. ")
        for n in 1:n_folds
            print("fold number = $n  ")
            test_bag = newbag[first:second]
            train_bag = vcat(newbag[1:first-1], newbag[second+1:end])
            first = first + k 
            second = second + k
            fit!(path, XX[train_bag,:], yy[train_bag])
            lambdas = path.postfit_lambdas
            ss_deviations = zeros(Float64, length(lambdas))
            for i in test_bag
                ss_deviations = ss_deviations + map(predict(path, row(XX,i))) do yhat
                    (yhat - yy[i])^2
                end
            end
            L = indmin(ss_deviations)
            if L in [0, length(path.postfit_lambdas)]
                Base.warn("Optimal value of lambda not found in search range.")
            end
            best_lambdas[n] = lambdas[L]
            println("lambda = $(best_lambdas[n])")
            errors[n] = sqrt(ss_deviations[L]/length(test_bag))
        end
        
        loglambdas = log(best_lambdas)
        rgs.loglambdaopt = mean(loglambdas)
        rgs.loglambdaopt_stde = std(loglambdas)
        rgs.lambdaopt = exp(rgs.loglambdaopt)
        rgs.cv_rmse = mean(errors)
        rgs.cv_rmse_stde = std(errors)

        # preparing for final train:
        lambda = rgs.lambdaopt
    else
        lambda = rgs.lambda 
    end
    
    # calculate path lambdas for final train
    λmax = lambda_max(XX, yy, rgs.alpha)
    λmax >= rgs.lambda || warn("lambda exceeding λmax.")
    path.lambdas = exp.(linspace(log(λmax), log(lambda), rgs.n_lambdas))
    # make sure last element of lambdas is *exactly* lambda:
    path.lambdas[end]=lambda
    
    # final train on all the data:
    fit!(path, XX, yy)
    
    # check path ends at optimal lambda:
    if path.postfit_lambdas[end] != lambda
        Base.warn("Early stopping of path before required lambda reached.")
    end
        
    # record model
    rgs.intercept =  path.intercepts[end]
    rgs.coefs = getcol(path.coefs, 0) # 0 gets last column

    rgs.fitted = true

    return rgs
    
end


function predict(rgs::ElasticNetRegressor, pattern::Vector{Float64})

    ret = rgs.intercept
    for i in eachindex(rgs.coefs.nzval)
        ret = ret + rgs.coefs.nzval[i]*pattern[rgs.coefs.nzind[i]]
    end

    return ret

end

function predict(rgs::ElasticNetRegressor, X::DataFrame, bag=[])

    rgs.fitted || error("Attempting to predict using $rgs.")

    if isempty(bag)
        bag = 1:size(X,1)
    end
    
    return [predict(rgs, row(X, i)) for i in bag]

end

""" 

Wrapper for constructing elastic net paths. Not exported but used
by ElasticNetRegressor fit! methods.

"""
type ElasticNetRegressorPath <: BaseRegressor

    # hyperparameters:
    lambdas::Vector{Float64} # proposed path of lambdas; automatically generated if empty
    n_lambdas::Int           # length of path if automatically generated
    alpha::Float64           # in (0,1], controlling ridge-lasso mix
    standardize::Bool 
    max_n_coefs::Int  # error thrown if number of coefficients along path exceeds this
    criterion::Symbol # criterion for stopping, :coef or :obj; see ElasticNetRegressor doc string
    lambda_min_ratio::Float64
    
    # model:
    model::Lasso.LassoPath

    # post-fitted parameters
    postfit_lambdas::Vector{Float64} 
    proportion_deviance_explained::Vector{Float64}
    intercepts::Vector{Float64}
    coefs::SparseMatrixCSC{Float64,Int64}
    
    fitted::Bool

    function ElasticNetRegressorPath(lambdas::Vector{Float64}, n_lambdas::Int, alpha::Float64, standardize::Bool, 
                                 max_n_coefs::Int, criterion::Symbol, lambda_min_ratio::Float64)
        if alpha <= 0.0 || alpha > 1.0
            alpha == 0.0 ? throw(Base.error("alpha=0 dissallowed."*
                                            " Consier ridge regression instead.")) : nothing
            throw(DomainError)
        end
        rgs = new(lambdas, n_lambdas, alpha, standardize, max_n_coefs, criterion, lambda_min_ratio)
        rgs.fitted = false
        return rgs
    end
    
end

@declare_hyperparameters ElasticNetRegressorPath [:lambdas, :n_lambdas, :alpha,
                                                  :standardize, :max_n_coefs, :criterion, :lambda_min_ratio]

function show(stream::IO, rgs::ElasticNetRegressorPath)
    tail(n) = "..."*string(n)[end-3:end]
    prefix = (rgs.fitted ? "" : "unfitted ")
    print(stream, prefix, "ElasticNetRegressorPath@$(tail(hash(rgs)))")
end

function showall(stream::IO, rgs::ElasticNetRegressorPath)
    show(stream, rgs); println()
    println("Hyperparameters:")
    display(get_params(rgs))
    if rgs.fitted
        println("Post-fit parameters:")
        d=Dict{Symbol,Any}()
        d[:postfit_lambdas]=rgs.postfit_lambdas
        d[:proportion_deviance_explained]=rgs.proportion_deviance_explained
        display(d)
    end
end

# lazy keyword constructors:
## without fitting:
function ElasticNetRegressorPath(;lambdas::Vector{Float64}=Float64[], n_lambdas::Int=100,
                             alpha=1.0, standardize::Bool=false, 
                             max_n_coefs::Int=0, criterion::Symbol=:coef, lambda_min_ratio=0.0)
    return ElasticNetRegressorPath(lambdas, n_lambdas, alpha, standardize, max_n_coefs, criterion, lambda_min_ratio)
end

## with immediate fitting:
ElasticNetRegressorPath(X::DataTable,
              y::Vector{Float64},
              bag; parameters...) = fit!(ElasticNetRegressorPath(;parameters...), X, y, bag)

## with immediate fitting (bagless version):
ElasticNetRegressorPath(X::DataTable,
              y::Vector{Float64}; parameters...) = fit!(ElasticNetRegressorPath(;parameters...), X, y)

fit!(rgs::ElasticNetRegressorPath, X::DataTable, y; verbose=false) =
                  fit!(rgs::ElasticNetRegressorPath, X::DataTable, y, Int[])

function fit!(rgs::ElasticNetRegressorPath, X::DataTable, y, bag)

    if sum(!X.scheme.is_ordinal) != 0
        throw(ArgumentError("You are attempting to fit a linear model to a DataTable with categoricals."*
                         " Consider one hot encoding."))
    end

    if size(X, 1) != length(y) 
        throw(DimensionMismatch("Number of input patterns and responses not matching."))
    end

    # unpack the data to be used:
    if isempty(bag)
        XX = X.raw
        yy = y
    else
        XX = Array(Float64, (length(bag), size(X, 2)))
        yy = Array(Float64, length(bag))
        for i in eachindex(bag)
            yy[i] = y[bag[i]]
        end
        for j in 1:size(X, 2)
            for i in eachindex(bag)
                XX[i,j] = X.raw[bag[i],j]
            end
        end
    end
     
    # build the model by finding the right call given hyperparams:
    if isempty(rgs.lambdas)
        if rgs.max_n_coefs == 0
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        else
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion,
                                      verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        end
    else
        reverse!(sort!(rgs.lambdas))
        if rgs.max_n_coefs == 0
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        else
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion,
                                      verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        end
    end
    
    rgs.postfit_lambdas = rgs.model.λ
    rgs.proportion_deviance_explained = rgs.model.pct_dev
    rgs.intercepts = rgs.model.b0
    rgs.coefs = rgs.model.coefs
    rgs.fitted = true
        
    return rgs
end

function fit!(rgs::ElasticNetRegressorPath, X::DataFrame, y, bag; verbose=false)

    is_ordinal = [eltype(X[j]) <: Real for j in 1:size(X, 2)]
    if sum(!is_ordinal) != 0
        throw(ArgumentError("You are attempting to fit a linear model to a DataTable with categoricals."*
                         " Consider one hot encoding."))
    end

    if size(X, 1) != length(y) 
        throw(DimensionMismatch("Number of input patterns and responses not matching."))
    end

    # unpack the data to be used:
    if isempty(bag)
        XX = Array(X)
        yy = y
    else
        XX = Array(Float64, (length(bag), size(X, 2)))
        yy = Array(Float64, length(bag))
        for i in eachindex(bag)
            yy[i] = y[bag[i]]
        end
        for j in 1:size(X, 2)
            for i in eachindex(bag)
                isna(X[bag[i],j]) && error("NA encountered.")
                XX[i,j] = X[bag[i],j]
            end
        end
    end
     
    # build the model by finding the right call given hyperparams:
    if isempty(rgs.lambdas)
        if rgs.max_n_coefs == 0
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        else
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      nλ=rgs.n_lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion,
                                      verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        end
    else
        reverse!(sort!(rgs.lambdas))
        if rgs.max_n_coefs == 0
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      criterion=rgs.criterion, verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        else
            if rgs.lambda_min_ratio == 0.0
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion, verbose=false)
            else
                rgs.model = Lasso.fit(Lasso.LassoPath, XX, yy;
                                      λ =rgs.lambdas, α=rgs.alpha, standardize=rgs.standardize,
                                      maxncoef=rgs.max_n_coefs, criterion=rgs.criterion,
                                      verbose=false, λminratio=rgs.lambda_min_ratio)
            end
        end
    end
    
    rgs.postfit_lambdas = rgs.model.λ
    rgs.proportion_deviance_explained = rgs.model.pct_dev
    rgs.intercepts = rgs.model.b0
    rgs.coefs = rgs.model.coefs
    rgs.fitted = true
        
    return rgs
   
end


"""

Note that this predict returns a column vector of predictions, one for
each `lambda` value in the final path.

"""
function predict(rgs::ElasticNetRegressorPath, pattern::Vector)

    row =  Lasso.predict(rgs.model, pattern')
    return vec(row) # a column vector
    
end

######################
# XGBoost            #
######################

""" 
# `XGBoostRegressor(X, y, parameters...)`

A ScikitLearnBase wrapper for `XGBoost.jl`.

Here the training input object `X` should be a two-dimensional `Array`
or `SparseMatrixCSC` object and `y` a one-dimensional `Array` or
`SparseVector` object. **N.B. The parameter denoted `num_round` in
`XGBoost` package has been renamed `n` here.**

## Methods

In addition to the standard ScikitLearn API (`fit!`, `predict`,
`get_params`, `set_params!`, `clone`) there is a `cv` function for
performing "running" cross-validation (errors reported at each
boosting step). Sample call:

    julia> rgs = XGBoostRegressor(n=100)
    julia> showall(rgs)
    unfitted XGBoostRegressor@...3557
    Dict{Symbol,Any} with 12 entries:
     :subsample         => 1.0
     :objective         => "reg:linear"
     :max_depth         => 6
     :eta               => 0.3
     :colsample_bytree  => 1.0
     :n         => 100
     :alpha             => 0.0
     :lambda            => 1.0
     :colsample_bylevel => 1.0
     :min_child_weight  => 1.0
     :tree_method       => "auto"
     :gamma             => 0.0

    julia> cv(rgs, X, y, n_folds=6, error="rmse")        # default values of parameters shown
    julia> cv(rgs, X, y, 1:100, n_folds=6, error="rmse") # with optional row bag included

## Parameters

The following are most of the parameters available for tuning XGBoost,
as copied from the [XGBoost
website](http://xgboost.readthedocs.io/en/latest//parameter.html).
They are not all yet implemented in the current wrap. In particular
DART parameters are not yet tunable.

## Parameters for Tree Booster

- n [default = 20, called `num_round` in unwrapped implementation]

    - number of boosting iterations.

- eta [default=0.3, alias: learning_rate]

    - step size shrinkage used in update to prevents
      overfitting. After each boosting step, we can directly get the
      weights of new features, and eta actually shrinks the feature
      weights to make the boosting process more conservative.

   - range: [0,1]

- gamma [default=0, alias: min_split_loss]

    - minimum loss reduction required to make a further partition on a
      leaf node of the tree. The larger, the more conservative the
      algorithm will be.

    - range: [0,∞]

- max_depth [default=6]

    - maximum depth of a tree, increase this value will make the model
      more complex / likely to be overfitting. 0 indicates no limit,
      limit is required for depth-wise grow policy.

    - range: [0,∞]

- min_child_weight [default=1]

    - minimum sum of instance weight (hessian) needed in a child. If
      the tree partition step results in a leaf node with the sum of
      instance weight less than min_child_weight, then the building
      process will give up further partitioning. In linear regression
      mode, this simply corresponds to minimum number of instances
      needed to be in each node. The larger, the more conservative the
      algorithm will be.

    - range: [0,∞]

- max_delta_step [default=0]

    - Maximum delta step we allow each tree’s weight estimation to
      be. If the value is set to 0, it means there is no
      constraint. If it is set to a positive value, it can help making
      the update step more conservative. Usually this parameter is not
      needed, but it might help in logistic regression when class is
      extremely imbalanced. Set it to value of 1-10 might help control
      the update

    - range: [0,∞]

- subsample [default=1]

    - subsample ratio of the training instance. Setting it to 0.5
      means that XGBoost randomly collected half of the data instances
      to grow trees and this will prevent overfitting.

    - range: (0,1]

- colsample_bytree [default=1]

    - subsample ratio of columns when constructing each tree.

    -range: (0,1]

- colsample_bylevel [default=1]

    - subsample ratio of columns for each split, in each level.

    - range: (0,1]

- lambda [default=1, alias: reg_lambda]

    - L2 regularization term on weights, increase this value will make
      model more conservative.

- alpha [default=0, alias: reg_alpha]

    - L1 regularization term on weights, increase this value will make
      model more conservative.

- tree_method, string [default="auto"]

   - The tree construction algorithm used in XGBoost(see description
     in the reference paper)

   - Distributed and external memory version only support approximate
     algorithm.

   - Choices: {"auto", "exact", "approx", "hist", "gpu_exact", "gpu_hist"}

        - "auto": Use heuristic to choose faster one.

            - For small to medium dataset, exact greedy will be used.

            - For very large-dataset, approximate algorithm will be chosen.

            - Because old behavior is always use exact greedy in
              single machine, user will get a message when approximate
              algorithm is chosen to notify this choice.

        - "exact": Exact greedy algorithm.

        - "approx": Approximate greedy algorithm using sketching and histogram.

        - "hist": Fast histogram optimized approximate greedy
          algorithm. It uses some performance improvements such as
          bins caching.

        - "gpu_exact": GPU implementation of exact algorithm.

        -"gpu_hist": GPU implementat

- sketch_eps, [default=0.03]

    - This is only used for approximate greedy algorithm.

    - This roughly translated into O(1 / sketch_eps) number of
      bins. Compared to directly select number of bins, this comes
      with theoretical guarantee with sketch accuracy.

    - Usually user does not have to tune this. but consider setting to
      a lower number for more accurate enumeration.

    - range: (0, 1)

- scale_pos_weight, [default=1]

    - Control the balance of positive and negative weights, useful for
      unbalanced classes. A typical value to consider: sum(negative
      cases) / sum(positive cases) See Parameters Tuning for more
      discussion. Also see Higgs Kaggle competition demo for examples:
      R, py1, py2, py3

- updater, [default="grow_colmaker,prune"]

    - A comma separated string defining the sequence of tree updaters
      to run, providing a modular way to construct and to modify the
      trees. This is an advanced parameter that is usually set
      automatically, depending on some other parameters. However, it
      could be also set explicitely by a user. The following updater
      plugins exist:
        
        - "grow_colmaker": non-distributed column-based construction of trees.

        - "distcol": distributed tree construction with column-based
          data splitting mode.

        - "grow_histmaker": distributed tree construction with
          row-based data splitting based on global proposal of
          histogram counting.

        - "grow_local_histmaker": based on local histogram counting.

        - "grow_skmaker": uses the approximate sketching algorithm.

        - "sync": synchronizes trees in all distributed nodes.

        - "refresh": refreshes tree's statistics and/or leaf values
          based on the current data. Note that no random subsampling
          of data rows is performed.

        - "prune": prunes the splits where loss < min_split_loss (or gamma).

    - In a distributed setting, the implicit updater sequence value
      would be adjusted as follows:

        - "grow_histmaker,prune" when dsplit="row" (or default) and
          prob_buffer_row == 1 (or default); or when data has multiple
          sparse pages

        - "grow_histmaker,refresh,prune" when dsplit="row" and
          prob_buffer_row < 1

        - "distcol" when dsplit="col"

- refresh_leaf, [default=1]

    - This is a parameter of the "refresh" updater plugin. When this
      flag is true, tree leafs as well as tree nodes' stats are
      updated. When it is false, only node stats are updated.

- process_type, [default="default"]

    - A type of boosting process to run.

    - Choices: {"default", "update"}

        - "default": the normal boosting process which creates new trees.

        - "update": starts from an existing model and only updates its
          trees. In each boosting iteration, a tree from the initial
          model is taken, a specified sequence of updater plugins is
          run for that tree, and a modified tree is added to the new
          model. The new model would have either the same or smaller
          number of trees, depending on the number of boosting
          iteratons performed. Currently, the following built-in
          updater plugins could be meaningfully used with this process
          type: "refresh", "prune". With "update", one cannot use
          updater plugins that create new nrees.

- grow_policy, string [default="depthwise"]
    
    - Controls a way new nodes are added to the tree.

    - Currently supported only if tree_method is set to "hist".

    - Choices: {"depthwise", "lossguide"}

        - "depthwise": split at nodes closest to the root.

        - "lossguide": split at nodes with highest loss change.
- max_leaves, [default=0]

    - Maximum number of nodes to be added. Only relevant for the
      "lossguide" grow policy.

- max_bin, [default=256]

    - This is only used if "hist" is specified as tree_method.

    - Maximum number of discrete bins to bucket continuous features.

    - Increasing this number improves the optimality of splits at the
      cost of higher computation time.

- predictor, [default="cpu_predictor"]

    - The type of predictor algorithm to use. Provides the same
      results but allows the use of GPU or CPU.

        - "cpu_predictor": Multicore CPU prediction algorithm.

        - "gpu_predictor": Prediction using GPU. Default for
          "gpu_exact" and "gpu_hist" tree method.

## Learning Task Parameters 

Specify the learning task and the corresponding learning
objective. The objective options are below:

- objective [default=reg:linear]

    - "reg:linear" –linear regression

    - "reg:logistic" –logistic regression

    - "binary:logistic" –logistic regression for binary
      classification, output probability

    - "binary:logitraw" –logistic regression for binary
      classification, output score before logistic transformation

    - "count:poisson" –poisson regression for count data, output mean
      of poisson distribution max_delta_step is set to 0.7 by default in
      poisson regression (used to safeguard optimization)

    - "multi:softmax" –set XGBoost to do multiclass classification
      using the softmax objective, you also need to set
      num_class(number of classes)

    - "multi:softprob" –same as softmax, but output a vector of ndata
      * nclass, which can be further reshaped to ndata, nclass
      matrix. The result contains predicted probability of each data
      point belonging to each class.

    - "rank:pairwise" –set XGBoost to do ranking task by minimizing
      the pairwise loss

    - "reg:gamma" –gamma regression with log-link. Output is a mean of
      gamma distribution. It might be useful, e.g., for modeling
      insurance claims severity, or for any outcome that might be
      gamma-distributed

    - "reg:tweedie" –Tweedie regression with log-link. It might be
      useful, e.g., for modeling total loss in insurance, or for any
      outcome that might be Tweedie-distributed.

- base_score [default=0.5]

    - the initial prediction score of all instances, global bias for
      sufficient number of iterations, changing this value will not have too
      much effect.

- eval_metric [default according to objective]

    - evaluation metrics for validation data, a default metric will be
      assigned according to objective (rmse for regression, and error
      for classification, mean average precision for ranking )

    - User can add multiple evaluation metrics, for python user,
      remember to pass the metrics in as list of parameters pairs
      instead of map, so that latter "eval_metric" won’t override
      previous one

    - The choices are listed below: 

        - "rmse": root mean square error

        - "mae": mean absolute error

        - "logloss": negative log-likelihood

        - "error": Binary classification error rate. It is calculated
          as #(wrong cases)/#(all cases). For the predictions, the
          evaluation will regard the instances with prediction value
          larger than 0.5 as positive instances, and the others as
          negative instances.

        - "error@t": a different than 0.5 binary classification
          threshold value could be specified by providing a numerical
          value through "t".

        - "merror": Multiclass classification error rate. It is
          calculated as #(wrong cases)/#(all cases).

        - "mlogloss": Multiclass logloss

        - "auc": Area under the curve for ranking evaluation.

        - "ndcg":Normalized Discounted Cumulative Gain

        - "map":Mean average precision

        - "ndcg@n","map@n": n can be assigned as an integer to cut off
          the top positions in the lists for evaluation.

        - "ndcg-","map-","ndcg@n-","map@n-": In XGBoost, NDCG and MAP
          will evaluate the score of a list without any positive
          samples as 1. By adding "-" in the evaluation metric XGBoost
          will evaluate these score as 0 to be consistent under some
          conditions. training repeatedly

        - "poisson-nloglik": negative log-likelihood for Poisson regression

        - "gamma-nloglik": negative log-likelihood for gamma regression

        - "gamma-deviance": residual deviance for gamma regression

        - "tweedie-nloglik": negative log-likelihood for Tweedie
          regression (at a specified value of the
          tweedie_variance_power parameter)

- seed [default=0]

    - random number seed.


"""
type XGBoostRegressor <: BaseRegressor

    # hyperparameters:
    n::Int  # [default=20] number of boosting iterations
    eta::Float64    # [default=0.3, alias: learning_rate]
    gamma::Float64  # [default=0, alias: min_split_loss] penalty on number of leaves
    max_depth::Int  # [default=6]
    min_child_weight::Float64   # [default=1]
    subsample::Float64          # [default=1] proportion of rows to sampled to build each tree
    colsample_bytree::Float64   # [default=1] proportion of features to be sampled per tree
    colsample_bylevel::Float64  # [default=1] proportion of features to be sampled the node split
    lambda::Float64 # [default=1, alias: reg_lambda] L2 penalty on leaf predictions
    alpha::Float64  # [default=0, alias: reg_alpha] L1 penalty on leaf predictions
    tree_method::String         # string [default="auto"]
    objective::String           # objective function to minimized with regularization penalties
    
    # model (wrapped C object):
    model::XGBoost.Booster

    # post-fitted parameters
    
    fitted::Bool
    
    function XGBoostRegressor(n::Int, eta, gamma, max_depth::Int,
                              min_child_weight, subsample, colsample_bytree, colsample_bylevel,
                              lambda, alpha, tree_method::String, objective::String)
                              
        
        rgs = new(n, eta, gamma, max_depth,
                  min_child_weight,
                  subsample, colsample_bytree, colsample_bylevel,
                  lambda, alpha, tree_method, objective)
        
        rgs.fitted = false
        
        return rgs
        
    end
    
end

@declare_hyperparameters XGBoostRegressor [:n, :eta, :gamma, :max_depth,
                  :min_child_weight, :subsample, :colsample_bytree,
                  :colsample_bylevel, :lambda, :alpha, :tree_method,
                  :objective]

function show(stream::IO, rgs::XGBoostRegressor)
    tail(n) = "..."*string(n)[end-3:end]
    prefix = (rgs.fitted ? "" : "unfitted ")
    print(stream, prefix, "XGBoostRegressor@$(tail(hash(rgs)))")
end

function showall(stream::IO, rgs::XGBoostRegressor)
    show(stream, rgs); println()
    display(get_params(rgs))
end

# lazy keyword constructors:
## without fitting:
function XGBoostRegressor(;n=20, eta=0.3, gamma=0.0, max_depth=6,
                          min_child_weight=1.0,
                          subsample=1.0, colsample_bytree=1.0, colsample_bylevel=1.0,
                          lambda=1.0, alpha=0.0, tree_method="auto", objective="reg:linear")
    
    return XGBoostRegressor(n, eta, gamma, max_depth,
                  min_child_weight, subsample, colsample_bytree,
                  colsample_bylevel, lambda, alpha, tree_method,
                            objective)

end

## with immediate fitting:
XGBoostRegressor(X, y, bag; parameters...) = fit!(XGBoostRegressor(;parameters...), X, y, bag)

## with immediate fitting (bagless version):
XGBoostRegressor(X, y; parameters...) = XGBoostRegressor(X, y, 1:length(y); parameters...)

function fit!(rgs::XGBoostRegressor, X, y, bag; verbose=false)

    if length(y) == length(bag)
        XX = X
        yy = y
    else
        XX = X[bag,:]
        yy = y[bag]
    end

    parameters = get_params(rgs)

    n = parameters[:n]
    delete!(parameters, :n)
    parameters[:label] = yy

    rgs.model = XGBoost.xgboost(XX, n; silent=1, parameters...)

    rgs.fitted=true

    return rgs

end

function predict(rgs::XGBoostRegressor, Xtest::Array{Float64, 2}, bag=[])
    rgs.fitted || error("Attempting to predict using $rgs.")

    if isempty(bag)
        return XGBoost.predict(rgs.model, Xtest)
    else
        return XGBoost.predict(rgs.model, Xtest[bag,:])
    end
end
    
function cv(rgs::XGBoostRegressor, X, y, bag; n_folds=6, metric="rmse")

    if length(y) == length(bag)
        XX = X
        yy = y
    else
        XX = X[bag,:]
        yy = y[bag]
    end

    parameters = get_params(rgs)
    n = parameters[:n]
    delete!(parameters, :n)
    
    XGBoost.nfold_cv(XX, n, n_folds, label=yy, param=parameters, metrics=[metric], silent=1)

end

cv(rgs::XGBoostRegressor, X, y; params...) = cv(rgs, X, y, 1:length(y); params...)

"""

# `parse_cv_output(s)`

This function takes the text output `s` of the running cross-validation function `cv` and
outputs the test and train scores as array suitable for printing:


## Return value

`(iterations, errors)` where `iterations` is a vector of the boosting
step numbers and `errors` is a two-column matrix with the test and train scores.

**This routine uses regular expressions which have not been
  comprehensively tested and should be used with caution**

"""

function parse_cv_output(s::String)
    # Build regex to capture the float-string after "test-????:" where
    # "????" is some non-digit string of characters:
    rtest = r"cv-test-\D+:(\d+\.\d+)"   
    rtrain = r"cv-train-\D+:(\d+\.\d+)" 
    num_round = Int[]
    test = Float64[]
    train = Float64[]
    i = 1 # loop counter
    m = match(rtest,s)
    n = match(rtrain,s)
    while m != nothing 
        push!(num_round, i)
        push!(test, parse(Float64,m.captures[1]))
        push!(train, parse(Float64,n.captures[1]))
        i = i + 1
        m = match(rtest, s, m.offset + 1)
        n = match(rtrain, s, n.offset + 1)
    end
    return num_round, hcat(test, train)
end

######################
# Stacked Regressors #
######################

"""

# `StackedRegressor`

A two-layer stacked regressor, with a ScikitLearnBase API

## Sample use

Suppose `rgs1`, `rgs2`, `rgs3` are unfitted `BaseRegressor` objects to
be trained on input data `X` and target `y`. We suppose that each
regressor requires some preprocessing of the input, via transformers
`trf1`, `trf2`, `trf3`. For example

    julia> fit!(rgs1, trf1(X), y, train)

trains `rgs1` using all rows of `trf1` with indices in the iterator
`train`. The objective is to place the three regressors at the bottom
layer of a two-layer stack with a single regressor, `meta`, say, at
the second layer. Now suppose the model `meta` is to be trained
on meta-features obtained from 9-fold cross-predictions of the layer 1
regressors `rgs1`, `rgs2`, `rgs3`, which will constitute the columns
of a `m` X 3 matrix, where `m` is the number of patterns in the
training set. Let `meta_trf` be the transformer which converts this
matrix into a format suitable for `meta` (eg, a `DataFrame`
object). Then the stack is instantiated as follows:

    julia> stack = StackedRegressor(layer1=[rgs1, rgs2, rgs3],
                                    layer2=meta,
                                    transform01=[trf1, trf2,trf3],
                                    transform12=meta_trf,
                                    n_folds=9,          # default
                                    loss_function=rms)  # default

The `stack` is then trained as usual with a call to `fit!`:

    julia> fit!(stack, X, y, train; parallel = true)

However, this training can also be split into two phases:

    julia> prefit!(stack, X, y, train; parallel = true)

This constructs the matrix of meta features (stored as `stack.Xmeta`)
and trains the layer 1 regressors on all available data, but does not
train `meta`. The following call

    julia> fit!(stack, X, y, train; parallel = true, use_prefit=true)

will now train `meta` without recalculating the meta-features and
retraining the layer 0 regressors, unless no call to `prefit!` or
`fit!` has been previously made. Since training `meta` is usually very
fast once the meta-features are computed, it is convenient to tune
`meta`'s parameters with such calls to `fit!`.

**Warning:** Do not set `use_prefit` to `true` when calling `cv_errors`
on `stack` as each call to `fit!` in this case will be on new
data. (Internal checks will generally throw an exception in this
case.)

After pre-fitting, cross-validation errors of the layer 1 regressors
are available as `stack.cv_errors`.

"""
type StackedRegressor{T<:BaseRegressor} <: BaseRegressor

    # hyperparameters
    n_folds::Int
    layer2::T                       # Note that this *is* trained during fitting
    transform12::Function
    layer1::Vector{BaseRegressor}  # Note these *are* trained during prefitting
    transform01::Vector{Function}
    loss_function::Function # For internal computation of cv_error of each base model

    # data set at pre-fit
    n_features::Int
    Xmeta::Array{Float64,2}    # The matrix of meta features
    target::Vector{Float64}    # Corresponding target values (to be checked during final fit)
    cv_errors::Vector{Float64} # Internally computed cv errors of base models
    prefitted::Bool

    # model: The final model consists of the trained versions of
    # layer2 and base_regessors above
    
    # post-fit parameters
    fitted::Bool

    function StackedRegressor{S}(n_folds, layer2::S, transform12::Function,
                                 layer1::Vector{BaseRegressor},
                                 transform01::Vector{Function}, loss_function::Function)
        length(layer1) == length(transform01) || throw(DimensionMismatch())
        ret = new(n_folds, layer2, transform12, layer1, transform01, loss_function)
        ret.prefitted = false
        ret.fitted = false
        return ret
    end

end

StackedRegressor{T}(n_folds, layer2::T, transform12::Function,
                    layer1::Vector,
                    transform01::Vector{Function},
                    loss_function::Function) = StackedRegressor{T}(n_folds, layer2,
                                                                   transform12,
                                                                   layer1,
                                                                   transform01,
                                                                   loss_function)

@provide_parameter_interface StackedRegressor [:n_folds, :layer2, :transform12,
                                               :layer1, :transform01, :loss_function]

function clone(stack::StackedRegressor)
    return StackedRegressor(stack.n_folds, clone(stack.layer2), stack.transform12,
                           BaseRegressor[clone(rgs) for rgs in stack.layer1],
                           stack.transform01, stack.loss_function)
end

function show{T}(stream::IO, rgs::StackedRegressor{T})
    tail(n) = "..."*string(n)[end-3:end]
    if rgs.fitted
        prefix = ""
    elseif rgs.prefitted
        prefix = "prefitted "
    else
        prefix = "unfitted "
    end
    print(stream, prefix, "StackedRegressor{$T}@$(tail(hash(rgs)))")
end

function showall(stream::IO, rgs::StackedRegressor)
    show(stream, rgs); println(stream, "\nHyperparameters:")
    display(get_params(rgs))
    if rgs.prefitted
        println("Post-fit parameters:")
        d=Dict{Symbol,Any}()
        d[:cv_errors] = rgs.cv_errors
        d[:n_features] = rgs.n_features
        display(d)
    end
end

# lazy keyword constructors:
## without fitting:
StackedRegressor(;n_folds=10, layer2=TreeRegressor(),
                 transform12=identity_transformer,
                 layer1=BaseRegressor[TreeRegressor()],
                 transform01=Function[identity_transformer],
                 loss_function=rms) = StackedRegressor(n_folds, layer2,
                                                             transform12,
                                                             convert(Vector{BaseRegressor},
                                                                     layer1),
                                                             convert(Vector{Function},
                                                                     transform01),
                                                             loss_function)
                              
## with immediate fitting:
function StackedRegressor(X, y, bag; params...)
    parameters = Dict(params)
    @extract_from parameters parallel true
    @extract_from parameters use_prefit false
    return prefit!(StackedRegressor(;parameters...), X, y, bag; parallel=parallel, use_prefit=use_prefit)
end

## with immediate fitting (bagless version):
StackedRegressor(X, y; parameters...) = StackedRegressor(X, y, 1:length(y); parameters...)

prefit!(stack::StackedRegressor, X, y; params...) = prefit!(stack, X, y, 1:length(y) ; params...)

function prefit!(stack::StackedRegressor, X, y, bag; parallel=false)

    # Note: In code doc below, "model" and "regressor" are synonyms

    # dimension checks:
    length(stack.layer1) == length(stack.transform01) || throw(DimensionMismatch("Number of base regressors and base transformers must be the same."))

    n_patterns = length(bag)
    stack.n_features = size(X, 2)
    
    # unpack data according to row bag passed:
    if length(y) == n_patterns
        XX = X
        yy = y
    else
        XX = X[bag,:]
        yy = y[bag]
    end

    function cv_predict(rgs::BaseRegressor, X, y)

        K = floor(Int, n_patterns/stack.n_folds)

        # function to return the predictions for the fold with row
        # indices `f:s` with zero prediction on the remaining
        # indices:
        function pred(f, s)
            prediction = zeros(Float64, n_patterns)
            train_bag = vcat(1:(f - 1), (s + 1):n_patterns)
            fit!(rgs, X, y, train_bag)
            test_prediction = predict(rgs, X, f:s) # only K long
            k = 1
            for i in f:s
                prediction[i] = test_prediction[k]
                k += 1
            end
            return prediction
        end
        
        firsts = 1:K:((stack.n_folds - 1)*K + 1) # itr of first test_bag index
        seconds = K:K:(stack.n_folds*K)          # itr of ending test_bag index

        # append last left-over fold if there is one:
        if stack.n_folds*K != n_patterns
            firsts = vcat(firsts, stack.n_folds*K + 1)
            seconds = vcat(seconds, n_patterns)
        end

        if parallel && nworkers() > 1
            println("Distributing cross-validation computation among $(nworkers()) workers.")
            return @parallel (+) for nf in eachindex(firsts)
                pred(firsts[nf], seconds[nf])
            end
        else # to do: next bit not optimal
            return sum([pred(firsts[nf], seconds[nf]) for nf in eachindex(firsts)])
        end
        
    end

    n_models = length(stack.layer1)

    # transform the base model input data:
    Xs = Any[stack.transform01[k](XX) for k in 1:n_models]

    stack.cv_errors = Array(Float64, n_models)

    # build meta features matrix, `Xmeta` (n_fold-cross predictions of layer1 models):
    Xmeta = Array(Float64,(length(bag), n_models))
    for k in 1:n_models
        println("Building meta feature number $k...")
        yhat = cv_predict(stack.layer1[k], Xs[k], yy)
        Xmeta[:,k] = yhat
        # record cv_error
        stack.cv_errors[k] = stack.loss_function(yhat, yy)
    end
    stack.Xmeta = Xmeta
    stack.target = yy

    # train the base regressors on all the data
    for k in 1:n_models
        println("Training base regressor number $k...")
        fit!(stack.layer1[k], Xs[k], yy)
    end
        
    stack.prefitted = true

    return stack
end

function fit!(stack::StackedRegressor, X, y, bag;
              parallel=false, use_prefit=false, verbose=false)

    # Note: In code doc below, "model" and "regressor" are synonyms

    n_patterns = length(bag)

    # unpack data according to row bag passed:
    if length(y) == n_patterns
        yy = y
    else
        yy = y[bag]
    end

    # Pre-fit the data if necessary:
    if !stack.prefitted || !use_prefit
        prefit!(stack, X, y, bag; parallel=parallel)
    else
        # Check data for consistency with pre-fit data:
        err1 = "Number of patterns presented different from number in pre-fit. "*
        "Called fit! with correct row bag?"
        err2 = "Regressor pre-fitted with different target."
        n_patterns == size(stack.Xmeta, 1) || error(err1)
        yy == stack.target || error(err2)
    end
    
    # train layer2 on the meta features
    println("\nTraining the meta regressor...")
    fit!(stack.layer2, stack.transform12(stack.Xmeta), yy)

    stack.fitted = true

    return stack

end

function predict(stack::StackedRegressor, X, bag=[])
    stack.fitted || error("Attempting to predict using $stack.")
    stack.n_features == size(X, 2) || throw(DimensionMismatch())

    if isempty(bag)
        bag = 1:size(X, 1)
    end
    
    n_patterns = length(bag)
    n_models = length(stack.layer1)
    
    Xmeta = Array(Float64, (n_patterns, n_models))
    for k in eachindex(stack.layer1)
        Xmeta[:,k] = predict(stack.layer1[k], stack.transform01[k](X), bag)
    end

    return predict(stack.layer2, stack.transform12(Xmeta))
end


end # of module

##############################################################
# Comments on inner contructor methods for parametric types: #
##############################################################

# kmsquire commented on Aug 27, 2014: https://github.com/JuliaLang/julia/issues/8135

# Not sure if this helps, but the type parameters have somewhat different meanings in types and outer constructors.

# For types, the type parameter is part of the type name. So T{Int,5} is the type, and T is an abstract supertype.

# On the other hand, outer constructors are just functions, so the type parameter indicates that the Type should be extracted from the passed in parameters, and then is itself usable in the function, e.g., when creating a parametrized type. So in

# T{N}(a::N) = T{N,5}()
# the first N is set to the type of a (As in any function) and is then used to construct an object of type T{N,5}.



