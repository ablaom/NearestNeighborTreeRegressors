__precompile__()
# August 2017
# THIS IS AN EXTRACT OF THE ORIGINAL MODULE INCLUDING ONLY THE DECISION TREE MODEL
module Regressors

export TreeRegressor, feature_importance
export ConstantRegressor
export row, show, get_params, set_params!, add!
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

end # of module
