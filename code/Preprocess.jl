__precompile__()
module Preprocess

export DiscretizationScheme, inverse_transform
export how_much_not_a_function, reporting_period
export compare, number_of_nas, nas_after_first_non_na, row_of_first_non_na, remove_steps
export get_meta, review_ordinals!
export drop_and_pair
export spawn_large_categorical!, rank_features
export StringToIntScheme, BoxCoxScheme, StandardizationScheme, HotEncodingScheme 
export normalise, normality, UnivariateStandardizationScheme
export UnivariateHotEncodingScheme, UnivariateBoxCoxScheme

# extended
export fit!, transform, get_params, predict

import ADBUtilities: @colon, @extract_from, argmin, argmax, keys_ordered_by_values, @dbg
import DataFrames: DataFrame, DataArray, isna
import TreeCollections.DataTable
import UnicodePlots
import Regressors: LoessRegressor, BaggedRegressor, TreeRegressor
import Validation.vary_float_parameter
import Distributions
import Distributions.countmap
import ScikitLearn.@declare_hyperparameters

# to be extended:
import Base: show, showall
import ScikitLearnBase: fit!, transform, get_params, predict

typealias Small UInt8
const SMALL_MAX = Small(52)

"""
type DiscretizationScheme(x; n_classes=30)

Abstractly, a *discretization scheme* is any surjective,
order-preserving mapping from reals to a finite subset of integers
with no gaps (consecutive). A `Discretization Scheme` object contains
data sufficient to encode such a map, and encode an approximate
'inverse'.  Use such a scheme to discretize numerical data, e.g., to
convert a float-valued feature into an ordinal feature. The
constructor is fed a 1D array `x` on which to base the discretization,
and a parameter `n_classes` describing the resolution of the
discretization, as shown above. Discretization is chosen so as to
create classes with uniform membership within the input array. This
objective is not exaclty realized, and in particular in those cases
where x has many repeated values.

## Methods

* To transform a real number r to an integer using a Discretization Scheme `s` call transform(r, s).
* To obtain an 'approximate inverse' transform of integer `k`, call inverse_transform(k, s).
* Both above methods also take any DataArray as first argument, returning a DataArray.
"""
type DiscretizationScheme
    n_classes
    odd_quantiles
    even_quantiles
end

function DiscretizationScheme(x; n_classes=30)
    quantiles = quantile(x, Array(linspace(0,1,2*n_classes+1)))  
    clipped_quantiles = quantiles[2:2*n_classes] # drop 0% and 100% quantiles
    
    # odd_quantiles for transforming, even_quantiles used for inverse_transforming:
    odd_quantiles = clipped_quantiles[2:2:(2*n_classes-2)]
    even_quantiles = clipped_quantiles[1:2:(2*n_classes-1)]

    return DiscretizationScheme(n_classes, odd_quantiles, even_quantiles)

end

function transform(r, scheme::DiscretizationScheme)
    k = 1
    for level in scheme.odd_quantiles
        if r > level
            k = k + 1
        end
    end
    return k
end

function transform(v::AbstractVector, scheme::DiscretizationScheme)
    N = length(v)
    ret = DataArray(Int64,N)
    for i in 1:N
        k = 1
        for level in scheme.odd_quantiles
            if v[i] > level
                k = k + 1
            end
        end
        ret[i]=k
    end
    return ret
end
        
function inverse_transform(k, scheme::DiscretizationScheme)
    if k < 1
        return scheme.even_quantiles[1]
    elseif k > scheme.n_classes
        return scheme.even_quantiles[scheme.n_classes]
    end
    return scheme.even_quantiles[k]
end

function inverse_transform(J::AbstractVector, scheme::DiscretizationScheme)
    N = length(J)
    ret = DataArray(Float64,N)
    for i in 1:N
        k = J[i]
        if k < 1
            ret[i] = scheme.even_quantiles[1]
        elseif k > scheme.n_classes
            ret[i] = scheme.even_quantiles[scheme.n_classes]
        else
            ret[i] = scheme.even_quantiles[k]
        end    
            ret[i]=scheme.even_quantiles[J[i]]
    end
    return ret
end

######################
# BoxCox transformer #
######################

function normalise(v)
    map(v) do x
        (x - mean(v))/std(v)
    end
end
                   
function normality(v)

    n  = length(v)
    v = normalise(convert(Vector{Float64}, v))

    # sort and replace with midpoints
    v = midpoints(sort!(v))

    # find the (approximate) expected value of the size (n-1)-ordered statistics for
    # standard normal:
    d = Distributions.Normal(0,1)
    w= map(collect(1:(n-1))/n) do x
        quantile(d, x)
    end

    return cor(v, w)

end

function boxcox{T<:Real}(lambda, c, x::T)
    c + x <= 0 ? throw(DomainError) : (lambda == 0.0 ? log(c + x) : ((c + x)^lambda - 1)/lambda)
end

boxcox(lambda, c, v::Vector) = [boxcox(lambda, c, x) for x in v]    

function boxcox(v::Vector; n=171, shift::Bool = false)
    m = minimum(v)
    m >= 0 || throw(DomainError)

    c = 0.0 # default
    if shift
        if m == 0
            c = 0.2*mean(v)
        end
    else
        m != 0 || throw(DomainError) 
    end
  
    lambdas = linspace(-0.4,3,n)
    scores = [normality(boxcox(l, c, v)) for l in lambdas]
    lambda = lambdas[indmax(scores)]
    return  lambda, c, boxcox(lambda, c, v)
end

"""
# `type UnivariateBoxCoxScheme`

A wrapper for the data describing a Box-Cox transformation of a single
variable taking non-negative values, with a possible preliminary
shift. Recall that this transformation is of the form 

    x -> ((x + c)^λ - 1)/λ for λ not 0
    x -> log(x + c) for λ = 0

##  `s = UnivariateBoxCoxScheme(; n=171, shift=false)`

Returns an unfitted wrapper that on fitting to data (see below) will
try `n` different values of the Box-Cox exponent λ (between `-0.4` and
`3`) to find an optimal value, stored as the post-fit parameter
`s.lambda`. If `shift=true` and zero values are encountered in the
data then the transformation sought includes a preliminary positive
shift, stored as `s.c`. The value of the shift is always `0.2` times
the data mean. If there are no zero values, then `s.c=0`.

## `fit!(s, v)`

Attempts fit an `UnivariateBoxCoxScheme` instance `s` to a
vector `v` (eltype `Real`). The elements of `v` must
all be positive, or a `DomainError` will be thrown. If `s.shift=true`
zero-valued elements are allowed as discussed above. 

## `s = UnivariateBoxCoxScheme(v; n=171, shift=false)`

Combines the previous two steps into one.

## `w = transform(s, v)`

Transforms the vector `v` according to the Box-Cox transformation
encoded in the `UnivariateBoxCoxScheme` instance `s` (which must be
first fitted to some data). Stores the answere as `w`.

See also `BoxCoxScheme` a transformer for selected ordinals in a DataTable. 

"""

type UnivariateBoxCoxScheme
    
    # hyperparameters
    n::Int
    shift::Bool
    
    # post-fit parameters:
    lambda::Float64
    c::Float64

    function UnivariateBoxCoxScheme(n::Int, shift::Bool)
        ret = new(n, shift)
        ret.c = -1.0 # indicating not scheme not yet fitted
        return ret
    end
    
end

UnivariateBoxCoxScheme(; n=171, shift=false) = UnivariateBoxCoxScheme(n, shift)

function get_params(s::UnivariateBoxCoxScheme)
    d = Dict{Symbol, Any}()
    d[:n] = s.n; d[:shift] = s.shift
    return d
end

function show(stream::IO, s::UnivariateBoxCoxScheme)
    if s.c >= 0
        print(stream, "UnivariateBoxCoxScheme(($(s.lambda), $(s.c))")
    else
        print(stream, "unfitted UnivariateBoxCoxScheme()")
    end
end

function showall(stream::IO, s::UnivariateBoxCoxScheme)
    show(stream, s); println()
    println("Hyperparameters:")
    display(get_params(s))
    if s.c >= 0
        println("Post-fit parameters:")
        d = Dict{Symbol,Any}()
        d[:lambda] = s.lambda
        d[:c] = s.c
        display(d)
    end
end

function fit!(s::UnivariateBoxCoxScheme, v::Vector)
    s.lambda, s.c, _ = boxcox(v; n=s.n, shift=s.shift)
    return s
end

function UnivariateBoxCoxScheme(v::Vector; parameters...)
    s = UnivariateBoxCoxScheme(; parameters...)
    return fit!(s, v)
end

function transform{T<:Real}(s::UnivariateBoxCoxScheme, x::T)
    if s.c < 0
        throw(Base.error("Attempting to transform according to unfitted scheme."))
    end
    return boxcox(s.lambda, s.c, x)
end

transform(s::UnivariateBoxCoxScheme, v::Vector) = boxcox(s.lambda, s.c, v)

"""
## `type BoxCoxScheme`

Wrapper for the data needed to apply Box-Cox transformations to each ordinal fields of a `DataFrame` object.

### Method calls

    julia> s = BoxCoxScheme(X)    # calculate the transformation scheme appropriate for data frame `X`
    julia> XX = transform(s, Y) # transform data frame `Y` according to the scheme `s`
    
### Keyword arguments

Calls to the first method above may be issued with the following keyword arguments:

- `shift=true`: allow data shift in case of fields taking zero values
(otherwise no transformation will be applied).

- `n=171`: number of values of exponent `lambda` to try during optimization.

## See also

`UnivariateBoxCoxScheme`: The single variable version of the scheme implemented by `BoxCoxScheme`.

"""

type BoxCoxScheme

    # hyperparameters:
    n::Int                     # number of values considered in exponent optimizations
    shift::Bool                # whether or not to shift features taking zero as value
    features::Vector{Symbol}   # features to attempt fitting a transformation (empty means all)
    
    # post-fit parameters:
    schemes::Vector{UnivariateBoxCoxScheme}
    is_transformed::Vector{Bool}

    fitted::Bool
    
    function BoxCoxScheme(n::Int, shift::Bool, features::Vector{Symbol})
        ret = new(n, shift, features)
        ret.fitted = false
        return ret
    end
    
end

BoxCoxScheme(; n=171, shift = false, features=Symbol[]) = BoxCoxScheme(n, shift, features)

@declare_hyperparameters BoxCoxScheme [:n, :shift, :features]

function BoxCoxScheme(X; parameters...)
    s =  BoxCoxScheme(; parameters...)
    fit!(s, X) # defined below
    return s
end

function get_params(s::BoxCoxScheme)
    d = Dict{Symbol, Any}()
    d[:n] = s.n; d[:shift] = s.shift; d[:features] = s.features
    return d
end

function show(stream::IO, s::BoxCoxScheme)
    tail(n) = "..."*string(n)[end-3:end]
    print(stream, "BoxCoxScheme@$(tail(hash(s)))")
end

function showall(stream::IO, s::BoxCoxScheme)
    show(stream, s); println()
    display(get_params(s))
    if s.fitted
        display(s.schemes)
    end
end

function fit!(s::BoxCoxScheme, X::DataTable)

    # determine indices of features to be transformed
    features_to_try = (isempty(s.features) ? X.names : s.features)
    s.is_transformed = Array(Bool, X.ncols)
    for j in 1:X.ncols
        if X.names[j] in features_to_try && X.scheme.is_ordinal[j] && minimum(X[j]) >= 0
            s.is_transformed[j] = true
        else
            s.is_transformed[j] = false
        end
    end

    # fit each of those features with best Box Cox transformation
    s.schemes = Array(UnivariateBoxCoxScheme, X.ncols)
    println("Box-Cox transformations: ")
    for j in 1:X.ncols
        if s.is_transformed[j]
            if minimum(X[j]) == 0 && !s.shift
                println("  :$(X.names[j])    (*not* transformed, contains zero values)")
                s.is_transformed[j] = false
                s.schemes[j] = UnivariateBoxCoxScheme()
            else                
                n_values = length(countmap(X[j]))
                if n_values < 16
                    println("  :$(names(X)[j])    (*not* transformed, too few values)")
                    s.is_transformed[j] = false
                    s.schemes[j] = UnivariateBoxCoxScheme()
                else                    
                    uscheme = UnivariateBoxCoxScheme(X[j]; shift=s.shift, n=s.n)
                    if uscheme.lambda in [-0.4, 3]
                        println("  :$(X.names[j])    (*not* transformed, lambda too extreme)")
                        s.is_transformed[j] = false
                        s.schemes[j] = UnivariateBoxCoxScheme()
                    elseif uscheme.lambda == 1.0
                        println("  :$(X.names[j])    (*not* transformed, not skewed)")
                        s.is_transformed[j] = false
                        s.schemes[j] = UnivariateBoxCoxScheme()
                    else
                        s.schemes[j] = uscheme 
                        println("  :$(X.names[j])    lambda=$(s.schemes[j].lambda)  shift=$(s.schemes[j].c)")
                    end
                end
            end
        else
            s.schemes[j] = UnivariateBoxCoxScheme()
        end
    end

    s.shift ? nothing : println("To transform non-negative features with zero values use shift=true.")
    
    s.fitted = true
    
    return s
end

function fit!(s::BoxCoxScheme, X::DataFrame)

    # determine indices of features to be transformed
    features_to_try = (isempty(s.features) ? names(X) : s.features)
    s.is_transformed = Array(Bool, size(X, 2))
    for j in 1:size(X, 2)
        if names(X)[j] in features_to_try && eltype(X[j]) <: Real && minimum(X[j]) >= 0
            s.is_transformed[j] = true
        else
            s.is_transformed[j] = false
        end
    end

    # fit each of those features with best Box Cox transformation
    s.schemes = Array(UnivariateBoxCoxScheme, size(X, 2))
    println("Box-Cox transformations: ")
    for j in 1:size(X,2)
        if s.is_transformed[j]
            if minimum(X[j]) == 0 && !s.shift
                println("  :$(names(X)[j])    (*not* transformed, contains zero values)")
                s.is_transformed[j] = false
                s.schemes[j] = UnivariateBoxCoxScheme()
            else
                n_values = length(countmap(X[j]))
                if n_values < 16
                    println("  :$(names(X)[j])    (*not* transformed, too few values)")
                    s.is_transformed[j] = false
                    s.schemes[j] = UnivariateBoxCoxScheme()
                else                    
                    uscheme = UnivariateBoxCoxScheme(collect(X[j]); shift=s.shift, n=s.n)
                    if uscheme.lambda in [-0.4, 3]
                        println("  :$(names(X)[j])    (*not* transformed, lambda too extreme)")
                        s.is_transformed[j] = false
                        s.schemes[j] = UnivariateBoxCoxScheme()
                    elseif uscheme.lambda == 1.0
                        println("  :$(names(X)[j])    (*not* transformed, not skewed)")
                        s.is_transformed[j] = false
                        s.schemes[j] = UnivariateBoxCoxScheme()
                    else
                        s.schemes[j] = uscheme 
                        println("  :$(names(X)[j])    lambda=$(s.schemes[j].lambda)  shift=$(s.schemes[j].c)")
                    end
                end
            end
        else
            s.schemes[j] = UnivariateBoxCoxScheme()
        end
    end

    s.shift ? nothing : println("To transform non-negative features with zero values use shift=true.")
    
    s.fitted = true
    
    return s
end

# for Scikitlearn API
fit!(s::BoxCoxScheme, X, y) = fit!(s,X)

function transform(s::BoxCoxScheme, X::DataTable)
    Xnew = copy(X)
    for j in 1:X.ncols
        if s.is_transformed[j]
            try
                Xnew.raw[:,j] = transform(s.schemes[j], X[j])
            catch DomainError
                error("Data outside of the domain of fitted Box-Cox"*
                        " transformation encountered in feature $(X.names[j]).")
            end
        end
    end
    return Xnew
end

function transform(s::BoxCoxScheme, X::DataFrame)
    Xnew = copy(X)
    for j in 1:size(X, 2)
        if s.is_transformed[j]
            try
                Xnew[j] = transform(s.schemes[j], collect(X[j]))
            catch DomainError
                error("Data outside of the domain of fitted Box-Cox"*
                        " transformation encountered in feature $(names(df)[j]). Transformed to zero.")
            end
        end
    end
    return Xnew
end

################################
# Standardization transformers #
################################

type UnivariateStandardizationScheme
    
    # hyperparameters: None
    
    # post-fit parameters:
    mu::Float64
    sigma::Float64

    fitted::Bool

    function UnivariateStandardizationScheme()
        ret = new()
        ret.fitted = false
        return ret
    end
    
end

function show(stream::IO, s::UnivariateStandardizationScheme)
    tail(n) = "..."*string(n)[end-3:end]
    if s.fitted
        print(stream, "UnivariateStandardizationScheme($(s.mu), $(s.sigma))")
    else
        print(stream, "unfitted UnivariateStandardizationScheme()")
    end
end

function showall(stream::IO, s::UnivariateStandardizationScheme)
    show(stream, s); println()
    if s.fitted
        println("Post-fit parameters:")
        d = Dict{Symbol,Any}()
        d[:mu] = s.mu
        d[:sigma] = s.sigma
        display(d)
    end
end

function fit!(s::UnivariateStandardizationScheme, v::Vector)
    s.mu, s.sigma = (mean(v), std(v))
    s.fitted = true
    return s
end

function UnivariateStandardizationScheme(v::Vector)
    s = UnivariateStandardizationScheme()
    return fit!(s, v)
end

function transform{T<:Real}(s::UnivariateStandardizationScheme, x::T)
    if !s.fitted
        throw(Base.error("Attempting to transform according to unfitted scheme."))
    end
    return (x - s.mu)/s.sigma
end
transform(s::UnivariateStandardizationScheme, v::Vector) = [transform(s,x) for x in v]
    
type StandardizationScheme

    # hyperparameters:
    features::Vector{Symbol}
    
    # post-fit parameters:
    schemes::Vector{UnivariateStandardizationScheme}
    is_transformed::Vector{Bool}

    fitted::Bool
    
    function StandardizationScheme(features::Vector{Symbol})
        ret = new(features)
        ret.fitted = false
        return ret
    end
    
end

StandardizationScheme(; features=Symbol[]) = StandardizationScheme(features)

function StandardizationScheme(X; parameters...)
    s = StandardizationScheme(;parameters...)
    fit!(s, X) # defined below
    return s
end

function get_params(s::StandardizationScheme)
    d = Dict{Symbol, Any}()
    d[:features] = s.features
    return d
end

function show(stream::IO, s::StandardizationScheme)
    tail(n) = "..."*string(n)[end-3:end]
    print(stream, "StandardizationScheme@$(tail(hash(s)))")
end

function showall(stream::IO, s::StandardizationScheme)
    show(stream, s); println()
    display(get_params(s))
    try
        display(s.schemes)
    catch
        nothing
    end
end

function fit!(s::StandardizationScheme, X::DataTable)

    # determine indices of features to be transformed
    features_to_try = (isempty(s.features) ? X.names : s.features)
    s.is_transformed = Array(Bool, X.ncols)
    for j in 1:X.ncols
        if X.names[j] in features_to_try && X.scheme.is_ordinal[j] 
            s.is_transformed[j] = true
        else
            s.is_transformed[j] = false
        end
    end

    # fit each of those features
    s.schemes = Array(UnivariateStandardizationScheme, X.ncols)
    println("Features standardized: ")
    for j in 1:X.ncols
        if s.is_transformed[j]
            s.schemes[j] = UnivariateStandardizationScheme(X[j])
            println("  :$(X.names[j])    mu=$(s.schemes[j].mu)  sigma=$(s.schemes[j].sigma)")
        else
            s.schemes[j] = UnivariateStandardizationScheme()
        end
    end

    s.fitted = true
    
    return s
end

function fit!(s::StandardizationScheme, X::DataFrame)

    # determine indices of features to be transformed
    features_to_try = (isempty(s.features) ? names(X) : s.features)
    s.is_transformed = Array(Bool, size(X, 2))
    for j in 1:size(X, 2)
        if names(X)[j] in features_to_try && eltype(X[j]) <: Real
            s.is_transformed[j] = true
        else
            s.is_transformed[j] = false
        end
    end

    # fit each of those features
    s.schemes = Array(UnivariateStandardizationScheme, size(X, 2))
    println("Features standarized: ")
    for j in 1:size(X, 2)
        if s.is_transformed[j]
            s.schemes[j] = UnivariateStandardizationScheme(collect(X[j]))
            println("  :$(names(X)[j])    mu=$(s.schemes[j].mu)  sigma=$(s.schemes[j].sigma)")
        else
            s.schemes[j] = UnivariateStandardizationScheme()
        end
    end

    s.fitted = true
    
    return s
end
                                                 
# for Scikitlearn API
fit!(s::StandardizationScheme, X::DataTable, y) = fit!(s,X)

function transform(s::StandardizationScheme, X::DataTable)
    s.fitted ? nothing : throw(Base.error("Attempting to transform according to unfitted scheme."))
    Xnew = copy(X)
    for j in 1:X.ncols
        if s.is_transformed[j]
            Xnew.raw[:,j] = transform(s.schemes[j], X[j])
        end
    end
    return Xnew
end

function transform(s::StandardizationScheme, X::DataFrame)
    s.fitted ? nothing : throw(Base.error("Attempting to transform according to unfitted scheme."))
    Xnew = copy(X)
    for j in 1:size(X, 2)
        if s.is_transformed[j]
            Xnew[j] = transform(s.schemes[j], collect(X[j]))
        end
    end
    return Xnew
end
                                                 
####################################
# One-hot encoding transformations #
####################################

type HotEncodingScheme

    # hyperparameters:
    drop_last::Bool
    

    # post-fit params:
    names::Vector{Symbol} # names of all features of fitted DataFrame
    values_given_name::Dict{Symbol,Vector{String}} # `schemes[name]` is
                                                   # a vector of
                                                   # values taken on
                                                   # by the feature
                                                   # `name`, for each
                                                   # categorical
                                                   # feature with
                                                   # symbol `name' in the fitted DataFrame

    fitted::Bool

    function HotEncodingScheme(;drop_last::Bool=false)
        ret = new(drop_last)
        ret.fitted = false
        return ret
    end

end

@declare_hyperparameters HotEncodingScheme [:drop_last]

function HotEncodingScheme(X::DataFrame; params...)
    s= HotEncodingScheme(;params...)
    fit!(s, X) # defined later
    return s
end

function show(stream::IO, s::HotEncodingScheme)

    tail(n) = "..."*string(n)[end-3:end]

    if s.fitted
        print(stream, "HotEncodingScheme@$(tail(hash(s)))")
    else
        print(stream, "unfitted HotEncodingScheme()")
    end

end

function showall(stream::IO, s::HotEncodingScheme)
    show(stream, s); println()
    if s.fitted
        display(get_params(s))
        for feature in keys(s.values_given_name)
            println(stream, "$feature will split as $(length(s.values_given_name[feature])) ordinals")
        end
    end
end

function fit!(s::HotEncodingScheme, X::DataFrame)
    s.names = names(X)
    s.values_given_name = Dict{Symbol,Vector{String}}()
    for j in 1:size(X, 2)
        if eltype(X[j]) == String
            s.values_given_name[s.names[j]] = sort!(collect(keys(Distributions.countmap(X[j]))))
            s.drop_last ? s.values_given_name[s.names[j]] = s.values_given_name[s.names[j]][1:(end - 1)] : nothing
        end
    end
    s.fitted = true
    return s
end

function transform(s::HotEncodingScheme, X::DataFrame)
    Xout = DataFrame()
    for feature_name in s.names
        if eltype(X[feature_name]) == String
            for value in s.values_given_name[feature_name]
                subfeature_name = Symbol(string(feature_name,"__",value))
                while subfeature_name in s.names
                    subfeature_name = Symbol(string(subfeature_name,"_"))
                end
                Xout[subfeature_name] = map(collect(X[feature_name])) do x
                    x == value ? 1.0 : 0.0
                end 
            end
        else
            Xout[feature_name] = X[feature_name]
        end
    end
    return Xout
end

######################################

""" 
# compare(x,y)

Returns true if `x=y` in the case neither variable is NA. Otherwise returns true only if both variables are NA. This function is need because of poisonous nature of NA, for `NA == NA` returns `NA` instead of `true`.
"""
function compare(x,y)
    if isna(x)
        if isna(y)
            return true
        else
            return false
        end
    else
        if isna(y)
            return false
        else
            return x == y
        end
    end
end

"""
# how_much_not_a function(v,w)

Tests to see whether the relation defined by pairs of corresponding elements of the vectors `v` and `w` is a function. 

## Return value

The number of times of elements of `w` that must be changed to make the relation a function. 
"""
function how_much_not_a_function(v,w)
    d = Dict()
    ret = 0
    n = length(v)
    for i in 1:n
        if v[i] in keys(d)
           
           if !compare(w[i], d[v[i]]) # need to test strong equality to handle NA's
                ret = ret + 1
           end
        else
            d[v[i]] = w[i]
        end
    end
    return ret
end

number_of_nas(s) = sum(map(isna, s))


"""
# function row_of_first_non_na(s)

Returns the index of the first row of the one-dimensional DataArray s
that is not NA in value. If `s` is *all* then index of last row is returned is returned

"""
function row_of_first_non_na(s)
    for r in 1:length(s)
        if !isna(s[r])
            return r
        end
    end
    return length(s)
end

"""
# function nas_after_first_non_na(s)

Returns the number of NA entries after the first non-NA entry in a
one-dimensional DataArray.

"""
function nas_after_first_non_na(s)
    nas = 0
    first = row_of_first_non_na(s)
    if isna(first)
        return 0
    end
    for r in first:length(s)
        if isna(s[r])
            nas = nas + 1
        end
    end
    return nas
end

function get_meta(df::DataFrame)
    n_patterns = size(df)[1]
    meta = DataFrame()
    meta[:field]=names(df)
    meta[:type] = map(f -> eltype(df[f]), meta[:field])
    meta[:n_values] = map(f -> length(keys(countmap(df[f]))), meta[:field])
    meta[:n_nas]=map(f -> number_of_nas(df[f]), meta[:field])
    meta[:percent_nas]=map(n -> round(Int, 1000*n/n_patterns)/10.0, meta[:n_nas])
    meta[:row_of_first_non_na] = map(f -> row_of_first_non_na(df[f]), meta[:field])
    meta[:nas_after_first_non_na]=map(f -> nas_after_first_non_na(df[f]), meta[:field])
    return meta
end

"""
## `function review_ordinals!(df::DataFrame)`

User-interactive review of those fields in `df` whose `eltype` is a
subtype of `Real`.  When the user elects to change such a field to
categorical each value is prefixed with an underscore and the
corresponding column `eltype` becomes `String`.

"""
function review_ordinals!(df::DataFrame)
    fields =  names(df)
    n_fields = length(df)
    head_size = min(size(df, 1), 4)
    i = 0
    println("RETURN -  yes")
    println("n      -  change to categorical")
    println("q      -  quit without further changes")
    println()
    while i != n_fields
        i = i + 1
        fld = fields[i]
        column = df[fld]
        t = eltype(column)
        if t <: Real
            print(fld, ": ")
            for j in 1:head_size
                print(column[j], ", ")
            end
            println("...")
            println()
            println("Ordinal type? ")
            response = chomp(readline())
            if response in ["n", "N"] 
                new = [string("_", column[i]) for i in eachindex(column)]
                df[fld] = new
            elseif response in ["q", "Q"]
                i = n_fields # we are ready to exit while loop
            end
        end
    end
    println()
    println("No more ordinals to review.")
end


"""

For data that is calculated periodically but reported more often (lots
of repeated entries). The function takes two 1-D DataArrays, the first
representing time, returns shortened versions of these two arrays so
that only times when the values of the second change are reported, and
only one version of each value is reported.

"""
function remove_steps(t, v)

    # Clean data
    t_clean = []
    v_clean = []
    n_clean = 0
    for i in 1:length(t)
        if !isna(v[i])
            n_clean = n_clean + 1
            push!(t_clean, t[i])
            push!(v_clean, v[i])
        end
    end

    # Find switching indices (first index for a given value)
    switches = Int[1]
    value = v_clean[1]

    for i in 2:n_clean
        if v_clean[i] != value
            push!(switches, i)
        end
        value = v_clean[i]
    end
    push!(switches, n_clean)
    
    # output the non-step versions inputs
    n_steps = length(switches)
    t_out = DataArray([t_clean[switches[i]] for i in 1:n_steps])
    y_out = DataArray([v_clean[switches[i]] for i in 1:n_steps])

    return t_out, y_out
end

""" 

This function takes two data arrays of the same length and returns two
new data arrays of the same length with the NAs is removed. 

"""
function drop_and_pair{S,T}(t::DataArray{S, 1}, v::DataArray{T, 1})

    if length(t) != length(v)
        throw(Base.error("DataArrays must have same length"))
    end
    
    n_patterns = length(t)
    x = S[]
    y = T[]
    for i in 1:n_patterns
        if !isna(v[i]) && !isna(t[i])
            push!(x, t[i])
            push!(y, v[i])
        end
    end
    return x, y
end

function reporting_period(v)
    intervals = Int[]
    value = v[1]
    latest = 1
    for i in 2:length(v)
        if !compare(v[i], value)
            push!(intervals, i - latest)
            latest = i
            value = v[i]
        end
    end
    push!(intervals, length(v) - latest + 1)
    return mean(intervals)
end

immutable StringToIntScheme
    n_levels::Int
    int_given_string::Dict{String, Int}
    string_given_int::Dict{Int, String}

    function StringToIntScheme(v::Vector{String})
        int_given_string = Dict{String, Int}()
        string_given_int = Dict{Int, String}()
        vals = sort!(collect(Set(v)))
        n_levels = length(vals)
        if length(vals) > 2^62
            throw(Base.error("Trying to construct a StringToIntScheme with a vector
                             having more than $(2^62) values."))
        end
        i = 0
        for c in vals
            int_given_string[c] = i
            string_given_int[i] = c
            i = i + 1
        end
        return new(n_levels, int_given_string, string_given_int)
    end
end

const VoidScheme=StringToIntScheme([""])

function show(stream::IO, s::StringToIntScheme)
    tail(n) = "..."*string(n)[end-3:end]
    if s == VoidScheme
        print(stream, "Scheme()")
    else
        print(stream, "$(s.n_levels)-level StringToIntScheme@$(tail(hash(s)))")
    end
end

function showall(stream::IO, s::StringToIntScheme)
    show(stream, s); println()
    for k in sort!(collect(keys(s.int_given_string)))
        println(stream, "$k => $(s.int_given_string[k])")
    end
end

transform(c::String, s::StringToIntScheme) = s.int_given_string[c]
transform(v::Vector{String}, s::StringToIntScheme) = Int[transform(c, s) for c in v]
inverse_transform(f::Int, s::StringToIntScheme) =
    s.string_given_int[f]
inverse_transform(v::Vector{Float64}, s::StringToIntScheme) =
    String[inverse_transform(f, s) for f in v]

function as_digits(j, base, n_digits)
    if j >= base^n_digits
        throw(Base.error("Error. Length of number exceeds number of digits alloted."))
    end
    digit = Dict{Int,Int}()
    for i in (n_digits - 1):-1:0 # for i = n_digits-1, n_digits-2, ..., 0
        power = base^i
        digit[i] = div(j, power)
        j = j - digit[i]*power
    end
    return [digit[i] for i in (n_digits - 1):-1:0]
end

"""

## `spawn_large_categorical!(df, feature)

This function is for replacing a categorical feature taking on a large
number of values with several categorical features taking on fewer
values. Specifically, the new features will take on no more than
`MAX_SMALL + 1` values, where `MAX_SMALL` is a constant defined in the
defining module, currently `52`. The function automatically determines
the number of new features needed.

### Arguments:

* `df::DataFrame` containing the feature
* `feature` index of the categorical feature in `df`

### Return values:

* None

Note that the old feature is not actually removed from the data
frame. (It might be used, for example, for later lookup purposes.)

"""
function spawn_large_categorical!(df::DataFrame, feature)

    # convert col values to strings for safety:
    col=map(string, collect(df[feature]))
    n_patterns = length(col)

    feature_name = names(df)[feature]

    # convert to integers:
    s=StringToIntScheme(col)
    icol = transform(col, s)

    n_vals = length(keys(s.int_given_string))
    n_cols = round(Int, floor(log(SMALL_MAX + 1, n_vals - 1)) + 1)

    a = Array(Int, (n_patterns, n_cols))
    for i in 1:n_patterns
        for j = 1:n_cols
            digits = as_digits(icol[i], SMALL_MAX + 1, n_cols)
            a[i,j] = digits[j]
        end
    end

    for j in 1:n_cols
        col_name = Symbol(string(feature_name, "__", j))
        df[col_name] = [string("_",a[i,j]) for i in 1:n_patterns]
    end
end

"""
## `rank_features(X, y, [bag]; penalty=0.5, n_trees=100, threshold=0.6*n_trees)`

Ranks the relative importance of the features (fields) in an input
`DataTable`, `X`, in predicting a corresponding target
`y::Vector{Float64}`. Some features may be considered "irrelevant" and
will not be included as keys in the returned dictionary. For a
stricter cutoff in this regard, lower the value of `threshold <
n_trees`. Importance is ranked on a scale of 0 to 1, 1 being the most
important. The lower the ranking the more important the feature. (0
just means least important, not "irrelevant").

The ranking is computed as follows: A number of decision trees are
constructed with the specified `penalty` as described under
`TreeRegressor` docs. For each tree the depth at which each feature is
introduced for the first time is noted. The depths are then averaged
over all trees. However, to be listed in the final ranking, the number
of trees using a feature in some decision must be greater than the
specified value of `threshold`. The ranking is then `1 - d/D` where
`d` is the average depth introduced and `D` the maximum depth.

"""
function rank_features(X,y,bag=[]; penalty=0.5, n_trees = 100, threshold=0)
    if threshold == 0
        threshold = round(Int, 0.6*n_trees)
    end
    if isempty(bag)
        bag = 1:length(y)
    end
            
    rgs = BaggedRegressor(X, y, bag; atom=TreeRegressor(penalty=penalty), n=n_trees) 
    
    # construct an iterable of indices of all features used for
    # some decision in some tree in the ensemble:
    kys = union([Set(keys(tree.importance_given_feature)) for tree in rgs.model]...)

    mean_importance_given_feature = Dict{Int,Float64}()
    cnt = Dict{Int,Int}()
    for j in kys
        mean_importance_given_feature[j] = 0.0
        cnt[j] = 0
    end
    
    for tree in rgs.model
        dic = tree.importance_given_feature
        for j in keys(dic)
            mean_importance_given_feature[j] += dic[j]
            cnt[j] = cnt[j] + 1
        end
    end

    ret = Dict{Int,Float64}()
    for j in kys
        if cnt[j] >= threshold
            ret[j] = mean_importance_given_feature[j]/cnt[j]
        end
    end

    # display user-friendly version of results (but return `ret` above)
    
    field_names = rgs.model[1].names
    x = Symbol[]
    y = Float64[]
    for j in reverse(keys_ordered_by_values(ret))
        push!(x, field_names[j])
        push!(y, ret[j])
    end
    UnicodePlots.show(UnicodePlots.barplot(x,y,title="Relative Feature Importance"))
    return ret
end

# end  # of module

end # of module
