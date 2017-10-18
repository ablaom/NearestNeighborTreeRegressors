__precompile__()
module TreeCollections

export Small, DataTable, row, head, countmap
export IntegerSet, push!, in, show, showall, Float64, round
export Node, is_stump, is_leaf, has_left, has_right, make_leftchild!, make_rightchild!
export unite!, child, issparse, ndims, getindex
export fit!, transform, FrameToTableScheme

using DataFrames
using ADBUtilities

import StatsBase.countmap
import DataFrames.head
import Base: getindex, show, showall, convert, length, size, start, next, done, issparse, ndims, copy
import Base: in, push!, Float64, round
import ScikitLearnBase: fit!, transform

typealias Small UInt8
typealias Big UInt64

# Currently cannot exceed Small(52) in the following constant declaration
const SMALL_MAX = Small(52) 
const BIG_MAX = Big(2^(Int(SMALL_MAX) + 1) - 1)

const TWO = Big(2)
const ZERO = Big(0)
const ONE = Small(1)

function mode(v)
    d = countmap(v)
    return keys_ordered_by_values(d)[end]
end

type Node{T}
    parent::Node{T}
    left::Node{T}
    right::Node{T}
    data::T
    depth::Int
    function Node(datum)
        node = new()
        node.parent = node
        node.left = node
        node.right = node
        node.data = datum
        node.depth = 0
        return node
    end
end

Node{T}(data::T)=Node{T}(data)

# Testing connectivity

is_stump(node) = node.parent == node
is_left(node) =  (node.parent != node) && (node.parent.left == node) 
is_right(node) = (node.parent != node) && (node.parent.right == node)
has_left(node) =  (node.left  != node)
has_right(node) = (node.right != node)
is_leaf(node) = node.left == node && node.right == node

# Connecting nodes

function make_leftchild!(child, parent)
    parent.left = child
    child.parent = parent
    child.depth = parent.depth + 1
end

function make_rightchild!(child, parent)
    parent.right = child
    child.parent = parent
    child.depth = parent.depth + 1
end

# Locating children

"""
# TreeCollections.child(parent, gender)

Returns the left child of `parent` of a `Node` object if `gender` is 1
and right child if `gender is 2. If `gender` is `0` the routine throws
an error if the left and right children are different and otherwise
returns their common value.  For all other values of gender an error
is thrown.

## Return type:
`Node`

"""
function child(parent, gender)
    if gender == 1
        return parent.left
    elseif gender == 2
        return parent.right
    elseif gender == 0
        if parent.left != parent.right
            throw(Base.error("Left and right children different."))
        else
            return parent.left
        end
    end
    throw(Base.error("Only genders 0, 1 or 2 allowed."))
end

"""
# BinaryNodes.unite!(child, parent, gender)

Makes `child` the `left` or `right` child of a `Node` object `parent`
in case `gender` is `1` or `2` respectively; and makes `parent` the
parent of `child`. For any other values of `gender` the routine makes
`child` simultaneously the left and right child of `parent`, and
`parent` the parent of `child`.

## Return value:
None.

"""
function unite!(child, parent, gender)
    if gender == 1
        make_leftchild!(child, parent)
    elseif gender == 2
        make_rightchild!(child, parent)
    else
        make_leftchild!(child, parent)
        make_rightchild!(child, parent)
    end
end
    

# Display functionality

function spaces(n)
    s = ""
    for i in 1:n
        s = string(s, " ")
    end
    return s
end

function gender(node)
    if is_stump(node)
        return 'A' # androgenous
    elseif is_left(node)
        return 'L'
    else
        return 'R'
    end
end

tail(n) = "..."*string(n)[end-3:end]

function show(stream::IO, node::Node)
    print(stream, "Node{$(typeof(node).parameters[1])}@$(tail(hash(node)))")
end

function showall(stream::IO, node::Node)
    gap = spaces(node.depth + 1)
    println(stream, string(gender(node), gap, node.data))
    if has_left(node)
        showall(stream, node.left)
    end
    if has_right(node)
        showall(stream, node.right)
    end
    return
end

showall(node::Node)=showall(STDOUT, node)

# for testing purposes:
# Node(data) = Node{typeof(data)}(data)

function Node(data, parent::Node)
    child = Node(data)
    make_leftchild!(child, parent)
    return child
end

function Node(parent::Node, data)
    child = Node(data)
    make_rightchild!(child, parent)
    return child
end

"""
# `TreeCollections.IntegerSet`

A type of collection for storing subsets of {0, 1, 2, ... 52}. Every
such subset can be stored as an Float64 object. To convert an
IntegerSet object `s` to a floating point number, use Float64(s). To
recover the original object from a float `f`, use `round(IntegerSet,
f)`.

To instantiate an empty collection use, `IntegerSet()`. To add an
element `i::Integer` use `push!(s, i)` which is quickest if `i` is
type `UInt8`. Membership is tested as usual with `in`. One can also
instantiate an object with multiple elements as in the following example:

    julia> 15 in IntegerSet([1, 24, 16])
    false

"""
type IntegerSet
    coded::Big
end

function IntegerSet{T<:Integer}(v::Vector{T})
    s = IntegerSet()
    for k in v
        push!(s, k) # push! defined below
    end
    return s
end

IntegerSet() = IntegerSet(ZERO)

function in(k::Small, s::IntegerSet)
    large = TWO^k
    return large & s.coded == large
end

function in(k::Integer, s::IntegerSet)
    large = TWO^Small(k)
    return large & s.coded == large
end

function push!(s::IntegerSet, k::Small)
    if k > SMALL_MAX
        throw(Base.error("Cannot push! an integer larger 
           than $(Int(SMALL_MAX)) into an IntegerSet object."))
    end
    if !(k in s)
        s.coded = s.coded | TWO^k
    end
    return s
end

function push!(s::IntegerSet, k::Integer)
    if k > SMALL_MAX
        throw(Base.error("Cannot push! an integer larger 
           than $(Int(SMALL_MAX)) into an IntegerSet object."))
    end
    push!(s, Small(k))
end

function show(stream::IO, s::IntegerSet)
    for i in 0:62
        if i in s
            print(stream, "$i, ")
        end
    end
end

Float64(s::IntegerSet) = Float64(s.coded)

function round(T::Type{IntegerSet}, f::Float64)
    if f < 0 || f > BIG_MAX
        throw(Base.error("Float64 numbers outside the 
           range [0,BIG_MAX] cannot be rounded to IntegerSet values"))
    end
    return IntegerSet(round(Big, f))
end

immutable StringToSmallFloatScheme
    small_given_string::Dict{String, Small}
    string_given_small::Dict{Small, String}
    mode::String
    
    function StringToSmallFloatScheme(v::Vector{String}; overide=false)
        small_given_string = Dict{String, Small}()
        string_given_small = Dict{Small, String}()
        vals = sort(collect(Set(v)))
        if length(vals) > SMALL_MAX + 1 && !overide
            throw(Base.error("Trying to construct a StringToSmallFloatScheme with a vector
                             having more than $(SMALL_MAX + 1) values."))
        end
        i = Small(0)
        for c in vals
            small_given_string[c] = i
            string_given_small[i] = c
            i = i + ONE
        end
        return new(small_given_string, string_given_small, mode(v))
    end
end

const VoidScheme=StringToSmallFloatScheme([""])

function show(stream::IO, s::StringToSmallFloatScheme)
    if s == VoidScheme
        print(stream, "VoidScheme")
    else
        print(stream, "StringToSmallFloatScheme@$(tail(hash(s)))")
    end
end

function showall(stream::IO, s::StringToSmallFloatScheme)
    if s == VoidScheme
        show(stream, s)
    else
        show(stream, s)
        str = "\n"
        for k in sort(collect(keys(s.small_given_string)))
            str = string(str, k, " => ", Float64(s.small_given_string[k]), "\n")
        end
        print(stream, str)
    end
end


"""
# `function transform(s::StringToSmallFloatScheme, c::String)`

If the string input `c` does not appear in the dictionary of scheme `s` then `s.mode` is returned.

"""
function transform(s::StringToSmallFloatScheme, c::String)
    small = s.mode
    try
        small = s.small_given_string[c]
    catch
        small = s.small_given_string[s.mode]
    end
    return Float64(small)
end

transform(s::StringToSmallFloatScheme, v::Vector{String}) = Float64[transform(s, c) for c in v]
inverse_transform(s::StringToSmallFloatScheme, f::Float64) =
    s.string_given_small[round(Small, f)]
inverse_transform(s::StringToSmallFloatScheme, v::Vector{Float64}) =
    String[inverse_transform(s, f) for f in v]

type FrameToTableScheme
    
    # post-fit parameters
    schemes::Vector{StringToSmallFloatScheme}
    is_ordinal::Vector{Bool}
    
    FrameToTableScheme() = new()
end

function copy(s::FrameToTableScheme)
    t = FrameToTableScheme()
    t.schemes = copy(s.schemes)
    t.is_ordinal = copy(s.is_ordinal)
    return t
end

function FrameToTableScheme(X::DataFrame)
    s = FrameToTableScheme()
    fit!(s, X)
    return s
end

function show(stream::IO, s::FrameToTableScheme)
    print(stream, "FrameToTableScheme@$(tail(hash(s)))")
end

function fit!(scheme::FrameToTableScheme, X::DataFrame)
    ncols = size(X, 2)
    schemes = Array(StringToSmallFloatScheme, ncols)
    is_ordinal = Array(Bool, ncols)
    for j in 1:ncols
        column_type = eltype(X[j])
        if column_type <: Real
            is_ordinal[j] = true
            schemes[j] = VoidScheme
        elseif column_type in [String, Char]
            is_ordinal[j] = false
            col = sort([string(s) for s in X[j]])
            schemes[j] = StringToSmallFloatScheme(col)
        else
            error("I have encountered a DataFrame column of inadmissable type for DataTable constuction.")
        end
    end
    scheme.schemes = schemes
    scheme.is_ordinal = is_ordinal
    return scheme
end

# For ScikitLearn API compatiblity:
fit!(scheme::FrameToTableScheme, X::DataFrame, y) = fit!(scheme, X)    
               
"""
# `TreeCollections.DataTable`

A `TreeCollections.DataTable` object is a immutable data structure
that presents externally as a DataFrame whose columns are a mixture of
categorical and ordinal type. Internally, however it stores this data
as an ordinary `Float64` array that can then be passed to
high-performance tree-based machine learning algorithms. The number of
values taken by a categorical feature in this data structure is
intentionally limited to 53.  This is because at nodes of a decision
tree or tree regressor the criterion for a binary split based on such
a feature (specifically, a subset of `1:53`) can be encoded in a
single `Float64` number, just as the threshold for ordinal
features. The nodes in such a tree can therefore be of homogeneous
type.

The `Float64` array encoding the data is stored in the field
`raw::Array{Float64, 2}`. The other fields are `names`, `nrows` and
`ncols` (which are self-explanatory) and a field `scheme` (of type
`FrameToTableScheme`) which stores information on which columns are
categorical and how to transform back and forth between a categorical
feature and its equivalent floating point representation.

`Datatable` objects can be constructed in various ways shown in the
following examples:

    julia> df = DataFrame.readtable("train.csv")
    julia> names(df)
    [:KM, :n_doors, :year]
    julia> df_test = DataFrame.readtable("test.csv")
   
    julia> dt = DataTable(df) 
    julia> dt.names == names(df)
    true
    julia> s = dt.scheme
    julia> dt_test = DataTable(df_test, s) # a second `DataTable` object with same encoding
    julia> dt_test = transform(s, df_test) # the same thing using ScikitLearn syntax

    julia> s = FrameToTableScheme()
    julia> fit!(s, df) # with df as return value
    julia> dt = transform(s, df)
    julia> dt.scheme == s
    true

    julia> u, v, w = df[1], df[2], df[3]
    julia> dt_with_default_naming = DataTable(u, v, w)
    julia> dt_with_default_names.names
    [:x1, :x2, :x3]

    julia> dt = DataTable(columns = [u,v,w], names = [:X,:Y,:Z]) # with names prescribed

Note that these constructors automatically convert all numerical
features (eltype a subtype of `Real`) to `Float64` type (to
be treated as ordinals) and all features of `Char` or `String` type are
identified as categorical.

One can iterate over the columns of the `DataTable` and some basic
indexing for retrieving (but not setting) elements is
implemented. There is a `row` method.

## Sample use of implemented methods:

    length(dt), size(dt), convert(DataFrame, dt), head(dt), countmap(dt[1])


"""
immutable DataTable
    raw::Array{Float64,2}
    names::Vector{Symbol}
    nrows::Int
    ncols::Int
    scheme::FrameToTableScheme
end

copy(X::DataTable) = DataTable(copy(X.raw), copy(X.names), X.nrows, X.ncols, copy(X.scheme))

function DataTable(df::DataFrame, scheme::FrameToTableScheme)

    nrows, ncols = size(df)
    raw    = Array(Float64, (nrows, ncols))
    
    if length(scheme.schemes) != ncols
        throw(BoundsError)
    end
    for j in 1:ncols
        col = [x for x in df[j]] # vector with type demoted as far as possible
        if scheme.is_ordinal[j]
            if !(eltype(col) <: Real)
                throw(TypeError)
            end
            for i in 1:nrows
                raw[i,j] = Float64(df[i,j])
            end
        else
            if !(eltype(col) in [String, Char])
                throw(BoundsError)
            end
            for i in 1:nrows
                raw[i,j] = transform(scheme.schemes[j], string(df[i,j]))
            end
        end
    end

    return DataTable(raw, names(df), nrows, ncols, scheme)

end

# For ScikitLearn API compatiblity:
transform(scheme::FrameToTableScheme, df::DataFrame) = DataTable(df, scheme)

DataTable(df::DataFrame) = DataTable(df, FrameToTableScheme(df))

function DataTable(;columns::Vector=Any[], names::Vector{Symbol}=Symbol[])
    ncols = length(columns)
    if ncols == 0
        throw(Base.error("Error constructing DataTable object. 
                         It must have at least one column."))
    end
    if length(names) != ncols
        throw(Base.error("You must supply one column name per column."))
    end
    
    nrows = length(columns[1])
    if sum([length(v)!=nrows for v in columns]) != 0
        throw(Base.error("Error constructing DataTable object. 
                         All columns must have same length."))
    end
    df = DataFrame(columns, names)
    return DataTable(df)
end
                                
function DataTable(column_tuple...)
    n_cols = length(column_tuple)
    cols = collect(column_tuple)
    colnames = Vector{Symbol}(n_cols)
    for i in 1:n_cols
        colnames[i]=Symbol(string("x",i))
    end
    return DataTable(columns=cols, names=colnames)
end

function columns(dt::DataTable)
    cols = Any[]
    for j in 1:dt.ncols
        rawcol = Float64[dt.raw[i,j] for i in 1:dt.nrows]
        if dt.scheme.is_ordinal[j]
            push!(cols, rawcol)
        else
            push!(cols, inverse_transform(dt.scheme.schemes[j], rawcol))
        end
    end
    return cols
end
                  
function convert(T::Type{DataFrame}, dt::DataTable)
    return DataFrame(columns(dt), dt.names)
end

function show(stream::IO, dt::DataTable; nrows=0)
    if nrows == 0
        nrows = dt.nrows
    end
    ncols = dt.ncols
    types = Array(String, ncols)
    for j in 1:ncols
        if dt.scheme.is_ordinal[j]
            types[j] = "ord"
        else
            types[j] = "cat"
        end
    end
    header = [Symbol(string(dt.names[j], " ($j,$(types[j]))")) for j in 1:ncols]
    println("(Displaying DataTable as DataFrame)")
    show(stream, DataFrame(columns(dt), header)[1:nrows,:])
    println()
end

head(dt::DataTable) =  show(STDOUT, dt; nrows=min(4, dt.nrows))

getindex(dt::DataTable, i::Int, j::Int) = dt.raw[i,j]
getindex(dt::DataTable, j::Int) = dt.raw[:,j]

function getindex(dt::DataTable, col_name::Symbol)
    j = 0
    for k in eachindex(dt.names)
        if dt.names[k] == col_name
            j = k
        end
    end
    if j == 0
        throw(DomainError)
    end
    return dt[j]
end

function getindex(dt::DataTable, bs::Vector{Symbol})
    index_given_name = Dict{Symbol,Int}()
    for j in eachindex(dt.names)
        index_given_name[dt.names[j]] = j
    end
    b = [index_given_name[sym] for sym in bs]
    raw = dt.raw[:,b] 
    col_names = dt.names[b]
    nrows = dt.nrows
    ncols = length(b)
    scheme = FrameToTableScheme()
    scheme.schemes = dt.scheme.schemes[b]
    scheme.is_ordinal = dt.scheme.is_ordinal[b]
    return DataTable(raw, col_names, nrows, ncols, scheme)    
end

# function getindex(dt::DataTable, c::Colon, j::Int)
#     raw = dt.raw[:,j:j] # j:j forces two-dimensionality of the array
#     col_names = dt.names[j:j]
#     nrows = dt.nrows
#     ncols = 1
#     scheme = FrameToTableScheme()
#     scheme.schemes = dt.scheme.schemes[j:j]
#     scheme.is_ordinal = dt.scheme.is_ordinal[j:j]
#     return DataTable(raw, col_names, nrows, ncols, scheme)    
# end

getindex(dt::DataTable, c::Colon, j::Int) = dt.raw[:,j]
getindex(dt::DataTable, i::Int, c::Colon) = dt.raw[i,:]
row(dt::DataTable, i::Int) = dt.raw[i,:]

function getindex(dt::DataTable, a::Vector{Int}, c::Colon)
    raw = dt.raw[a,:]
    col_names = dt.names
    nrows = length(a)
    ncols = dt.ncols
    scheme = dt.scheme
    return DataTable(raw, col_names, nrows, ncols, scheme)    
end

getindex(dt::DataTable, a::UnitRange{Int}, c::Colon) = dt[collect(a),:]

function getindex(dt::DataTable, c::Colon, b::Vector{Int})
    raw = dt.raw[:,b]
    col_names = dt.names[b]
    nrows = dt.nrows
    ncols = length(b)
    scheme = FrameToTableScheme()
    scheme.schemes = dt.scheme.schemes[b]
    scheme.is_ordinal = dt.scheme.is_ordinal[b]
    return DataTable(raw, col_names, nrows, ncols, scheme)
end
getindex(dt::DataTable, c::Colon, b::UnitRange{Int}) = dt[:,collect(b)]

function getindex(dt::DataTable, a::Vector{Int}, b::Vector{Int})
    raw = dt.raw[a,b]
    col_names = dt.names[b]
    nrows = length(a)
    ncols = length(b)
    scheme = FrameToTableScheme()
    scheme.schemes = dt.scheme.schemes[b]
    scheme.is_ordinal = dt.scheme.is_ordinal[b]
    return DataTable(raw, col_names, nrows, ncols, scheme)
end
getindex(dt::DataTable, a::UnitRange{Int}, b::UnitRange{Int}) = dt[collect(a),collect(b)]
getindex(dt::DataTable, a::UnitRange{Int}, b::Vector{Int}) = dt[collect(a),b]
getindex(dt::DataTable, a::Vector{Int}, b::UnitRange{Int}) = dt[a,collect(b)]

length(dt::DataTable) = dt.ncols
issparse(dt::DataTable) = false
ndims(dt::DataTable) = 2
size(dt::DataTable) = (dt.nrows, dt.ncols)

function size(dt::DataTable, i::Int)
    if  i == 1
        return dt.nrows
    elseif i == 2
        return dt.ncols
    else
        throw(BoundsError)
    end
end

# Iteration methods:
start(dt::DataTable) = 1
next(dt::DataTable, i) = (dt[i], i + 1)
done(dt::DataTable, i) = (i > dt.ncols)              

end # of module
