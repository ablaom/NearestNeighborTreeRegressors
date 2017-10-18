__precompile__()
module Load

export Data, @load_features, @load

using DataFrames, TreeCollections, ADBUtilities
import Distributions
import Base: show

const FILE = "2.cleaned/train.csv"
const RANDOMFILE = "2.cleaned/train_randomized.csv"
const TARGET = :target

function split_bag(bag; train=70.0, validate=20.0)
    bag = collect(bag)
    n_patterns = length(bag)
    n_full = round(Int, (train + validate)*n_patterns/100)
    n_train = round(Int, floor(train*n_patterns/100)) 
    n_validate = n_full - n_train
    b_train = bag[1:n_train]
    b_validate = bag[(n_train + 1):n_full]
#   b_full = copy(b_train)
#   b_full = append!(b_full, b_validate)
    b_test = bag[(n_full + 1):n_patterns]

    return b_train, b_validate, b_test
end

immutable Data
    file::String
    startrow::Int
    nrows::Int
    X::DataTable
    y::Vector{Float64}
    train::Vector{Int}
    validate::Vector{Int}
    test::Vector{Int}
end

function Data(;nrows=0, startrow=1, randomized=true, with_new_randomization=false,
              features=[], file="")

    df = DataFrame() # needed because try-catch block below has soft local scope
    
    # Get the right data from the right file, randomizing if necessary:
    if file != ""
        df = readtable(file)
        n_patterns = size(df)[1]
    elseif with_new_randomization == true
        dh = readtable(FILE)
        n_patterns, n_fields = size(dh)
        bag_randomized = Distributions.sample(1:n_patterns, n_patterns, replace=false)
        df = dh[bag_randomized, :] 
        writetable(RANDOMFILE, df)
        file = RANDOMFILE
        dh = DataFrame() # free up memory
    elseif randomized == false
        df = readtable(FILE)
        n_patterns = size(df)[1]
        file = FILE
    else
        try
            df = readtable(RANDOMFILE)
            file = RANDOMFILE
        catch
            println("By default randomized data is used. Currently no randomized version ",
                   "of $FILE exists so unrandomized data is being loaded. To load ",
                   "randomized data, use `Data(with_new_randomization=true)`.")
            df = readtable(FILE)
            file = FILE
            writetable(RANDOMFILE, df)
        end
        n_patterns = size(df)[1]
    end

    # Determine number of rows if none specified:
    if nrows == 0
       nrows = n_patterns - startrow + 1
    end

    # Use all features except target if none specified:
    if features == []
        features = names(df)
        filter!(x -> x!=TARGET, features)
    end

    # build input and output data structures:
    X = DataTable(df[startrow:(startrow + nrows - 1), features])

    y = zeros(Float64, nrows)
    try
        y = collect(df[startrow:(startrow + nrows - 1), TARGET])
    catch
        Base.warn("No target field appears to exist in the specified or default files.",
                  " Creating a target vector filled with zeros.")
    end
    
    # build bags:
    train, validate, test = split_bag(1:nrows)

    return Data(file, startrow, nrows, X, y, train, validate, test)
end

function show(stream::IO, d::Data)
    println(stream, "file = $(d.file)")
    println(stream, "startrow = $(d.startrow)")
    println(stream, "nrows = $(d.nrows)")
    println(stream, "y = $(length(d.y))-length vector of $TARGET values")
    println(stream, "X = $(d.nrows) X $(d.X.ncols) DataTable with fields:")
    nms = d.X.names
    typs = [d.X.scheme.is_ordinal[j] ? "Ordinal" : "Categorical" for j in eachindex(nms)]
    showall(stream, DataFrame([nms, typs],[:feature, :type]))
end

macro load()
    quote
        using TreeCollections, Validation, Regressors
        import DataFrames.readtable
        global features = convert(Vector{Symbol}, collect(readtable(
                  "3.important_features/important_features.csv")[:field]))
        d = Data(features=Main.features)
        global const y = d.y
        global const X = d.X
        global const train = d.train
        global const valid = d.validate
        global const test = d.test;
        show(d)
    end
end

end # of module    

                       
