append!(LOAD_PATH,["./code/", "./"])
addprocs(3)
        
import DataFrames: readtable, writetable, head, size
using Datasets, Regressors, Validation, TreeCollections
using Plots, ADBPlots
pyplot()

######################
# Bicycle Share data #
######################

df = load("bike_r")

const X = DataTable(df[2:end])
const y = convert(Vector{Float64}, df[:cnt_log])

train, valid = split_bag(1:size(df,1), 70)

base = rms_error(ConstantRegressor(X,y),X,y)
bike_reg = cv_errors(TreeRegressor(regularization=0.82), X, y, n_folds=12, parallel=true, verbose=false)/base
bike_prune = cv_errors(TreeRegressor(min_patterns_split=47), X, y, n_folds=12, parallel=true, verbose=false)/base

bike_reg_bar = string(mean(bike_reg), "±", 2*std(bike_reg))
bike_prune_bar = string(mean(bike_prune), "±", 2*std(bike_prune))
bike_playoff = compete(bike_prune, bike_reg)

bootstrap_histogram(bike_reg, label="nearest neighbor regularization")
bootstrap_histogram!(bike_prune, label="regularization by pruning")
plot!(xlab="estimated RMSL error for target predictions")
plot!(title="\"bike_r\" dataset")
savefig("assets/bike_r.png")


#######################
# Toyota Corolla data #
#######################

df = load("toyota_r")

const X = DataTable(df[2:end])
const y = log.(convert(Vector{Float64}, df[:Price]))

train, valid = split_bag(1:size(df,1), 70)

base = rms_error(ConstantRegressor(X,y),X,y)
toyota_reg = cv_errors(TreeRegressor(regularization=0.1), X, y, n_folds=12, parallel=true, verbose=false)/base
toyota_prune = cv_errors(TreeRegressor(min_patterns_split=2), X, y, n_folds=12, parallel=true, verbose=false)/base

toyota_reg_bar = string(mean(toyota_reg), "±", 2*std(toyota_reg))
toyota_prune_bar = string(mean(toyota_prune), "±", 2*std(toyota_prune))
toyota_playoff = compete(toyota_prune, toyota_reg)

bootstrap_histogram(toyota_reg, label="nearest neighbor regularization")
bootstrap_histogram!(toyota_prune, label="regularization by pruning")
plot!(xlab="estimated RMSL error for target predictions")
plot!(title="\"toyota_r\" dataset")
savefig("assets/toyota_r.png")

#########################
# Ames House Price data #
#########################

df = load("ames_12r")

const X = DataTable(df[2:end])
const y = convert(Vector{Float64}, df[:target])

train, valid = split_bag(1:size(df,1), 70)

base = cv_error(ConstantRegressor(),X,y,n_folds=12)

ames_reg = cv_errors(TreeRegressor(regularization=0.89), X, y, n_folds=12, parallel=true, verbose=false)/base
ames_prune = cv_errors(TreeRegressor(min_patterns_split=33), X, y, n_folds=12, parallel=true, verbose=false)/base

ames_reg_bar = string(mean(ames_reg), "±", 2*std(ames_reg))
ames_prune_bar = string(mean(ames_prune), "±", 2*std(ames_prune))
ames_playoff = compete(ames_prune, ames_reg)

bootstrap_histogram(ames_reg, label="nearest neighbor regularization")
bootstrap_histogram!(ames_prune, label="regularization by pruning")
plot!(xlab="estimated RMSL error for Price predictions")
plot!(title="\"ames_r\" dataset")
savefig("assets/ames_r.png")

################
# Abalone data #
################

df = load("abalone_r")

const X = DataTable(df[2:end])
const y = log(convert(Vector{Float64}, df[:rings]))

train, valid = split_bag(1:size(df,1), 70)

base = cv_error(ConstantRegressor(),X,y,n_folds=12)
abalone_reg = cv_errors(TreeRegressor(regularization=0.933), X, y, n_folds=12, parallel=true, verbose=false)/base
abalone_prune = cv_errors(TreeRegressor(min_patterns_split=135), X, y, n_folds=12, parallel=true, verbose=false)/base

abalone_reg_bar = string(mean(abalone_reg), "±", 2*std(abalone_reg))
abalone_prune_bar = string(mean(abalone_prune), "±", 2*std(abalone_prune))
abalone_playoff = compete(abalone_prune, abalone_reg)

bootstrap_histogram(abalone_reg, label="nearest neighbor regularization")
bootstrap_histogram!(abalone_prune, label="regularization by pruning")
plot!(xlab="estimated RMSL error for target predictions")
plot!(title="\"abalone_r\" dataset")
savefig("assets/abalone_r.png")

####################
# Power Plant data #
####################

df = load("power_r")

const X = DataTable(df[2:end])
const y = convert(Vector{Float64}, df[:PE])

train, valid = split_bag(1:size(df,1), 70)

base = cv_error(ConstantRegressor(),X,y,n_folds=12)
power_reg = cv_errors(TreeRegressor(regularization=0.78), X, y, n_folds=12, parallel=true, verbose=false)/base
power_prune = cv_errors(TreeRegressor(min_patterns_split=47), X, y, n_folds=12, parallel=true, verbose=false)/base

power_reg_bar = string(mean(power_reg), "±", 2*std(power_reg))
power_prune_bar = string(mean(power_prune), "±", 2*std(power_prune))
power_playoff = compete(power_prune, power_reg)

bootstrap_histogram(power_reg, label="nearest neighbor regularization")
bootstrap_histogram!(power_prune, label="regularization by pruning")
plot!(xlab="estimated RMS error for target predictions")
plot!(title="\"power_r\" dataset")
savefig("assets/power_r.png")

#################
# Concrete data #
#################

df = load("concrete_r")

const X = DataTable(df[2:end])
const y = convert(Vector{Float64}, df[:strength])

train, valid = split_bag(1:size(df,1), 70)

base = cv_error(ConstantRegressor(),X,y,n_folds=12)
concrete_reg = cv_errors(TreeRegressor(regularization=0.615), X, y, n_folds=12, parallel=true, verbose=false)/base
concrete_prune = cv_errors(TreeRegressor(min_patterns_split=7), X, y, n_folds=12, parallel=true, verbose=false)/base

concrete_reg_bar = string(mean(concrete_reg), "±", 2*std(concrete_reg))
concrete_prune_bar = string(mean(concrete_prune), "±", 2*std(concrete_prune))
concrete_playoff = compete(concrete_prune, concrete_reg)

bootstrap_histogram(concrete_reg, label="nearest neighbor regularization")
bootstrap_histogram!(concrete_prune, label="regularization by pruning")
plot!(xlab="estimated RMS error for target predictions")
plot!(title="\"concrete_r\" dataset")
savefig("assets/concrete_r.png")

# cross-validation tune:
# u, v = @getfor rr linspace(0,0.99,200) rms_error(TreeRegressor(regularization=rr,X,y,train),X,y,valid)/base
# u,v = @getfor rr linspace(0.5,0.7,41) cv_error(TreeRegressor(regularization=rr), X, y, parallel=true, verbose=false, n_folds=12)/base
# u, v = @getfor rr 2:200 rms_error(TreeRegressor(min_patterns_split=rr,X,y,train),X,y,valid)/base
# u,v = @getfor rr 2:20 cv_error(TreeRegressor(min_patterns_split=rr), X, y, parallel=true, verbose=false, n_folds=12)/base




