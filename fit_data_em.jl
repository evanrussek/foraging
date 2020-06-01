
parallel = true # Run on multiple CPUs. If you are having trouble, set parallel = false: easier to debug
full = false    # Maintain full covariance matrix (vs a diagional one) a the group level
emtol = 1e-3    # stopping condition (relative change) for EM

using Distributed
if (parallel)
	# only run this once
	addprocs()
end

# this loads the packages needed -- the @everywhere makes sure they
# available on all CPUs

@everywhere using DataFrames
@everywhere using SharedArrays
@everywhere using ForwardDiff
@everywhere using Optim
@everywhere using LinearAlgebra       # for tr, diagonal
@everywhere using StatsFuns           # logsumexp
@everywhere using SpecialFunctions    # for erf
@everywhere using Statistics          # for mean
@everywhere using Distributions
@everywhere using GLM

using CSV
using DataFramesMeta
using CategoricalArrays
using Gadfly
# using Statistics
# using Distributions
#using SpecialFunctions
using StatsFuns
#using Optim
using ForwardDiff
using Cairo
using Fontconfig

## prepare the data...

@everywhere cd("/Users/evanrussek/foraging/")

# basic real data clearning, etc, functions...
@everywhere include("/Users/evanrussek/lockin_data/lockin_analysis/forage_data_funs.jl")
@everywhere include("sim_lag_functions.jl")
@everywhere include("sim_learn_funcs.jl")
@everywhere include("simulation_functions.jl")
@everywhere include("lik_funs.jl")

# read in the data...
data = CSV.read("/Users/evanrussek/forage_jsp/analysis/data/run5_data.csv");
# check the travel keys are correctly labeled...
travel_keys = unique(data.travel_key)
travel_key_easy = travel_keys[1]
travel_key_hard = travel_keys[2]
travel_keys_he = [travel_key_hard travel_key_easy];
cdata, n_subj = clean_group_data(data,travel_keys_he);
cdata[!,:sub] = cdata[!,:s_num];
pdata = by(cdata, :sub, df -> prep_subj_data(df));

param_names = ["travel_cost_easy", "travel_cost_hard", "harvest_cost", "lr_R_hat_pre", "R_hat_start", "choice_beta", "lag_beta",
"harvest_bias", "lag_rr_sen"];

n_param = length(param_names)
n_subj = length(unique(pdata[!,:sub]))
min_nll = zeros(n_subj);
p_hat = zeros(n_subj, n_param);

# figure out what went wrong here...
bad_subjs = [39];
pdata[!,:remove] .= false
for bs in bad_subjs
    pdata[pdata[!,:sub] .== bs,:remove] .= true
end

pdata_clean = @where(pdata, :remove .== false);
pdata_clean.sub = groupindices(groupby(pdata_clean,:subjectID));

# should be ready...
@everywhere em_dir =  "/Users/evanrussek/em";
@everywhere include("$em_dir/em.jl");
@everywhere include("$em_dir/common.jl");
@everywhere include("$em_dir/likfuns.jl")

# ok...
NS = length(unique(unique(pdata_clean.sub)))
subs = 1:NS;
# group level design matrix
X = ones(NS);


# starting points (change this so it runs...)
# how many parameters ar there?
NP = length(param_names)

betas = [1. 1. 1. -8. 1. 2. 1. 1. 1.];
sigma = [2., 2., 2., 2., 4., 5., 5., 5., 2.];

param_names

# fit em...
(betas,sigma,x,l,h) = em(pdata_clean,subs,X,betas,sigma,forage_lik_rs; emtol=emtol, parallel=parallel, full=full);


# 8 parameters now... might be overparamaterized b/c there's choice/lag betas and also sensitivity and costs



# simulate this data
p_hat_dict = Dict()
for i in 1:length(param_names)
	p_hat_dict[param_names[i]] = betas[i];
end

p_hat_dict



sim_data = DataFrame();
## simulate...
for r = 1:10
	print(r)
	s_data = sim_forage_rs(p_hat_dict);
	s_data[!,:sub] .= r;
	s_data[!,:subjectID] .= r;
	s_data[!,:upper_lag_thresh] .= Inf;
	s_data[!,:lower_lag_thresh] .= -Inf;
	global sim_data = [sim_data;s_data];
end
sim_data[!,:remove] .= false;

@where(sim_data, :phase .== "HARVEST")

sim_data

make_group_lag_plot(pdata_clean)

make_group_lag_plot(sim_data)

make_group_exit_plot(pdata_clean)
make_group_exit_plot(sim_data)

make_group_lag_plot(pdata_clean)




ll1 = lml(x,l,h)

n_data_points = size(pdata_clean,1)

ibic(x,l,h,betas,sigma,n_data_points)

Diag(sigma)

sigma

### great.
