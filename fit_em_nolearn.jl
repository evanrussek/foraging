
parallel = false # Run on multiple CPUs. If you are having trouble, set parallel = false: easier to debug
full = false    # Maintain full covariance matrix (vs a diagional one) a the group level
emtol = 1e-3    # stopping condition (relative change) for EM

using Distributed
if (parallel)
	# only run this once
	addprocs()
end

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

 cd("/Users/evanrussek/foraging/")

# basic real data clearning, etc, functions...
include("/Users/evanrussek/lockin_data/lockin_analysis/forage_data_funs.jl")
include("sim_lag_functions.jl")
include("sim_learn_funcs.jl")
include("simulation_functions.jl")
include("lik_funs.jl")


###
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

### can we fit a model that has reduced foraging effect...
group_exit = make_group_exit_plot(pdata_clean;  title = "Data", bounds = [15, 60])
group_lag = make_group_lag_plot(pdata_clean; title = "Data")


param_names = ["choice_r_hard_low_beta", "choice_r_easy_beta", "choice_r_reward_beta",
                "lag_r_hard_low_beta", "lag_r_easy_beta", "lag_r_reward_beta",
                "harvest_cost", "travel_cost_easy", "travel_cost_hard",
                "choice_beta"];
n_params = length(param_names);

####

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

betas = [.2 .2 .2 .2 .2 .2 .2 .2 .2 .2];
sigma = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.];

(betas,sigma,x,l,h) = em(pdata_clean,subs,X,betas,sigma,forage_lik_nolearn; emtol=emtol, parallel=parallel, full=full);

betas

p_hat_dict = Dict();
for n in 1:n_params
    p_hat_dict[param_names[n]] = betas[n]
end

p_hat_dict["lag_beta"] = 1;

group_rec_data = DataFrame();
for i in 1:10
	print(i)
	rec_data = sim_forage_no_learn(p_hat_dict)
	rec_data[!,:lower_lag_thresh] .= -Inf;
	rec_data[!,:upper_lag_thresh] .= Inf;
	rec_data[!,:subjectID] .= i;
	rec_data[!,:sub] .= i;
	global group_rec_data = [group_rec_data; rec_data];
end
group_rec_data[!,:remove] .= false;
rec_mn_exit = make_group_exit_plot(group_rec_data; title = "Recovered", bounds = [15, 60])
rec_mn_lag = make_group_lag_plot(group_rec_data; title = "Recovered")

####  #set the bounds on these to 20 and 60


draw(PNG("plots/nolearn_nob_recovery/exit_data_model.png", 6inch, 3inch), hstack([group_exit, rec_mn_exit]))
draw(PNG("plots/nolearn_nob_recovery/lag_data_model.png", 4inch, 6inch), vstack([group_lag, rec_mn_lag]))

cor(x)

cor_x = cor(x);
corr_dict = Dict();
for i = 1:n_params
	corr_dict[param_names[i]] = cor_x[:,i];
end
corr_dict["pname"] = param_names;
corr_df = DataFrame(corr_dict)
corr_df = sort(corr_df, :pname)
permutecols!(corr_df, [:pname])
CSV.write("plots/nolearn_nob_recovery/model_corr.csv", corr_df)
