using CSV
using DataFramesMeta
using CategoricalArrays
using Gadfly
using Statistics
using Distributions
using SpecialFunctions
using StatsFuns
using Optim
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


param_dict = Dict();

# spread parameters# lag beta...

# learning rate parameters
param_dict["lr_R_hat_pre_lag"] = -8;
param_dict["lr_R_hat_pre_choice"]= -8; # make these the same?

param_dict["harvest_cost_lag"] = .000005;
param_dict["travel_cost_easy_lag"] = .0000015;
param_dict["travel_cost_hard_lag"] = .000003;

param_dict["harvest_cost_choice"] = 2;
param_dict["travel_cost_easy_choice"] = 4;
param_dict["travel_cost_hard_choice"] = 12;

param_dict["choice_beta"] = 1.;
param_dict["lag_beta"] = .2;

#
sim_data = sim_forage_learn_mult(param_dict)
sim_data[!,:lower_lag_thresh] .= -Inf;
sim_data[!,:upper_lag_thresh] .= Inf;
make_lag_plot(sim_data)

make_exit_plot(sim_data)

# let's try to fit this...

#### fit the EM

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


# should be ready...
@everywhere em_dir =  "/Users/evanrussek/em";
@everywhere include("$em_dir/em.jl");
@everywhere include("$em_dir/common.jl");
@everywhere include("$em_dir/likfuns.jl")

##################################
####################
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


### check the lag thresh (also make the filtering less extreme...)
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

########################

param_names = ["choice_beta", "lag_beta",
	"lr_R_hat_pre_lag", "lr_R_hat_pre_choice",
	"harvest_cost_lag", "travel_cost_easy_lag", "travel_cost_hard_lag",
	"harvest_cost_choice", "travel_cost_easy_choice", "travel_cost_hard_choice"];
NP = length(param_names);

# ok...
NS = length(unique(unique(pdata_clean.sub)))
subs = 1:NS;
# group level design matrix
X = ones(NS);

param_vals = zeros(NP)

for i = 1:NP
	param_vals[i] = param_dict[param_names[i]]
end

param_names

forage_lik_learn_mult(param_vals, sim_data)

betas = [.2 .2 -8. -8. .2 .2 .2 .2 .2 .2, 1., 1.];
sigma = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.];
forage_lik_learn_mult(betas, sim_data)


(betas,sigma,x,l,h) = em(pdata_clean,subs,X,betas,sigma,forage_lik_learn_mult; emtol=emtol, parallel=parallel, full=full);



betas

set_default_plot_size(12cm, 12cm)
cor_x = cor(x)
spy(cor_x,  label = string.(cor_x), Geom.label,Scale.y_discrete(labels = i->param_names[i]),
	Scale.x_discrete(labels = i->param_names[i]))


p_hat_dict = Dict();
for n in 1:NP
    p_hat_dict[param_names[n]] = betas[n]
end



group_rec_data = DataFrame();
for i in 1:NS
	print(i)
	these_params = x[i,:];
	for n in 1:NP
	    p_hat_dict[param_names[n]] = these_params[n]
	end
	rec_data = sim_forage_learn_mult(p_hat_dict)
	rec_data[!,:lower_lag_thresh] .= -Inf;
	rec_data[!,:upper_lag_thresh] .= Inf;
	rec_data[!,:subjectID] .= i;
	rec_data[!,:sub] .= i;
	global group_rec_data = [group_rec_data; rec_data];
end
group_rec_data[!,:remove] .= false;
rec_mn_exit = make_group_exit_plot(group_rec_data; title = "Recovered", bounds = [15, 60])
rec_mn_lag = make_group_lag_plot(group_rec_data; title = "Recovered")


draw(PNG("plots/mult_learn/exit_data_model.png", 6inch, 3inch), hstack([group_exit, rec_mn_exit]))
draw(PNG("plots/mult_learn/lag_data_model.png", 4inch, 6inch), vstack([group_lag, rec_mn_lag]))


using HypothesisTests
a = OneSampleZTest.(atanh.(cor_x),1, NS-3)
pvals = round.(pvalue.(a), digits = 4)
corr_dict = Dict();
pval_dict = Dict();
for i = 1:NP
	corr_dict[param_names[i]] = cor_x[:,i]
	pval_dict[param_names[i]] = pvals[:,i]
end
corr_dict["pname"] = param_names;
pval_dict["pname"] = param_names;

pval_df = DataFrame(pval_dict)
#pval_df = sort(pval_df, :pname)

corr_df = DataFrame(corr_dict)
#corr_df = sort(corr_df, :pname)

permutecols!(pval_df, vcat(:pname, Symbol.(param_names)))
permutecols!(corr_df, vcat(:pname, Symbol.(param_names)))
CSV.write("plots/mult_learn/corr_df.csv", corr_df)
CSV.write("plots/mult_learn/pval_df.csv", pval_df)

# do a parameter recovery check...
function gen_sim_params()
    param_dict = Dict();
    # form reward rate used for choices from trial params...
    param_dict["choice_r_hard_low_beta"] = 4*rand();
    param_dict["choice_r_easy_beta"] = 4*rand();
    param_dict["choice_r_reward_beta"] = 4*rand();

    # form reward rate used for lags from trial settings
    param_dict["lag_r_hard_low_beta"] = 4*rand();
    param_dict["lag_r_easy_beta"] = rand();
    param_dict["lag_r_reward_beta"] = rand();

    # costs (relevant only for lags...) # why are these multiplied by 10?
    param_dict["harvest_cost"] = 3*rand();
    param_dict["travel_cost_easy"] = 4*rand();
    param_dict["travel_cost_hard"] = 4*rand();

    param_dict["choice_beta"] = 2*rand();
    param_dict["lag_beta"] = #10*rand();#1.;#2*rand();

    return param_dict

end
