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
@everywhere include("/Users/evanrussek/lockin_data/lockin_analysis/forage_data_funs.jl")
@everywhere include("sim_lag_functions.jl")
@everywhere include("sim_learn_funcs.jl")
@everywhere include("simulation_functions.jl")
@everywhere include("lik_funs.jl")


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

parallel = true # Run on multiple CPUs. If you are having trouble, set parallel = false: easier to debug
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

###############################################################
######################## fit the full model ###################
###############################################################
# then fit a model where you have 2 betas for lag...

param_names = ["choice_beta", "lag_beta",
	"lr_R_hat_pre_lag", "lr_R_hat_pre_choice",
	"harvest_cost_lag", "travel_cost_easy_lag", "travel_cost_hard_lag",
	"harvest_cost_choice", "travel_cost_easy_choice", "travel_cost_hard_choice",
	"R_hat_start_lag", "R_hat_start_choice"
	];

NP = length(param_names);

# ok...
NS = length(unique(unique(pdata_clean.sub)))
subs = 1:NS;
# group level design matrix
X = ones(NS);

betas = [.2 .2 -7. -7. .2 .2 1. 1. 1. 1. 1. 1.];
sigma = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.];
forage_lik_learn_mult(betas, sim_data)

(betas,sigma,x,l,h) = em(pdata_clean,subs,X,betas,sigma,forage_lik_learn_mult; emtol=emtol, parallel=parallel, full=full);
sigma_diag = diag(sigma);

betas_m1 = betas;
sigma_m1 = sigma;
x_m1 = x;
l_m1 = l;
h_m1 = h;
# get the IBIC
ibic_m1 = ibic(x_m1,l_m1,h_m1,betas_m1,sigma_m1,size(pdata_clean,1))
iaic_m1 = iaic(x,l,h,betas,sigma)
m1_res = Dict();
m1_res["param_names"] = param_names;
m1_res["betas"] = betas_m1;
m1_res["sigma"] = sigma_m1;å
m1_res["x"] = x_m1;
m1_res["l"] = l_m1;
m1_res["h"] = h_m1;
m1_res["ibic"] = ibic_m1;
m1_res["iaic"] = iaic_m1;
@save("model_res/full_m1.jld", m1_res)


function make_recov_plots(x, param_names, sim_func; n_rep = 2)
	# re-simulate this model...
	group_rec_data = DataFrame();
	for rep = 1:n_rep
		for i in 1:NS
			print(i)
			these_params = x_m1[i,:];
			for n in 1:NP
			    p_hat_dict[param_names[n]] = these_params[n]
			end
			rec_data = sim_func(p_hat_dict)
			rec_data[!,:lower_lag_thresh] .= -Inf;
			rec_data[!,:upper_lag_thresh] .= Inf;
			rec_data[!,:subjectID] .= NS*(rep-1)+i;
			rec_data[!,:sub] .= NS*(rep-1)+i;
			group_rec_data = [group_rec_data; rec_data];
		end
	end
	group_rec_data[!,:remove] .= false;
	rec_mn_exit = make_group_exit_plot(group_rec_data; title = "Recovered", bounds = [15, 60])
	rec_mn_lag = make_group_lag_plot(group_rec_data; title = "Recovered")

	return(rec_mn_exit, rec_mn_lag)

end

(rec_mn_exit, rec_mn_lag) = make_recov_plots(x, param_names, sim_forage_learn_mult; n_rep = 3)
# save these...
rec_mn_exit
rec_mn_lag
draw(PNG("model_res/m1_rec_choice.png", 6inch, 3inch), vstack([group_exit, rec_mn_exit]))
draw(PNG("model_res/m1_rec_lag.png", 4inch, 6inch), vstack([group_lag, rec_mn_lag]))



############################################################################
################# model 2: 1 rho, multiple lag betas########################
############################################################################

include("lik_funs.jl")

param_names = ["choice_beta", "lag_beta_opp", "lag_beta",
				"lr_R_hat_pre",
				"harvest_cost", "travel_cost_easy", "travel_cost_hard",
				"R_hat_start"];
NP = length(param_names);

X = ones(NS);

# 7 of these
betas = [1. 1. 1. -7. 1. 1. 1. 10.];
sigma = [.5, .5, 2., 2., 2., 5., 5., 10.];
forage_lik_R1_mult_lb(betas,pdata_clean)

(betas_m2,sigma_m2,x_m2,l_m2,h_m2) = em(pdata_clean,subs,X,betas,sigma,forage_lik_R1_mult_lb; emtol=emtol, parallel=parallel, full=full);
