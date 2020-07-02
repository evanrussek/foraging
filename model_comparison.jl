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
# figure out what went wrong here... # more than just this...
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
	"R_hat_start_lag", "R_hat_start_choice"];

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


function make_recov_plots(x_m1, param_names, sim_func; n_rep = 2)
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

### make this plot...

# save these...
rec_mn_exit
rec_mn_lag


draw(PNG("model_res/group_exit.png", 3inch, 3inch), group_exit)
draw(PNG("model_res/rec_m1_exit.png", 3inch, 3inch), rec_mn_exit)
draw(PNG("model_res/group_lag.png", 3.3inch, 3inch), group_lag)
draw(PNG("model_res/rec_m1_lag.png", 3.5inch, 3inch), rec_mn_lag)




draw(PNG("model_res/m1_rec_choice.png", 6inch, 3inch), vstack([group_exit, rec_mn_exit]))

draw(PNG("model_res/m1_rec_choice.png", 6inch, 3inch), vstack([group_exit, rec_mn_exit]))


draw(PNG("model_res/m1_rec_lag.png", 4inch, 6inch), vstack([group_lag, rec_mn_lag]))



############################################################################
################# model 2: 1 rho, multiple lag betas########################
############################################################################

@everywhere include("lik_funs.jl")

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
# get the bic and save...

ibic_m2 = ibic(x_m2,l_m2,h_m2,betas_m2,sigma_m2,size(pdata_clean,1))
iaic_m2 = iaic(x_m2,l_m2,h_m2,betas_m2,sigma_m2);
m2_res = Dict();
m2_res["param_names"] = param_names;
m2_res["betas"] = betas_m2;
m2_res["sigma"] = sigma_m2;
m2_res["x"] = x_m2;
m2_res["l"] = l_m2;
m2_res["h"] = h_m2;
m2_res["ibic"] = ibic_m2;
m2_res["iaic"] = iaic_m2;
@save("model_res/full_m2.jld", m2_res)

### let's simulate and plot this? ... do it in a bit though...


###############################################################################
##################### MODEL 3 no R_hat learning for R #########################
###############################################################################

param_names = ["choice_beta", "lag_beta", "lr_R_hat_pre_choice",
	"harvest_cost_lag", "travel_cost_easy_lag", "travel_cost_hard_lag",
	"harvest_cost_choice", "travel_cost_easy_choice", "travel_cost_hard_choice",
	"R_hat_start_lag", "R_hat_start_choice"
	];

NP = length(param_names);

X = ones(NS);

# 7 of these
betas = [0.174387 1.27726 -8.15324 7.19817 7.52151 11.9106 -0.405248 20.5891 49.2173 32.9397 -4.65928];

sigma = [0.00553526, 1.90876, 5.00612, 13.6704, 15.2953, 41.2576, 4.51844, 144.717, 365.065, 332.238, 55.373];
forage_lik_learn_mult_nolagR(betas,pdata_clean)

(betas_m3,sigma_m3,x_m3,l_m3,h_m3) = em(pdata_clean,subs,X,betas,sigma,forage_lik_learn_mult_nolagR; emtol=emtol, parallel=parallel, full=full);

# get the bic...
ibic_m3 = ibic(x_m3,l_m3,h_m3,betas_m3,sigma_m3,size(pdata_clean,1))
iaic_m3 = iaic(x_m3,l_m3,h_m3,betas_m3,sigma_m3);
m3_res = Dict();
m3_res["param_names"] = param_names;
m3_res["betas"] = betas_m3;
m3_res["sigma"] = sigma_m3;
m3_res["x"] = x_m3;
m3_res["l"] = l_m3;
m3_res["h"] = h_m3;
m3_res["ibic"] = ibic_m3;
m3_res["iaic"] = iaic_m3;
@save("model_res/full_m3.jld", m3_res)


###########

param_names4 = ["choice_beta", "lag_beta", "lr_R_hat_pre_lag",
	"harvest_cost_lag", "travel_cost_easy_lag", "travel_cost_hard_lag",
	"harvest_cost_choice",
	"R_hat_start_lag", "R_hat_start_choice"
	];

NP = length(param_names4);

X = ones(NS);

# 7 of these
betas = [0.174387 1.27726 -8.15324 7.19817 7.52151 11.9106 -0.405248 32.9397 -4.65928];

sigma = [0.00553526, 1.90876, 5.00612, 13.6704, 15.2953, 41.2576, 4.51844, 332.238, 55.373];
forage_lik_learn_mult_nochoiceR(betas,pdata_clean)
# 9 parameters...
(betas_m4,sigma_m4,x_m4,l_m4,h_m4) = em(pdata_clean,subs,X,betas,sigma,forage_lik_learn_mult_nochoiceR; emtol=emtol, parallel=false, full=full);


# get the bic...
ibic_m4 = ibic(x_m4,l_m4,h_m4,betas_m4,sigma_m4,size(pdata_clean,1))
iaic_m3 = iaic(x_m3,l_m3,h_m3,betas_m3,sigma_m3);


m4_res = Dict();
m4_res["param_names"] = param_names;
m4_res["betas"] = betas_m4;
m4_res["sigma"] = sigma_m4;
m4_res["x"] = x_m4;
m4_res["l"] = l_m4;
m4_res["h"] = h_m4;
m4_res["ibic"] = ibic_m4;
m4_res["iaic"] = iaic_m4;
@save("model_res/full_m4.jld", m4_res)

############################################################################
####### try removing the last subject from all the comparison #############
###########################################################################

# subject 25 is the problem...
# redo the bics, removing outlies... (might need to re-do this down the road.)

cond_idx = .&(l_m4 .< .5e5,  l_m5 .< .5e5); ### check on these individuals, and see whether it makes sense to exclude them...

#### check this more in a bit...
ibic_m5 = ibic(x_m5[cond_idx,:],l_m5[cond_idx],h_m5[:,:,cond_idx],
			betas_m5,sigma_m5,size(pdata_clean,1))

ibic_m4 = ibic(x_m4[cond_idx,:],l_m4[cond_idx],h_m4[:,:,cond_idx],
			betas_m4,sigma_m4,size(pdata_clean,1))

ibic_m3 = ibic(x_m3[cond_idx,:],l_m3[cond_idx],h_m3[:,:,cond_idx],
			betas_m3,sigma_m3,size(pdata_clean,1))
### look at outliers in the data...

ibic_m2 = ibic(x_m2[cond_idx,:],l_m2[cond_idx],h_m2[:,:,cond_idx],
			betas_m2,sigma_m2,size(pdata_clean,1))

ibic_m1 = ibic(x_m1[cond_idx,:],l_m1[cond_idx],h_m1[:,:,cond_idx],
			betas_m1,sigma_m1,size(pdata_clean,1))

best = [0,]

# make a bar plot...
set_default_plot_size(4inch,4inch)
bic_vals = [ibic_m2, ibic_m1,  ibic_m3, ibic_m4]
model_names = ["Shared ρ", "Different ρ", "No Lag ρ", "No Choice ρ"];
model_comp_bar = plot(x = model_names, y = bic_vals, color = [false,true,false,false],
	Geom.bar(orientation =:vertical),
	Guide.ylabel("iBIC"), Guide.xlabel("Model"),
	Theme(bar_spacing=2mm,key_position = :none,
	minor_label_font_size=10pt, major_label_font_size=12pt))
# save this...
draw(PNG("model_res/iBIC_comp.png", 4inch, 3inch), model_comp_bar)


# show that best model can capture lag and choice data...



#### if you add a harvest bias to the single model, can it fit the data?



################################################################################################
################# model 5: 1 rho, multiple lag betas, add harvest bias##########################
################################################################################################

@everywhere include("lik_funs.jl")

param_names = ["choice_beta", "lag_beta_opp", "lag_beta",
				"lr_R_hat_pre",
				"harvest_cost", "travel_cost_easy", "travel_cost_hard",
				"R_hat_start", "harvest_bias"];
NP = length(param_names);

X = ones(NS);

# 7 of these
betas = [1. 1. 1. -7. 1. 1. 1. 10. 10.];
sigma = [.5, .5, 2., 2., 2., 5., 5., 10., 10.];
forage_lik_R1_mult_lb(betas,pdata_clean)
cost_fun = (x,data) -> forage_lik_R1_mult_lb(x,data; add_h_bias = true)

(betas_m5,sigma_m5,x_m5,l_m5,h_m5) = em(pdata_clean,subs,X,betas,sigma,cost_fun; emtol=emtol, parallel=parallel, full=full);
# get the bic and save...

ibic_m5 = ibic(x_m5,l_m5,h_m5,betas_m5,sigma_m5,size(pdata_clean,1))
iaic_m5 = iaic(x_m5,l_m5,h_m5,betas_m5,sigma_m5);
m5_res = Dict();
m5_res["param_names"] = param_names;
m5_res["betas"] = betas_m5;
m5_res["sigma"] = sigma_m5;
m5_res["x"] = x_m5;
m5_res["l"] = l_m5;
m5_res["h"] = h_m5;
m5_res["ibic"] = ibic_m5;
m5_res["iaic"] = iaic_m5;
@save("model_res/full_m5.jld", m5_res)

### in the future,
