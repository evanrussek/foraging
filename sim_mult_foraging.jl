## this is to try to understand the foraging model...


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
make_group_exit_plot(pdata_clean)

param_dict = Dict();
# form reward rate used for choices from trial params...
param_dict["choice_r_hard_low_beta"] = 2;
param_dict["choice_r_easy_beta"] = 4;
param_dict["choice_r_reward_beta"] = 8;

# form reward rate used for lags from trial settings
param_dict["lag_r_hard_low_beta"] = 1;
param_dict["lag_r_easy_beta"] = .01;
param_dict["lag_r_reward_beta"] = .05;

# costs (relevant only for lags...) # why are these multiplied by 10?
param_dict["harvest_cost"] = 1;
param_dict["travel_cost_easy"] = 2;
param_dict["travel_cost_hard"] = 2;

param_dict["choice_beta"] = 1;
param_dict["lag_beta"] = 1;

# simulate
sim_data = sim_forage_no_learn(param_dict)
sim_data[!,:lower_lag_thresh] .= -Inf;
sim_data[!,:upper_lag_thresh] .= Inf;

make_lag_plot(sim_data)

make_exit_plot(sim_data)

#### fit this data...
param_names = ["choice_r_hard_low_beta", "choice_r_easy_beta", "choice_r_reward_beta",
                "lag_r_hard_low_beta", "lag_r_easy_beta", "lag_r_reward_beta",
                "harvest_cost", "travel_cost_easy", "travel_cost_hard",
                "choice_beta"];

n_params = length(param_names);

param_vals_orig = zeros(n_params);
for i in 1:n_params
    param_vals_orig[i] = param_dict[param_names[i]]
end

# does the likelihood run -- yes
forage_lik_nolearn(param_vals_orig, sim_data)
# generate start p for this...
n_params

start_p = generate_start_vals((x) -> forage_lik_nolearn(x, sim_data), n_params)

# fit it starting at these parameter values...
a_both = optimize(
    (x) -> forage_lik_nolearn(x,sim_data),start_p, LBFGS(),
    Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
    autodiff=:forward)


# make the new dict...
param_vals_recov = a_both.minimizer;
nll_recov = a_both.minimum;
nll_orig = forage_lik_nolearn(param_vals_orig,sim_data);

recov_df = DataFrame(name = param_names, orig = param_vals_orig, recov = a_both.minimizer);
recov_df[!,:nll_orig] .= nll_orig;
recov_df[!,:nll_recov] .= nll_recov;

recov_df

p_hat_dict = Dict();
for n in 1:n_params
    p_hat_dict[param_names[n]] = a_both.minimizer[n]
end

p_hat_dict["choice_beta"] = 1;
p_hat_dict["lag_beta"] = 1;

rec_data = sim_forage_no_learn(p_hat_dict)
rec_data[!,:lower_lag_thresh] .= -Inf;
rec_data[!,:upper_lag_thresh] .= Inf;

e_orig =  make_exit_plot(sim_data)
e_rec = make_exit_plot(rec_data)

hstack([e_orig, e_rec])

l_orig =  make_lag_plot(sim_data)
l_rec = make_lag_plot(rec_data)

vstack([l_orig, l_rec])

#### now do it for a range of  values...

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


param_recov_df = DataFrame();
for s_idx = 1:100
    print(s_idx)
    param_dict = gen_sim_params();
    sim_data = sim_forage_no_learn(param_dict)
    sim_data[!,:lower_lag_thresh] .= -Inf;
    sim_data[!,:upper_lag_thresh] .= Inf;

    param_vals_orig = zeros(n_params);
    for i in 1:n_params
        param_vals_orig[i] = param_dict[param_names[i]]
    end

    start_p = generate_start_vals((x) -> forage_lik_nolearn(x, sim_data), n_params)

    a_both = optimize(
        (x) -> forage_lik_nolearn(x,sim_data),start_p, LBFGS(),
        Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
        autodiff=:forward)

    # make the new dict...
    param_vals_recov = a_both.minimizer;
    nll_recov = a_both.minimum;
    nll_orig = forage_lik_nolearn(param_vals_orig,sim_data);

    recov_df = DataFrame(name = param_names, orig = param_vals_orig, recov = a_both.minimizer);
    recov_df[!,:nll_orig] .= nll_orig;
    recov_df[!,:nll_recov] .= nll_recov;
    recov_df[!,:sim_num] .= s_idx;

    global param_recov_df = [param_recov_df; recov_df];
end

# plot this...
plot(@where(param_recov_df, :name .== param_names[2]), x = :orig, y = :recov)

using Compose
set_default_plot_size(12cm, 20cm)
M = Array{Union{Compose.Context, Plot}}(undef,4,3)
for p_idx = 1:n_params
    p = plot(@where(param_recov_df,:name .== param_names[p_idx], :recov .< 10, :recov .> -10),
     x = :orig, y = :recov,
     Guide.title(param_names[p_idx]),
     Coord.Cartesian(ymin= 0 ,ymax=4, xmin = 0, xmax = 4))
    M[p_idx] = p;
end

for i = n_params+1:12
    M[i] = context()
end


param_recov_plot1 = gridstack(M)

draw(PNG("plots/nolearn_nob_recovery/choice_brecov_plot.png", 6inch, 8inch), param_recov_plot1)

# now let's run it with betas as free parameters...

#

# compute the bottom threshold


sqrt(p_hat_dict["travel_cost_hard"]*10)*sqrt(p_hat_dict["lag_r_hard_low_beta"]) +

p_hat_dict["choice_r_hard_low_beta"]
