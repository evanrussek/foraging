
using CSV
using DataFrames
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

cd("/Users/evanrussek/foraging/")

# basic real data clearning, etc, functions...
include("/Users/evanrussek/lockin_data/lockin_analysis/forage_data_funs.jl")
include("sim_lag_functions.jl")
include("sim_learn_funcs.jl")
include("simulation_functions.jl")
include("lik_funs.jl")

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

param_names = ["travel_cost_easy", "travel_cost_hard", "harvest_cost", "lr_R_hat_pre", "reward_scale"];

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

# lag threshs?

function fit_subject(s_data)
    start_p = generate_start_vals((x) -> forage_lik_rs(x,s_data; lag_threshs = [s_data[1,:lower_lag_thresh],s_data[1,:upper_lag_thresh]],
        which_data = "both"), length(param_names));
    a = optimize((x) -> forage_lik_rs(x,s_data; lag_threshs = [s_data[1,:lower_lag_thresh],s_data[1,:upper_lag_thresh]], which_data = "both"),
            start_p, LBFGS(),Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
            autodiff=:forward)
    return (a.minimum, a.minimizer)
end

#(min_nll[s_idx], p_hat[s_idx,:]) #=

# fit all the subjects...
for s_idx in 1:n_subj
    print(s_idx)
    if !(s_idx in bad_subjs)
        (min_nll[s_idx], p_hat[s_idx,:]) = fit_subject(@where(pdata, :sub .== s_idx));
    end
end

# try to EM fit it...
