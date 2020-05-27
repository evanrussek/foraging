
# <codecell>
cd("/Users/evanrussek/foraging/")

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

include("sim_lag_functions.jl")
include("sim_learn_funcs.jl")

# <codecell>

param_dict = Dict();
param_dict["harvest_cost"] = 1.;#.1 + 10*rand();
param_dict["travel_cost_easy"] = 2.;#param_dict["harvest_cost"] + 5*rand();
param_dict["travel_cost_hard"] = 8.#;param_dict["travel_cost_easy"] + 8*rand();
param_dict["r_hat_start_reward_weight"] = .2;#.01 + .4*rand()
param_dict["r_hat_start_easy_weight"] = 1;#.01 * 10*rand();
param_dict["harvest_lag_hat_start"] = 2.#.01 + rand()*5;#1.0; # don't fit this...
param_dict["harvest_bias"] = 0;#100;#-10 + rand()*20;
param_dict["choice_beta"] = 4.; #.001 + rand()*5;
param_dict["lag_beta"] = 2.;#.001 + rand()*8.;
param_dict["lr_R_hat_pre"] = -2.7;#-4. + 4*rand();
param_dict["lr_harvest_lag_hat_pre"] = -2.;#-4 + 5*rand();
transform_lr(param_dict["lr_R_hat_pre"])

# show original params
param_dict

# <codecell>
# simulate tasks and make plots
sim_df = sim_forage_learn(param_dict);

# <codecell>
make_exit_plot(sim_df)

# <codecell>
make_lag_plot(sim_df)

# <codecell>
plot(sim_df, x = :time, y = :reward_obs, xgroup = :travel_key_cond,ygroup = :start_reward, color = :start_reward,
    Geom.subplot_grid(Geom.line))

# <codecell>
plot(sim_df, x = :time, y = :R_hat, xgroup = :travel_key_cond,
    group = :trial_num, color = :start_reward, linestyle = :travel_key_cond,
    Geom.subplot_grid(Geom.line))

# <codecell>
plot(sim_df, x = :time, y = :harvest_lag_hat, group = :trial_num, color = :start_reward,
 linestyle = :travel_key_cond, Geom.line)

 # <codecell>
plot(sim_df, x = :time, y = :threshold, group = :trial_num, color = :start_reward,
    linestyle = :travel_key_cond, Geom.line)


# <codecell>
param_names = [];
param_vals = Float64[];
for (k,v) in param_dict
    #println(k,v)
    push!(param_names, k)
    push!(param_vals, v)
end
print(param_names) # check that this matches the order in the likelihood function...
include("sim_learn_funcs.jl")


# <codecell>
start_p = generate_start_vals((x) -> forage_learn_lik2(x,sim_df,"both"));

# check that we can take the gradient at the first value...
#ForwardDiff.gradient(cost_fun,start_x)

# <codecell>

# fit the simulated data.
a_both = optimize(
    (x) -> forage_learn_lik2(x,sim_df, "both"),start_p, LBFGS(),
    Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
    autodiff=:forward)



n_starts = 2;
best_val = 1e9;
best_struc = [];
print("start: ")
for i in 1:n_starts
    print(i)
    start_p = generate_start_vals((x) -> forage_learn_lik2(x,sim_df,"both"));
    # fit the simulated data.
    a_choice = optimize(
        (x) -> forage_learn_lik2(x,sim_df, "choice"),start_p, LBFGS(),
        Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
        autodiff=:forward)
    if a_choice.minimum < best_val
        best_val = a_choice.minimum;
        best_struc = a_choice;
    end
end
a_choice = best_struc;



# fit the simulated data.
a_lag = optimize(
    (x) -> forage_learn_lik2(x,sim_df, "lag"),start_p, LBFGS(),
    Optim.Options(allow_f_increases=true, iterations = 4000, show_trace = false),
    autodiff=:forward)



# <codecell>

## tradeoffs: rewards and costs trade off (so, both could be higher or lower)
## choice beta trades off with harvest_cost/harvest_lag_hat...
##

### don't need harvest bias?
### don't need any lag learning... -- try to fix this in the simulation...
### ...

fit_df_both = make_recov_df(a_both,param_names,param_dict, (x) -> forage_learn_lik2(x,sim_df,"both"))

cp_both, l_both = make_rec_plots(a_both.minimizer, param_names,sim_df);
cp_both

l_both

# <codecell>
fit_df_choice = make_recov_df(a_choice,param_names,param_dict, (x) -> forage_learn_lik2(x,sim_df,"choice"))
cp_choice, l_choice = make_rec_plots(a_choice.minimizer, param_names,sim_df);
cp_choice

l_choice

# <codecell>
fit_df_lag = make_recov_df(a_lag,param_names,param_dict, (x) -> forage_learn_lik2(x,sim_df,"lag"))
# lag one....
cp_lag, l_lag = make_rec_plots(a_lag.minimizer, param_names,sim_df);
cp_lag

l_lag


# <codecell>

# make a function to make side-by-side model / recovered plots...

function make_rec_plots(p_hat, param_names, sim_df)
    p_hat_dict = Dict()
    for j in 1:length(param_names)
        p_hat_dict[param_names[j]] = p_hat[j]
    end
    sim_df_rec = sim_forage_learn(p_hat_dict);
    p_rec_choice = make_exit_plot(sim_df_rec)
    p_orig_choice = make_exit_plot(sim_df)
    choice_plot = vstack([p_orig_choice; p_rec_choice]);

    # the lag looks correct...
    p_rec_lag = make_lag_plot(sim_df_rec)
    p_orig_lag = make_lag_plot(sim_df)
    lag_plot = vstack([p_orig_lag; p_rec_lag]);

    return (choice_plot, lag_plot)

end

# lag one....
cp_lag, l_lag = make_rec_plots(a_lag.minimizer, param_names,sim_df);
cp_lag

l_lag

#### let's get rid of the harvest lag prediction stuff...
