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
using Cairo
using Fontconfig

include("sim_lag_functions.jl")
include("sim_learn_funcs.jl")
include("simulation_functions.jl")

param_dict = Dict();
param_dict["harvest_cost"] = 5.;#.1 + 10*rand();
param_dict["travel_cost_easy"] = 5.;#param_dict["harvest_cost"] + 5*rand();
param_dict["travel_cost_hard"] = 15.#;param_dict["travel_cost_easy"] + 8*rand();
param_dict["reward_scale"] = 2.; # reward scale...
param_dict["lr_R_hat_pre"] = -8;   #-4. + 4*rand();
lr_R_hat = (.5 + .5*erf(param_dict["lr_R_hat_pre"]/5));
# maybe also fit the start reward

reward_scales = [.25, .5, 1., 2.];
rs_exit = DataFrame();
data = DataFrame();
for rs in reward_scales
    param_dict["reward_scale"] = rs; # reward scale...
    rs_data = sim_forage3(param_dict);
    rs_data[!,:rs] .= rs;
    data = [data; rs_data];
end

exit_tables = DataFrame();
for rs in reward_scales
    this_exit_tbl = make_subj_trial_exit(@where(data, :rs .== rs));
    this_exit_tbl[!,:rs] .= rs;
    exit_tables = [exit_tables; this_exit_tbl];
end

lag_tables = DataFrame();
for rs in reward_scales
    this_lag_tbl = make_subj_lag_table(@where(data, :rs .== rs));
    this_lag_tbl[!,:rs] .= rs;
    lag_tables = [lag_tables; this_lag_tbl];
end


#plot(sim_df, x = :time, y = :R_hat, xgroup = :travel_key_cond,
#    group = :trial_num, color = :start_reward, linestyle = :travel_key_cond,
#    Geom.subplot_grid(Geom.line))


# does the original optimal policy show this effect...
plot(data, x = :time, y = :R_hat, xgroup = :travel_key_cond,
    ygroup = :rs,
    group = :trial_num, color = :start_reward, linestyle = :travel_key_cond,
    Geom.subplot_grid(Geom.line))

#
plot(exit_tables, x= :start_reward_cat, y =:exit_thresh, color = :travel_key_cond, xgroup = :rs,
    Geom.subplot_grid(Geom.line(), Geom.point()), Scale.x_discrete(levels = sort(unique(exit_tables[!,:start_reward_cat]))),
        Scale.color_discrete(levels = ["HARD", "EASY"]),
        Guide.title("Effect of Reward Scaling on Choice"))


set_default_plot_size(10cm, 15cm)
p = plot(lag_tables, x = :start_reward_cat, y = :log_lag, color = :travel_key_cond,
            xgroup = :phase, ygroup = :rs, Geom.subplot_grid(Geom.line(), Geom.point()),
            Scale.x_discrete(levels = sort(unique(lag_tables[!,:start_reward_cat]))),
            Scale.color_discrete(levels = ["HARD", "EASY"])
            )




names(data)

plot(data, x = :time, y = :R_hat, xgroup = :travel_key_cond,
    group = :trial_num, color = :start_reward, linestyle = :travel_key_cond,
    Geom.subplot_grid(Geom.line))

plot(data, x = :time, y = :lag, xgroup = :travel_key_cond,
        group = :trial_num, color = :start_reward, linestyle = :travel_key_cond,
        Geom.subplot_grid(Geom.line))


make_exit_plot(data)

make_lag_plot(data)
