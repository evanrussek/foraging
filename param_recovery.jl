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

pres_df = DataFrame();
for r_idx = 1:200

    print(r_idx)

    (sim_df, param_dict) = gen_params_data();

    param_names = [];
    param_vals = Float64[];
    for (k,v) in param_dict
        #println(k,v)
        push!(param_names, k)
        push!(param_vals, v)
    end

    start_p = generate_start_vals((x) -> forage_learn_lik3(x,sim_df,"both"), length(param_vals))

    a_both = optimize(
        (x) -> forage_learn_lik3(x,sim_df, "both"),start_p, LBFGS(),
        Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
        autodiff=:forward)

    ## fit a reward sensitivity as well?

    fit_df = make_recov_df(a_both,param_names,param_dict, (x) -> forage_learn_lik3(x,sim_df,"both"))
    fit_df[!,:run] .= r_idx;
    pres_df = [fit_df; pres_df];

end

first(pres_df,6)

# 4 params...
recov_plot = plot(pres_df, x = :original_val, y = :recovered_val, xgroup =:pname,
    Geom.subplot_grid(Geom.point()), Guide.title("Parameter Recovery"),
    Guide.xlabel("Original Value by Parameter Name", orientation = :horizontal),
    Guide.ylabel("Recovered Value"))


draw(PNG("plots/p_rec_4p.png", 6inch, 6inch), recov_plot)



###

function gen_params_data()
    param_dict = Dict();
    param_dict["harvest_cost"] = 1. + 20*rand();
    param_dict["travel_cost_easy"] = 2. + 20*rand();#param_dict["harvest_cost"] + 5*rand();
    param_dict["travel_cost_hard"] = 3. + 20*rand()#;param_dict["travel_cost_easy"] + 8*rand();
    param_dict["lr_R_hat_pre"] = 2. + -12*rand();
    try
        sim_df = sim_forage2(param_dict);
        return (sim_df, param_dict)
    catch
        gen_params_data()
    end
end


function forage_learn_lik3(param_vals, data, which_data)

    #println(param_vals)

    # get rid of lookup
    #params = params .+ .0001;
    #lag_beta = param_vals[1];
    travel_cost_easy = param_vals[1]*10;
    travel_cost_hard = param_vals[2]*10;
    lr_R_hat_pre = param_vals[3];
    #println(param_vals[3])
    #println(string("lr_R_hat_pre: ", lr_R_hat_pre))

    harvest_cost = param_vals[4]*10;
    choice_beta = 1.;#param_vals[5];
    lag_beta = 1.;
    #println(string("lr_R_hat_pre: ", lr_R_hat_pre))


    lr_R_hat = (.5 + .5*erf(lr_R_hat_pre/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    # get unique trials...
    travel_costs = Dict();
    travel_costs["HARD"] = travel_cost_hard;
    travel_costs["EASY"] = travel_cost_easy;

    ###### for picking a lag -- what numbers to evaluate......
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;

    lag_ll = 0;
    choice_ll = 0;

    trial_list = unique(data[!,:trial_num])

    for trial_idx in trial_list

        # is this the problem???

        trial_data = data[data[!,:trial_num] .== trial_idx,:];

        lag = trial_data[!,:lag];
        choice = trial_data[!,:choice];
        phase = trial_data[!,:phase];
        reward_obs = trial_data[!,:reward_obs];

        trial_travel_key = trial_data[1,:travel_key_cond]
        travel_cost = travel_costs[trial_travel_key]
        trial_start_reward = trial_data[1,:start_reward];

        last_reward_obs = copy(trial_start_reward);

        #
        R_hat = convert(typeof(lag_beta),5.); # maybe you need to fit this...

        # go through each measure in trial_data
        n_presses = size(lag,1);
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(R_hat,harvest_cost, lag_beta)
                    return 1e9
                end

                current_optimal_lag = sqrt(harvest_cost / R_hat)
                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost / current_optimal_lag);
                E_opportunity_cost_harvest = R_hat*current_optimal_lag;

                theta = [choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];
                choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);

                if (choice[press_idx] == 1) # choose harvest

                    #if val_fail(R_hat,harvest_cost, lag_beta)
                    #    return 1e9
                    #end

                    this_lag = lag[press_idx];
                    lag_ll = lag_ll + log_lik_lag(R_hat,harvest_cost,lag_beta,this_lag);

                    this_cost = harvest_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(R_hat,travel_cost, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost = travel_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                end
            else # travel session - just pick a lag...

                if val_fail(R_hat,travel_cost, lag_beta)
                    return 1e9
                end
                this_lag = lag[press_idx];
                lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);
                this_cost = travel_cost / this_lag;
                this_reward = reward_obs[press_idx];
            end

            # update R_hat...
            this_move_rr = (this_reward - this_cost) / this_lag;
            R_hat = (1 - (1 - lr_R_hat)^this_lag)*this_move_rr + ((1 - lr_R_hat)^this_lag)*R_hat;

        end
    end

    if which_data == "choice"
        return -1*choice_ll
    elseif which_data == "lag"
        return -1*lag_ll
    else
        return -1*(lag_ll + choice_ll);
    end

end


### functions
function gen_params_data()
    param_dict = Dict();
    param_dict["harvest_cost"] = 1. + 5*rand();
    param_dict["travel_cost_easy"] = 2. + 6*rand();#param_dict["harvest_cost"] + 5*rand();
    param_dict["travel_cost_hard"] = 3. + 20*rand()#;param_dict["travel_cost_easy"] + 8*rand();
    param_dict["lr_R_hat_pre"] = -4. + -8*rand();
    try
        sim_df = sim_forage2(param_dict);
        return (sim_df, param_dict)
    catch
        gen_params_data()
    end
end
