a = 1

cd("/Users/evanrussek/foraging/")


include("sim_lag_functions.jl")

R_hat = .5
press_cost = 10.
beta = 1.

function stay_dv(r_last, rho)
    lag = .5;
    press_cost = 10.
    return 1 ./ (1 + exp( -1 .* ( .5*r_last*.98 - press_cost/lag - rho*lag)))
end

plot([(x) -> stay_dv(x,.5), (x) -> stay_dv(x,10.)], 0, 100,
    Guide.xlabel("Last reward observed"), Guide.ylabel("Harvest Prob"),
    Guide.colorkey(title = "Rho", labels = [".5", "10"]))

include("sim_learn_funcs.jl")

param_dict = Dict();
param_dict["harvest_cost"] = 1.;#.1 + 10*rand();
param_dict["travel_cost_easy"] = 2.;#param_dict["harvest_cost"] + 5*rand();
param_dict["travel_cost_hard"] = 10.#;param_dict["travel_cost_easy"] + 8*rand();
#param_dict["choice_beta"] = 1;#4.; #.001 + rand()*5;
#param_dict["lag_beta"] = 1.;#.001 + rand()*8.;
param_dict["lr_R_hat_pre"] = -8;#-4. + 4*rand();
lr_R_hat = (.5 + .5*erf(param_dict["lr_R_hat_pre"]/5))/10

# -1.9 is the highest the LR can be w/o crashing...

sim_df = sim_forage2(param_dict);



plot(sim_df, x = :time, y = :reward_obs, xgroup = :travel_key_cond,ygroup = :start_reward, color = :start_reward,
    Geom.subplot_grid(Geom.point))

plot(sim_df, x = :time, y = :R_hat, xgroup = :travel_key_cond,
    group = :trial_num, color = :start_reward, linestyle = :travel_key_cond,
    Geom.subplot_grid(Geom.line))

plot(sim_df, x = :time, y = :lag, xgroup = :travel_key_cond,
        group = :trial_num, color = :start_reward, linestyle = :travel_key_cond,
        Geom.subplot_grid(Geom.line))

plot(sim_df, x = :time, y = :threshold, group = :trial_num, color = :start_reward,
    linestyle = :travel_key_cond, Geom.line)

make_exit_plot(sim_df)
# why does it cross at 60?

make_lag_plot(sim_df)




### try to fit it...

# what are the param names?

param_names = [];
param_vals = Float64[];
for (k,v) in param_dict
    #println(k,v)
    push!(param_names, k)
    push!(param_vals, v)
end
print(param_names) # check that this matches the order in the likelihood function...
param_vals

#forage_learn_lik3(param_vals,sim_df, "both")


lr_R_hat = (.5 + .5*erf(param_vals[3]/5))/10

(-4)^(1/2)

param_vals

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


start_p = generate_start_vals((x) -> forage_learn_lik3(x,sim_df,"both"), length(param_vals))


a_both = optimize(
    (x) -> forage_learn_lik3(x,sim_df, "both"),start_p, LBFGS(),
    Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
    autodiff=:forward)

fit_df_both = make_recov_df(a_both,param_names,param_dict, (x) -> forage_learn_lik3(x,sim_df,"both"))

lr_R_hat = (.5 + .5*erf(param_dict["lr_R_hat_pre"]/5))/10



# the original here is throwing... impossible values


param_dict




# do you need lag beta?


param_dict
### recovered has lower than original
param_names

a_both.minimizer
### model fitting favors impossibly high learning rates
### but lower betas...



function make_rec_plots(p_hat, param_names, sim_df)
    p_hat_dict = Dict()
    for j in 1:length(param_names)
        p_hat_dict[param_names[j]] = p_hat[j]
    end
    sim_df_rec = sim_forage2(p_hat_dict);
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
cp_lag, l_lag = make_rec_plots(a_both.minimizer, param_names,sim_df);
cp_lag

### domain error

##
plot([(x) -> (.5 + .5*erf(x/2))/10, (x)-> .001, (x) -> .003], -4, 4)
plot([(x) -> log(x), (x)-> .001], 0, 10)

sqrt(2)
