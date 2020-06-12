
function inrange(x, lower,upper)
    return (x >= lower) && (x <= upper)
end


function forage_lik_rs(param_vals, data; which_data = "both")

    # edit this so that lag_threshs is in the data...
    lag_threshs = [data[1,:lower_lag_thresh],data[1,:upper_lag_thresh]];

    #println(param_vals)

    # get rid of lookup
    #params = params .+ .0001;
    #lag_beta = param_vals[1];
    travel_cost_easy = param_vals[1]*10;
    travel_cost_hard = param_vals[2]*10;
    harvest_cost = param_vals[3]*10;
    lr_R_hat_pre = param_vals[4];
    reward_scale = 1.#param_vals[5];
    R_hat_start = param_vals[5]*10;
    choice_beta = param_vals[6];
    lag_beta = param_vals[7];
    harvest_bias = param_vals[8]*10;
    lag_rr_sen = param_vals[9]; # lag reward rate sensitivity

    #choice_beta = 1.;#param_vals[5];
    #lag_beta = 1.;
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

        R_hat = copy(R_hat_start);#20*reward_scale;

        # fit this...
        #R_hat = convert(typeof(lag_beta), 20. * reward_scale); # maybe you need to fit this...

        # go through each measure in trial_data
        n_presses = size(lag,1);

        first_round_harvest = true;
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(R_hat*lag_rr_sen,harvest_cost, lag_beta)
                    return 1e9
                end

                current_optimal_lag = sqrt(harvest_cost / R_hat*lag_rr_sen) # edit this so it's a recent average...
                E_next_reward_harvest = reward_scale*last_reward_obs*decay*harvest_success_prob - (harvest_cost / current_optimal_lag);
                E_opportunity_cost_harvest = R_hat*current_optimal_lag;

                theta = [choice_beta*(harvest_bias + E_next_reward_harvest), choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                            lag_ll = lag_ll + log_lik_lag(R_hat*lag_rr_sen,harvest_cost,lag_beta,this_lag);
                        end
                    else
                        first_round_harvest = false;
                    end

                    this_cost = harvest_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(R_hat*lag_rr_sen,travel_cost, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    # don't add first press to lag...
                    # lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost = travel_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                    first_round_harvest = true;
                end
            else # travel session - just pick a lag...

                if val_fail(R_hat*lag_rr_sen,travel_cost, lag_beta)
                    return 1e9
                end

                this_lag = lag[press_idx];
                if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                    lag_ll = lag_ll + log_lik_lag(R_hat*lag_rr_sen,travel_cost,lag_beta,this_lag);
                end
                this_cost = travel_cost / this_lag;
                this_reward = reward_obs[press_idx];
            end

            # update R_hat...
            this_move_rr = (reward_scale*this_reward - this_cost) / this_lag;
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

function forage_lik_nolearn(param_vals, data; which_data = "both")

    # edit this so that lag_threshs is in the data...
    lag_threshs = [data[1,:lower_lag_thresh],data[1,:upper_lag_thresh]];

    # form reward rate used for choices from trial params...
    choice_r_hard_low_beta = param_vals[1];
    choice_r_easy_beta = param_vals[2];
    choice_r_reward_beta = param_vals[3];

    # form reward rate used for lags from trial settings
    lag_r_hard_low_beta = param_vals[4];
    lag_r_easy_beta = param_vals[5];
    lag_r_reward_beta = param_vals[6];

    # don't multiply these by 10...
    # costs (relevant only for lags...) # why are these multiplied by 10?
    harvest_cost = param_vals[7];
    travel_cost_easy = param_vals[8];
    travel_cost_hard = param_vals[9];
    choice_beta = param_vals[10];#param_dict["choice_beta"];  # choice_beta
    lag_beta = 1;#param_vals[11];#param_dict["choice_beta"];  # choice_beta


    # factor to scale rewards...
    reward_scale = 1.;

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

        # R_hat...
        R_hat_choice = choice_r_hard_low_beta + ((trial_start_reward - 60)/60)*choice_r_reward_beta + choice_r_easy_beta*(trial_travel_key == "EASY");
        R_hat_lag = lag_r_hard_low_beta + ((trial_start_reward - 60)/60)*lag_r_reward_beta + lag_r_easy_beta*(trial_travel_key == "EASY");

        # go through each measure in trial_data
        n_presses = size(lag,1);

        first_round_harvest = true;
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                # relevant for R_hat_lag
                if val_fail(R_hat_lag,harvest_cost, lag_beta)
                    return 1e9
                end

                current_optimal_lag = sqrt(harvest_cost / R_hat_lag) # edit this so it's a recent average...
                E_next_reward_harvest = reward_scale*last_reward_obs*decay*harvest_success_prob - (harvest_cost / current_optimal_lag);
                E_opportunity_cost_harvest = R_hat_choice*current_optimal_lag;

                theta = [choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                            lag_ll = lag_ll + log_lik_lag(R_hat_lag,harvest_cost,lag_beta,this_lag);
                        end
                    else
                        first_round_harvest = false;
                    end

                    this_cost = harvest_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(R_hat_lag,travel_cost, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    # don't add first press to lag...
                    # lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost = travel_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                    first_round_harvest = true;
                end
            else # travel session - just pick a lag...

                if val_fail(R_hat_lag,travel_cost, lag_beta)
                    return 1e9
                end

                this_lag = lag[press_idx];
                if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                    lag_ll = lag_ll + log_lik_lag(R_hat_lag,travel_cost,lag_beta,this_lag);
                end
                this_cost = travel_cost / this_lag;
                this_reward = reward_obs[press_idx];
            end
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


#############################################################
##################### lik for multiple processes, learning
##############################################################
###### check whether this is recoverable...

# let it be a free parameter, whether to

function forage_lik_learn_mult(param_vals, data; which_data = "both")

    # edit this so that lag_threshs is in the data...
    # check this...
    lag_threshs = [data[1,:lower_lag_thresh],data[1,:upper_lag_thresh]];

    # spread parameters
    choice_beta = param_vals[1];
    lag_beta = param_vals[2];
    # learning rate parameters
    lr_R_hat_pre_lag = param_vals[3];
    lr_R_hat_pre_choice = param_vals[4];
    # lag costs
    harvest_cost_lag = param_vals[5]*10;
    travel_cost_easy_lag = param_vals[6]*10;
    travel_cost_hard_lag = param_vals[7]*10;
    # harvest costs
    harvest_cost_choice = param_vals[8]*10;
    travel_cost_easy_choice = param_vals[9]*10;
    travel_cost_hard_choice = param_vals[10]*10;

    # constants...
    R_hat_start_lag = param_vals[11];#3.;# reward rate for choice... # is this good?
    R_hat_start_choice = param_vals[12]; #3.;# reward rate for lag... # good?

    lr_harvest_lag_hat = .3; # set this arbitrarily # only relevant for choice...
    harvest_lag_hat_start = 2.; # only relevant for choice...

    # learning rate for reward rate for choice and lag...
    lr_R_hat_choice = (.5 + .5*erf(lr_R_hat_pre_choice/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    lr_R_hat_lag = (.5 + .5*erf(lr_R_hat_pre_lag/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));

    # travel costs
    travel_costs_lag = Dict();
    travel_costs_lag["HARD"] = travel_cost_hard_lag;
    travel_costs_lag["EASY"] = travel_cost_easy_lag;
    travel_costs_choice = Dict();
    travel_costs_choice["HARD"] = travel_cost_hard_choice;
    travel_costs_choice["EASY"] = travel_cost_easy_choice;

    ###### for picking a lag -- what numbers to evaluate......
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;

    lag_ll = 0;
    choice_ll = 0;

    trial_list = unique(data[!,:trial_num])

    for trial_idx in trial_list

        trial_data = data[data[!,:trial_num] .== trial_idx,:];

        lag = trial_data[!,:lag];
        choice = trial_data[!,:choice];
        phase = trial_data[!,:phase];
        reward_obs = trial_data[!,:reward_obs];

        trial_travel_key = trial_data[1,:travel_key_cond]
        travel_cost_lag = travel_costs_lag[trial_travel_key]
        travel_cost_choice = travel_costs_choice[trial_travel_key]
        trial_start_reward = trial_data[1,:start_reward];

        last_reward_obs = copy(trial_start_reward);

        R_hat_choice = copy(R_hat_start_choice);
        R_hat_lag = copy(R_hat_start_lag);
        harvest_lag_hat = copy(harvest_lag_hat_start);

        # go through each measure in trial_data
        n_presses = size(lag,1);

        first_round_harvest = true;
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(R_hat_lag,harvest_cost_lag, lag_beta)
                    return 1e9
                end

                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost_choice / harvest_lag_hat);
                E_opportunity_cost_harvest = R_hat_choice*harvest_lag_hat;

                theta = [choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # do learnng of the harvest lag...
                    harvest_lag_hat = (1 - lr_harvest_lag_hat)*harvest_lag_hat + lr_harvest_lag_hat*this_lag;

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                            lag_ll = lag_ll + log_lik_lag(R_hat_lag,harvest_cost_lag,lag_beta,this_lag);
                        end
                    else
                        first_round_harvest = false;
                    end

                    this_cost_lag = harvest_cost_lag / this_lag;
                    this_cost_choice = harvest_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(R_hat_lag,travel_cost_lag, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    # don't add first press to lag...
                    # lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost_lag = travel_cost_lag / this_lag;
                    this_cost_choice = travel_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                    first_round_harvest = true;
                end
            else # travel session - just pick a lag...

                if val_fail(R_hat_lag,travel_cost_lag, lag_beta)
                    return 1e9
                end

                this_lag = lag[press_idx];
                if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                    lag_ll = lag_ll + log_lik_lag(R_hat_lag,travel_cost_lag,lag_beta,this_lag);
                end
                this_cost_lag = travel_cost_lag / this_lag;
                this_cost_choice = travel_cost_choice / this_lag;
                this_reward = reward_obs[press_idx];
            end

            ## update R_hat...
            this_move_rr_choice = (this_reward - this_cost_choice) / this_lag;
            this_move_rr_lag = (this_reward - this_cost_lag) / this_lag;

            R_hat_choice = (1 - (1 - lr_R_hat_choice)^this_lag)*this_move_rr_choice + ((1 - lr_R_hat_choice)^this_lag)*R_hat_choice;
            R_hat_lag = (1 - (1 - lr_R_hat_lag)^this_lag)*this_move_rr_lag + ((1 - lr_R_hat_lag)^this_lag)*R_hat_lag; # probably want the same LR for both

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


### single R, mult lag beta
function forage_lik_R1_mult_lb(param_vals, data; which_data = "both", add_h_bias = false)

    # edit this so that lag_threshs is in the data...
    # check this...
    lag_threshs = [data[1,:lower_lag_thresh],data[1,:upper_lag_thresh]];

    # spread parameters
    choice_beta = param_vals[1];
    lag_beta_opp = param_vals[2];
    lag_beta = param_vals[3];

    # learning rate parameters
    lr_R_hat_pre_lag = param_vals[4];
    lr_R_hat_pre_choice = lr_R_hat_pre_lag;#param_vals[4];
    # lag costs
    harvest_cost_lag = param_vals[5]*10;
    travel_cost_easy_lag = param_vals[6]*10;
    travel_cost_hard_lag = param_vals[7]*10;
    # harvest costs
    harvest_cost_choice = harvest_cost_lag;# param_vals[8]*10;
    travel_cost_easy_choice = travel_cost_easy_lag;#param_vals[9]*10;
    travel_cost_hard_choice = travel_cost_hard_lag;#param_vals[10]*10;

    # constants...
    R_hat_start_lag = param_vals[8];#3.;# reward rate for choice... # is this good?
    R_hat_start_choice = R_hat_start_lag;#param_vals[12]; #3.;# reward rate for lag... # good?

    if add_h_bias
        harvest_bias = param_vals[9];
    else
        harvest_bias = 0;
    end

    lr_harvest_lag_hat = .3; # set this arbitrarily # only relevant for choice...
    harvest_lag_hat_start = 2.; # only relevant for choice...

    # learning rate for reward rate for choice and lag...
    lr_R_hat_choice = (.5 + .5*erf(lr_R_hat_pre_choice/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    lr_R_hat_lag = (.5 + .5*erf(lr_R_hat_pre_lag/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));

    # travel costs
    travel_costs_lag = Dict();
    travel_costs_lag["HARD"] = travel_cost_hard_lag;
    travel_costs_lag["EASY"] = travel_cost_easy_lag;
    travel_costs_choice = Dict();
    travel_costs_choice["HARD"] = travel_cost_hard_choice;
    travel_costs_choice["EASY"] = travel_cost_easy_choice;

    ###### for picking a lag -- what numbers to evaluate......
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;

    lag_ll = 0;
    choice_ll = 0;

    trial_list = unique(data[!,:trial_num])

    for trial_idx in trial_list

        trial_data = data[data[!,:trial_num] .== trial_idx,:];

        lag = trial_data[!,:lag];
        choice = trial_data[!,:choice];
        phase = trial_data[!,:phase];
        reward_obs = trial_data[!,:reward_obs];

        trial_travel_key = trial_data[1,:travel_key_cond]
        travel_cost_lag = travel_costs_lag[trial_travel_key]
        travel_cost_choice = travel_costs_choice[trial_travel_key]
        trial_start_reward = trial_data[1,:start_reward];

        last_reward_obs = copy(trial_start_reward);

        R_hat_choice = copy(R_hat_start_choice);
        R_hat_lag = copy(R_hat_start_lag);
        harvest_lag_hat = copy(harvest_lag_hat_start);

        # go through each measure in trial_data
        n_presses = size(lag,1);

        first_round_harvest = true;
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(lag_beta_opp*R_hat_lag,harvest_cost_lag, lag_beta)
                    return 1e9
                end

                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost_choice / harvest_lag_hat);
                E_opportunity_cost_harvest = R_hat_choice*harvest_lag_hat;

                theta = [harvest_bias + choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # do learnng of the harvest lag...
                    harvest_lag_hat = (1 - lr_harvest_lag_hat)*harvest_lag_hat + lr_harvest_lag_hat*this_lag;

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                            lag_ll = lag_ll + log_lik_lag(lag_beta_opp*R_hat_lag,harvest_cost_lag,lag_beta,this_lag);
                        end
                    else
                        first_round_harvest = false;
                    end

                    this_cost_lag = harvest_cost_lag / this_lag;
                    this_cost_choice = harvest_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(lag_beta_opp*R_hat_lag,travel_cost_lag, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    # don't add first press to lag...
                    # lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost_lag = travel_cost_lag / this_lag;
                    this_cost_choice = travel_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                    first_round_harvest = true;
                end
            else # travel session - just pick a lag...

                if val_fail(lag_beta_opp*R_hat_lag,travel_cost_lag, lag_beta)
                    return 1e9
                end

                this_lag = lag[press_idx];
                if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                    lag_ll = lag_ll + log_lik_lag(lag_beta_opp*R_hat_lag,travel_cost_lag,lag_beta,this_lag);
                end
                this_cost_lag = travel_cost_lag / this_lag;
                this_cost_choice = travel_cost_choice / this_lag;
                this_reward = reward_obs[press_idx];
            end

            ## update R_hat...
            this_move_rr_choice = (this_reward - this_cost_choice) / this_lag;
            this_move_rr_lag = (this_reward - this_cost_lag) / this_lag;

            R_hat_choice = (1 - (1 - lr_R_hat_choice)^this_lag)*this_move_rr_choice + ((1 - lr_R_hat_choice)^this_lag)*R_hat_choice;
            R_hat_lag = (1 - (1 - lr_R_hat_lag)^this_lag)*this_move_rr_lag + ((1 - lr_R_hat_lag)^this_lag)*R_hat_lag; # probably want the same LR for both

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


function forage_lik_learn_mult_nolagR(param_vals, data; which_data = "both")

    # edit this so that lag_threshs is in the data...
    # check this...
    lag_threshs = [data[1,:lower_lag_thresh],data[1,:upper_lag_thresh]];

    # spread parameters
    choice_beta = param_vals[1];
    lag_beta = param_vals[2];
    # learning rate parameters
    #lr_R_hat_pre_lag = param_vals[3];
    lr_R_hat_pre_choice = param_vals[3];
    # lag costs
    harvest_cost_lag = param_vals[4]*10;
    travel_cost_easy_lag = param_vals[5]*10;
    travel_cost_hard_lag = param_vals[6]*10;
    # harvest costs
    harvest_cost_choice = param_vals[7]*10;
    travel_cost_easy_choice = param_vals[8]*10;
    travel_cost_hard_choice = param_vals[9]*10;

    # constants...
    R_hat_start_lag = param_vals[10];#3.;# reward rate for choice... # is this good?
    R_hat_start_choice = param_vals[11]; #3.;# reward rate for lag... # good?

    lr_harvest_lag_hat = .3; # set this arbitrarily # only relevant for choice...
    harvest_lag_hat_start = 2.; # only relevant for choice...

    # learning rate for reward rate for choice and lag...
    lr_R_hat_choice = (.5 + .5*erf(lr_R_hat_pre_choice/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    lr_R_hat_lag = 0;#(.5 + .5*erf(lr_R_hat_pre_lag/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));

    # travel costs
    travel_costs_lag = Dict();
    travel_costs_lag["HARD"] = travel_cost_hard_lag;
    travel_costs_lag["EASY"] = travel_cost_easy_lag;
    travel_costs_choice = Dict();
    travel_costs_choice["HARD"] = travel_cost_hard_choice;
    travel_costs_choice["EASY"] = travel_cost_easy_choice;

    ###### for picking a lag -- what numbers to evaluate......
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;

    lag_ll = 0;
    choice_ll = 0;

    trial_list = unique(data[!,:trial_num])

    for trial_idx in trial_list

        trial_data = data[data[!,:trial_num] .== trial_idx,:];

        lag = trial_data[!,:lag];
        choice = trial_data[!,:choice];
        phase = trial_data[!,:phase];
        reward_obs = trial_data[!,:reward_obs];

        trial_travel_key = trial_data[1,:travel_key_cond]
        travel_cost_lag = travel_costs_lag[trial_travel_key]
        travel_cost_choice = travel_costs_choice[trial_travel_key]
        trial_start_reward = trial_data[1,:start_reward];

        last_reward_obs = copy(trial_start_reward);

        R_hat_choice = copy(R_hat_start_choice);
        R_hat_lag = copy(R_hat_start_lag);
        harvest_lag_hat = copy(harvest_lag_hat_start);

        # go through each measure in trial_data
        n_presses = size(lag,1);

        first_round_harvest = true;
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(R_hat_lag,harvest_cost_lag, lag_beta)
                    return 1e9
                end

                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost_choice / harvest_lag_hat);
                E_opportunity_cost_harvest = R_hat_choice*harvest_lag_hat;

                theta = [choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # do learnng of the harvest lag...
                    harvest_lag_hat = (1 - lr_harvest_lag_hat)*harvest_lag_hat + lr_harvest_lag_hat*this_lag;

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                            lag_ll = lag_ll + log_lik_lag(R_hat_lag,harvest_cost_lag,lag_beta,this_lag);
                        end
                    else
                        first_round_harvest = false;
                    end

                    this_cost_lag = harvest_cost_lag / this_lag;
                    this_cost_choice = harvest_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(R_hat_lag,travel_cost_lag, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    # don't add first press to lag...
                    # lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost_lag = travel_cost_lag / this_lag;
                    this_cost_choice = travel_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                    first_round_harvest = true;
                end
            else # travel session - just pick a lag...

                if val_fail(R_hat_lag,travel_cost_lag, lag_beta)
                    return 1e9
                end

                this_lag = lag[press_idx];
                if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                    lag_ll = lag_ll + log_lik_lag(R_hat_lag,travel_cost_lag,lag_beta,this_lag);
                end
                this_cost_lag = travel_cost_lag / this_lag;
                this_cost_choice = travel_cost_choice / this_lag;
                this_reward = reward_obs[press_idx];
            end

            ## update R_hat...
            this_move_rr_choice = (this_reward - this_cost_choice) / this_lag;
            this_move_rr_lag = (this_reward - this_cost_lag) / this_lag;

            R_hat_choice = (1 - (1 - lr_R_hat_choice)^this_lag)*this_move_rr_choice + ((1 - lr_R_hat_choice)^this_lag)*R_hat_choice;
            R_hat_lag = (1 - (1 - lr_R_hat_lag)^this_lag)*this_move_rr_lag + ((1 - lr_R_hat_lag)^this_lag)*R_hat_lag; # probably want the same LR for both

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

#################################### nochoiceR
function forage_lik_learn_mult_nochoiceR(param_vals, data; which_data = "both")

    # edit this so that lag_threshs is in the data...
    # check this...
    lag_threshs = [data[1,:lower_lag_thresh],data[1,:upper_lag_thresh]];

    # spread parameters
    choice_beta = param_vals[1];
    lag_beta = param_vals[2];
    # learning rate parameters
    lr_R_hat_pre_lag = param_vals[3];
    #lr_R_hat_pre_choice = param_vals[3];
    # lag costs
    harvest_cost_lag = param_vals[4]*10;
    travel_cost_easy_lag = param_vals[5]*10;
    travel_cost_hard_lag = param_vals[6]*10;
    # harvest costs
    harvest_cost_choice = param_vals[7]*10;
    travel_cost_easy_choice = 0;#param_vals[8]*10;
    travel_cost_hard_choice = 0;#param_vals[9]*10;

    # constants...
    R_hat_start_lag = param_vals[8];#3.;# reward rate for choice... # is this good?
    R_hat_start_choice = param_vals[9]; #3.;# reward rate for lag... # good?

    lr_harvest_lag_hat = .3; # set this arbitrarily # only relevant for choice...
    harvest_lag_hat_start = 2.; # only relevant for choice...

    # learning rate for reward rate for choice and lag...
    lr_R_hat_choice = 0;#(.5 + .5*erf(lr_R_hat_pre_choice/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    lr_R_hat_lag = (.5 + .5*erf(lr_R_hat_pre_lag/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));

    # travel costs
    travel_costs_lag = Dict();
    travel_costs_lag["HARD"] = travel_cost_hard_lag;
    travel_costs_lag["EASY"] = travel_cost_easy_lag;
    travel_costs_choice = Dict();
    travel_costs_choice["HARD"] = travel_cost_hard_choice;
    travel_costs_choice["EASY"] = travel_cost_easy_choice;

    ###### for picking a lag -- what numbers to evaluate......
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;

    lag_ll = 0;
    choice_ll = 0;

    trial_list = unique(data[!,:trial_num])

    for trial_idx in trial_list

        trial_data = data[data[!,:trial_num] .== trial_idx,:];

        lag = trial_data[!,:lag];
        choice = trial_data[!,:choice];
        phase = trial_data[!,:phase];
        reward_obs = trial_data[!,:reward_obs];

        trial_travel_key = trial_data[1,:travel_key_cond]
        travel_cost_lag = travel_costs_lag[trial_travel_key]
        travel_cost_choice = travel_costs_choice[trial_travel_key]
        trial_start_reward = trial_data[1,:start_reward];

        last_reward_obs = copy(trial_start_reward);

        R_hat_choice = copy(R_hat_start_choice);
        R_hat_lag = copy(R_hat_start_lag);
        harvest_lag_hat = copy(harvest_lag_hat_start);

        # go through each measure in trial_data
        n_presses = size(lag,1);

        first_round_harvest = true;
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(R_hat_lag,harvest_cost_lag, lag_beta)
                    return 1e9
                end

                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost_choice / harvest_lag_hat);
                E_opportunity_cost_harvest = R_hat_choice*harvest_lag_hat;

                theta = [choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # do learnng of the harvest lag...
                    harvest_lag_hat = (1 - lr_harvest_lag_hat)*harvest_lag_hat + lr_harvest_lag_hat*this_lag;

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                            lag_ll = lag_ll + log_lik_lag(R_hat_lag,harvest_cost_lag,lag_beta,this_lag);
                        end
                    else
                        first_round_harvest = false;
                    end

                    this_cost_lag = harvest_cost_lag / this_lag;
                    this_cost_choice = harvest_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(R_hat_lag,travel_cost_lag, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    # don't add first press to lag...
                    # lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost_lag = travel_cost_lag / this_lag;
                    this_cost_choice = travel_cost_choice / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                    first_round_harvest = true;
                end
            else # travel session - just pick a lag...

                if val_fail(R_hat_lag,travel_cost_lag, lag_beta)
                    return 1e9
                end

                this_lag = lag[press_idx];
                if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                    lag_ll = lag_ll + log_lik_lag(R_hat_lag,travel_cost_lag,lag_beta,this_lag);
                end
                this_cost_lag = travel_cost_lag / this_lag;
                this_cost_choice = travel_cost_choice / this_lag;
                this_reward = reward_obs[press_idx];
            end

            ## update R_hat...
            this_move_rr_choice = (this_reward - this_cost_choice) / this_lag;
            this_move_rr_lag = (this_reward - this_cost_lag) / this_lag;

            R_hat_choice = (1 - (1 - lr_R_hat_choice)^this_lag)*this_move_rr_choice + ((1 - lr_R_hat_choice)^this_lag)*R_hat_choice;
            R_hat_lag = (1 - (1 - lr_R_hat_lag)^this_lag)*this_move_rr_lag + ((1 - lr_R_hat_lag)^this_lag)*R_hat_lag; # probably want the same LR for both

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
