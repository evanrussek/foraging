
function sim_forage_rs(param_dict)
    ## simulate with only 6 parameters...
    time_scale = 1. / 100;
    n_travel_steps = 16;
    nmin = 2.33;
    total_time_units = nmin*60*1000*time_scale;
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;
    reward_noise = 2.4;
    start_reward_noise = 4;

    simID = "sim1";

    ############# game parameters that vary over trials
    start_rewards = [60., 90., 120., 60., 90., 120.];
    travel_keys = ["HARD","HARD","HARD","EASY","EASY","EASY"];

    ##################### agent parameters

    ##### 6 parameters...
    harvest_cost = param_dict["harvest_cost"]*10;
    travel_cost_easy = param_dict["travel_cost_easy"]*10;
    travel_cost_hard = param_dict["travel_cost_hard"]*10;
    choice_beta = param_dict["choice_beta"];  # choice_beta
    lag_beta = param_dict["lag_beta"]; # lag beta...
    lr_R_hat_pre = param_dict["lr_R_hat_pre"];
    reward_scale = 1.#param_dict["reward_scale"];
    R_hat_start = param_dict["R_hat_start"]*10;
    harvest_bias = param_dict["harvest_bias"]*10
    lag_rr_sen = 1.;#param_dict["lag_rr_sen"];

    #choice_beta = param_dict["choice_beta"];

    # more extreme scaling...
    lr_R_hat = (.5 + .5*erf(lr_R_hat_pre/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));


    ########################################

    travel_costs = Dict();
    travel_costs["HARD"] = travel_cost_hard;
    travel_costs["EASY"] = travel_cost_easy;

    ###### for picking a lag -- what numbers to evaluate...
    lag_incr = .005;
    lag_range = 0.001:lag_incr:50.;

    #R_hat_start = copy()#reward_scale*50.;

    # to hold structure DF
    sim_df = DataFrame()

    #for r_idx in 1:4
    for trial_idx in 1:6

        ############### structures to record data
        lags = Float64[];
        choices = Int64[];
        phase = String[];
        reward_obs = Float64[];
        reward_true = Float64[];
        time_units = Float64[];
        R_hat_hist = Float64[];
        round_num = Int64[];
        harvest_lag_hat_hist = Float64[];

        ######### game state parameters
        current_time_unit = 0.;
        steps_to_tree = n_travel_steps;
        this_round_hpress_number = 1;
        steps_to_tree = n_travel_steps;

        trial_start_reward_mean = start_rewards[trial_idx];
        trial_travel_key = travel_keys[trial_idx];
        travel_cost = travel_costs[trial_travel_key];

        R_hat = copy(R_hat_start);  #### update this...

        this_round = 1;
        last_reward_obs = trial_start_reward_mean;
        tree_current_reward = rand(Normal(trial_start_reward_mean,start_reward_noise));

        # begin trial loop
        while current_time_unit < total_time_units

            #println(steps_to_tree)

            # if we're at the tree then harvest
            if steps_to_tree == 0

                #
                this_lag = sample_lag(lag_beta, lag_rr_sen*R_hat, harvest_cost, lag_range)
                #expected_lag =
                # optimal lag...
                # this might mess things up...
                current_optimal_lag = sqrt(harvest_cost / lag_rr_sen*R_hat)
                E_next_reward_harvest = reward_scale*last_reward_obs*decay*harvest_success_prob - vigor_cost(harvest_cost, current_optimal_lag);
                E_opportunity_cost_harvest = R_hat*current_optimal_lag;

                # current optimal lag...
                #threshold = R_hat*harvest_lag_hat + vigor_cost(harvest_cost, current_optimal_lag);

                # if it's positive you stay, negative you leave
                dv = harvest_bias + E_next_reward_harvest - E_opportunity_cost_harvest;

                p_Harvest = 1 ./ (1 + exp(-1*(choice_beta*dv)));
                choose_Harvest =  rand() < p_Harvest;

                if choose_Harvest
                    push!(choices,1);
                    this_round_hpress_number = this_round_hpress_number+1;
                    this_unit_cost = harvest_cost;
                    this_phase = "HARVEST";
                    # choose a lag...
                    this_lag = sample_lag(lag_beta, lag_rr_sen*R_hat, this_unit_cost, lag_range)

                    this_cost = vigor_cost(this_unit_cost, this_lag);

                    # check if the move is successful
                    if rand() < harvest_success_prob
                        # sample reward
                        this_reward = rand(Normal(tree_current_reward, reward_noise));
                        last_reward_obs = this_reward;

                        # update trees current reward if move is successful...
                        tree_current_reward = tree_current_reward*decay;
                    else
                        this_reward = 0.; # nothing updates...
                    end
                    # reward observed

                else # chose travel...
                    this_unit_cost = travel_cost;
                    this_phase = "HARVEST";
                    # make a lag here, but overall, don't add this lag to the likelihood
                    this_lag = sample_lag(lag_beta, lag_rr_sen*R_hat, this_unit_cost, lag_range)
                    steps_to_tree = n_travel_steps; # this will push it into harvest round...
                    #tree_current_reward = trial_start_reward;
                    this_reward = 0.

                    this_cost = vigor_cost(this_unit_cost,this_lag)
                    push!(choices,2);

                    # update the round count...
                    this_round = this_round + 1;

                    # reset tree current reward and last_reward_obs
                    last_reward_obs = trial_start_reward_mean;
                    tree_current_reward = rand(Normal(trial_start_reward_mean,start_reward_noise));
               # break
                end

                push!(lags,this_lag)
                #push!(choices,1)
                push!(phase, this_phase)
                push!(reward_obs, this_reward)
                push!(time_units, current_time_unit)
                push!(round_num, this_round)
                push!(reward_true, tree_current_reward)

                # update the clock...
                current_time_unit = current_time_unit + this_lag;

            else # if we're traveling

                this_reward = 0.;
                this_unit_cost = travel_cost; # that's not the cost...
                this_lag = sample_lag(lag_beta, lag_rr_sen*R_hat, travel_cost, lag_range)
                this_cost = vigor_cost(this_unit_cost, this_lag);

                #### record data from the key press ##  also there's this cost
                push!(lags,this_lag)
                push!(choices,2)
                push!(phase, "TRAVEL")
                push!(reward_obs, this_reward)
                push!(time_units, current_time_unit)
                push!(round_num, this_round)
                push!(reward_true, 0.);


                #### update agent position
                if rand() < travel_success_prob
                    steps_to_tree = steps_to_tree - 1;
                end

                current_time_unit = current_time_unit + this_lag;
            end
            # store the R_hat that led to this decision

            push!(R_hat_hist, R_hat)
            #push!(threshold_hist, threshold)

            ## update R_hat...
            this_move_rr = (reward_scale*this_reward - this_cost) / this_lag;
            R_hat = (1 - (1 - lr_R_hat)^this_lag)*this_move_rr + ((1 - lr_R_hat)^this_lag)*R_hat;
            # store R_hat...

        end

        # put the trial_results in a dataframe # also put round in here...
        trial_df = DataFrame(lag = lags,
                choice = choices,
                phase = phase,
                reward_obs = reward_obs,
                time = time_units,
                R_hat = R_hat_hist,
                round = round_num,
                reward_true = reward_true);

        trial_df[!,:trial_num] .= trial_idx;
        trial_df[!,:start_reward] .= trial_start_reward_mean;
        trial_df[!,:travel_key_cond] .= trial_travel_key;
        trial_df[!,:subjectID] .= simID;
        trial_df[!,:lag_scale] = trial_df[!,:lag] ./ time_scale;

        sim_df = [sim_df; trial_df]
        #global sim_df = [sim_df; trial_df]

    end
    return sim_df
end




function sim_forage_no_learn(param_dict)
    ## simulate with only 6 parameters...
    time_scale = 1. / 100;
    n_travel_steps = 16;
    nmin = 2.33;
    total_time_units = nmin*60*1000*time_scale;
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;
    reward_noise = 2.4;
    start_reward_noise = 4;

    simID = "sim1";

    ############# game parameters that vary over trials
    start_rewards = [60., 90., 120., 60., 90., 120.];
    travel_keys = ["HARD","HARD","HARD","EASY","EASY","EASY"];

    ##################### agent parameters

    ##### 6 parameters...

    # form reward rate used for choices from trial params...
    choice_r_hard_low_beta = param_dict["choice_r_hard_low_beta"];
    choice_r_easy_beta = param_dict["choice_r_easy_beta"];
    choice_r_reward_beta = param_dict["choice_r_reward_beta"];

    # form reward rate used for lags from trial settings
    lag_r_hard_low_beta = param_dict["lag_r_hard_low_beta"];
    lag_r_easy_beta = param_dict["lag_r_easy_beta"];
    lag_r_reward_beta = param_dict["lag_r_reward_beta"];

    # costs (relevant only for lags...) # why are these multiplied by 10?
    harvest_cost = param_dict["harvest_cost"]*10;
    travel_cost_easy = param_dict["travel_cost_easy"]*10;
    travel_cost_hard = param_dict["travel_cost_hard"]*10;

    # noise on choice response...
    choice_beta = param_dict["choice_beta"];  # choice_beta
    # noise on lag response...
    lag_beta = param_dict["lag_beta"];



    lr_R_hat_pre = -10000000;#param_dict["lr_R_hat_pre"];
    reward_scale = 1.#param_dict["reward_scale"];

    # more extreme scaling...
    lr_R_hat = 0;#(.5 + .5*erf(lr_R_hat_pre/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));


    ########################################

    travel_costs = Dict();
    travel_costs["HARD"] = travel_cost_hard;
    travel_costs["EASY"] = travel_cost_easy;

    ###### for picking a lag -- what numbers to evaluate...
    lag_incr = .005;
    lag_range = 0.001:lag_incr:50.;

    #R_hat_start = copy()#reward_scale*50.;

    # to hold structure DF
    sim_df = DataFrame()

    #for r_idx in 1:4
    for trial_idx in 1:6

        ############### structures to record data
        lags = Float64[];
        choices = Int64[];
        phase = String[];
        reward_obs = Float64[];
        reward_true = Float64[];
        time_units = Float64[];
        R_hat_hist = Float64[];
        round_num = Int64[];
        harvest_lag_hat_hist = Float64[];

        ######### game state parameters
        current_time_unit = 0.;
        steps_to_tree = n_travel_steps;
        this_round_hpress_number = 1;
        steps_to_tree = n_travel_steps;

        trial_start_reward_mean = start_rewards[trial_idx];
        trial_travel_key = travel_keys[trial_idx];
        travel_cost = travel_costs[trial_travel_key];

        R_hat_choice = choice_r_hard_low_beta + ((trial_start_reward_mean - 60)/60)*choice_r_reward_beta + choice_r_easy_beta*(trial_travel_key == "EASY");
        R_hat_lag = lag_r_hard_low_beta + ((trial_start_reward_mean - 60)/60)*lag_r_reward_beta + lag_r_easy_beta*(trial_travel_key == "EASY");

        this_round = 1;
        last_reward_obs = trial_start_reward_mean;
        tree_current_reward = rand(Normal(trial_start_reward_mean,start_reward_noise));

        # begin trial loop
        while current_time_unit < total_time_units

            #println(steps_to_tree)

            # if we're at the tree then harvest
            if steps_to_tree == 0

                current_optimal_lag = sqrt(harvest_cost / R_hat_lag)
                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - vigor_cost(harvest_cost, current_optimal_lag);
                E_opportunity_cost_harvest = R_hat_choice*current_optimal_lag;

                # if it's positive you stay, negative you leave
                dv = E_next_reward_harvest - E_opportunity_cost_harvest;

                p_Harvest = 1 ./ (1 + exp(-1*(choice_beta*dv)));
                choose_Harvest =  rand() < p_Harvest;

                if choose_Harvest
                    push!(choices,1);
                    this_round_hpress_number = this_round_hpress_number+1;
                    this_unit_cost = harvest_cost;
                    this_phase = "HARVEST";
                    # choose a lag...
                    this_lag = sample_lag(lag_beta, R_hat_lag, this_unit_cost, lag_range)
                    this_cost = vigor_cost(this_unit_cost, this_lag);

                    # check if the move is successful
                    if rand() < harvest_success_prob
                        # sample reward
                        this_reward = rand(Normal(tree_current_reward, reward_noise));
                        last_reward_obs = this_reward;

                        # update trees current reward if move is successful...
                        tree_current_reward = tree_current_reward*decay;
                    else
                        this_reward = 0.; # nothing updates...
                    end
                    # reward observed

                else # chose travel...
                    this_unit_cost = travel_cost;
                    this_phase = "HARVEST";
                    # make a lag here, but overall, don't add this lag to the likelihood
                    this_lag = sample_lag(lag_beta, R_hat_lag, this_unit_cost, lag_range)
                    steps_to_tree = n_travel_steps; # this will push it into harvest round...
                    #tree_current_reward = trial_start_reward;
                    this_reward = 0.

                    this_cost = vigor_cost(this_unit_cost,this_lag)
                    push!(choices,2);

                    # update the round count...
                    this_round = this_round + 1;

                    # reset tree current reward and last_reward_obs
                    last_reward_obs = trial_start_reward_mean;
                    tree_current_reward = rand(Normal(trial_start_reward_mean,start_reward_noise));
               # break
                end

                push!(lags,this_lag)
                #push!(choices,1)
                push!(phase, this_phase)
                push!(reward_obs, this_reward)
                push!(time_units, current_time_unit)
                push!(round_num, this_round)
                push!(reward_true, tree_current_reward)

                # update the clock...
                current_time_unit = current_time_unit + this_lag;

            else # if we're traveling

                this_reward = 0.;
                this_unit_cost = travel_cost; # that's not the cost...
                this_lag = sample_lag(lag_beta, R_hat_lag, travel_cost, lag_range)
                this_cost = vigor_cost(this_unit_cost, this_lag);

                #### record data from the key press ##  also there's this cost
                push!(lags,this_lag)
                push!(choices,2)
                push!(phase, "TRAVEL")
                push!(reward_obs, this_reward)
                push!(time_units, current_time_unit)
                push!(round_num, this_round)
                push!(reward_true, 0.);


                #### update agent position
                if rand() < travel_success_prob
                    steps_to_tree = steps_to_tree - 1;
                end

                current_time_unit = current_time_unit + this_lag;
            end
            # store the R_hat that led to this decision

            #push!(R_hat_hist, R_hat)
            #push!(threshold_hist, threshold)

            ## update R_hat...
            #this_move_rr = (this_reward - this_cost) / this_lag;
            #R_hat = (1 - (1 - lr_R_hat)^this_lag)*this_move_rr + ((1 - lr_R_hat)^this_lag)*R_hat;
            # store R_hat...

        end

        # put the trial_results in a dataframe # also put round in here...
        trial_df = DataFrame(lag = lags,
                choice = choices,
                phase = phase,
                reward_obs = reward_obs,
                time = time_units,
                round = round_num,
                reward_true = reward_true);

        trial_df[!,:trial_num] .= trial_idx;
        trial_df[!,:start_reward] .= trial_start_reward_mean;
        trial_df[!,:travel_key_cond] .= trial_travel_key;
        trial_df[!,:subjectID] .= simID;
        trial_df[!,:lag_scale] = trial_df[!,:lag] ./ time_scale;

        sim_df = [sim_df; trial_df]
        #global sim_df = [sim_df; trial_df]

    end
    return sim_df
end
