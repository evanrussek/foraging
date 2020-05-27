### simulate the original task...
### what are the key task parameters?
# decay rate... .98 # decay only happens on a successful press...
#  n travel steps: 16
# reward noise: 2.4
# start reward noise: 4
# time: 2.34 minutes
# travel success prob: .8
# harvest success prob: .5

function sim_forage_learn(param_dict)

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
    ##### button press costs
    harvest_cost = param_dict["harvest_cost"] * 10.;
    travel_cost_easy = param_dict["travel_cost_easy"] * 10.;
    travel_cost_hard = param_dict["travel_cost_hard"] * 10.;
    r_hat_start_reward_weight = param_dict["r_hat_start_reward_weight"];
    r_hat_start_easy_weight = param_dict["r_hat_start_easy_weight"] * 10;
    harvest_lag_hat_start = param_dict["harvest_lag_hat_start"];
    harvest_bias = param_dict["harvest_bias"];
    choice_beta = param_dict["choice_beta"];
    lag_beta = param_dict["lag_beta"];
    lr_R_hat_pre = param_dict["lr_R_hat_pre"];  #### does this break if lr_R_hat is too large?
    lr_harvest_lag_hat_pre = param_dict["lr_harvest_lag_hat_pre"];
    lr_R_hat = 0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    lr_harvest_lag_hat = 0.5 + 0.5 * erf(lr_harvest_lag_hat_pre / sqrt(2));


    ########################################

    travel_costs = Dict();
    travel_costs["HARD"] = travel_cost_hard;
    travel_costs["EASY"] = travel_cost_easy;

    ###### for picking a lag -- what numbers to evaluate...
    lag_incr = .005;
    lag_range = 0.001:lag_incr:50.;


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
        threshold_hist = Float64[];

        ######### game state parameters
        current_time_unit = 0.;
        steps_to_tree = n_travel_steps;
        this_round_hpress_number = 1;
        steps_to_tree = n_travel_steps;

        trial_start_reward_mean = start_rewards[trial_idx];
        trial_travel_key = travel_keys[trial_idx];
        travel_cost = travel_costs[trial_travel_key];

        # get starting values for parameters that we'll learn from experience
        R_hat_start = r_hat_start_reward_weight*trial_start_reward_mean +
                        r_hat_start_easy_weight*(trial_travel_key == "EASY");

        R_hat = copy(R_hat_start);
        harvest_lag_hat = copy(harvest_lag_hat_start);

        #println("trial number" , trial_idx, "travel key", trial_travel_key, "start_reward", trial_start_reward_mean)

        this_round = 1;

        # begin trial loop
        while current_time_unit < total_time_units

            #println(steps_to_tree)

            # if we're at the tree then harvest
            if steps_to_tree == 0

                if this_round_hpress_number == 1
                    global last_reward_obs
                    global tree_current_reward
                    tree_current_reward = rand(Normal(trial_start_reward_mean,start_reward_noise))
                    last_reward_obs = tree_current_reward;
                end

                # maybe this should iteratively updated / learned from experience...???
                this_lag = sample_lag(lag_beta, R_hat, harvest_cost, lag_range)
                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - vigor_cost(harvest_cost, harvest_lag_hat);
                E_opportunity_cost_harvest = R_hat*harvest_lag_hat;

                threshold = R_hat*harvest_lag_hat + vigor_cost(harvest_cost, harvest_lag_hat);

                # if it's positive you stay, negative you leave
                dv = E_next_reward_harvest - E_opportunity_cost_harvest;

                p_Harvest = 1 ./ (1 + exp(-1*(harvest_bias + choice_beta*dv)));
                choose_Harvest =  rand() < p_Harvest;

                if choose_Harvest
                    push!(choices,1);
                    this_round_hpress_number = this_round_hpress_number+1;
                    this_unit_cost = harvest_cost;
                    this_phase = "HARVEST";
                    # choose a lag...
                    this_lag = sample_lag(lag_beta, R_hat, this_unit_cost, lag_range)

                    this_cost = vigor_cost(this_unit_cost, this_lag);

                    # update the harvest lag hat
                    harvest_lag_hat = lr_harvest_lag_hat*this_lag + (1 - lr_harvest_lag_hat)*harvest_lag_hat;


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
                    this_lag = sample_lag(lag_beta, R_hat, this_unit_cost, lag_range)
                    this_round_hpress_number = 1;
                    steps_to_tree = n_travel_steps; # this will push it into harvest round...
                    #tree_current_reward = trial_start_reward;
                    this_reward = 0.

                    this_cost = vigor_cost(this_unit_cost,this_lag)
                    push!(choices,2);

                    # update the round count...
                    this_round = this_round + 1;

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
                this_lag = sample_lag(lag_beta, R_hat, travel_cost, lag_range)
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
            threshold = R_hat*harvest_lag_hat + vigor_cost(harvest_cost, harvest_lag_hat);

            push!(R_hat_hist, R_hat)
            push!(threshold_hist, threshold)
            push!(harvest_lag_hat_hist, harvest_lag_hat)


            ## update R_hat...
            this_move_rr = (this_reward - this_cost) / this_lag;
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
                threshold = threshold_hist,
                harvest_lag_hat = harvest_lag_hat_hist,
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


function sim_forage2(param_dict)
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
    choice_beta = 1.;#param_dict["choice_beta"];  # choice_beta
    lag_beta = 1.#param_dict["lag_beta"];
    lr_R_hat_pre = param_dict["lr_R_hat_pre"];

    # more extreme scaling...
    lr_R_hat = (.5 + .5*erf(lr_R_hat_pre/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));


    ########################################

    travel_costs = Dict();
    travel_costs["HARD"] = travel_cost_hard;
    travel_costs["EASY"] = travel_cost_easy;

    ###### for picking a lag -- what numbers to evaluate...
    lag_incr = .005;
    lag_range = 0.001:lag_incr:50.;

    R_hat_start = 5.;


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
                this_lag = sample_lag(lag_beta, R_hat, harvest_cost, lag_range)
                #expected_lag =
                # optimal lag...
                current_optimal_lag = sqrt(harvest_cost / R_hat)
                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - vigor_cost(harvest_cost, current_optimal_lag);
                E_opportunity_cost_harvest = R_hat*current_optimal_lag;

                # current optimal lag...
                #threshold = R_hat*harvest_lag_hat + vigor_cost(harvest_cost, current_optimal_lag);

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
                    this_lag = sample_lag(lag_beta, R_hat, this_unit_cost, lag_range)

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
                    this_lag = sample_lag(lag_beta, R_hat, this_unit_cost, lag_range)
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
                this_lag = sample_lag(lag_beta, R_hat, travel_cost, lag_range)
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
            this_move_rr = (this_reward - this_cost) / this_lag;
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



function forage_learn_lik(params, data)

    harvest_bias = params[1];
    travel_cost_hard = params[2]*10; # just doing some scaling...
    r_hat_start_easy_weight  = params[3]*10;
    r_hat_start_reward_weight = params[4];
    choice_beta              = params[5];
    harvest_lag_hat_start    =  params[6] + .001;  # to avoid dividing by 0;
    lr_harvest_lag_hat_pre   = params[7];
    harvest_cost           = params[8]*10;
    lag_beta              = params[9];
    travel_cost_easy       = params[10]*10;
    lr_R_hat_pre = params[11];

    lr_R_hat = 0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    lr_harvest_lag_hat = 0.5 + 0.5 * erf(lr_harvest_lag_hat_pre / sqrt(2));

    # get unique trials...
    travel_costs = Dict();
    travel_costs["HARD"] = travel_cost_hard;
    travel_costs["EASY"] = travel_cost_easy;

    ###### for picking a lag -- what numbers to evaluate......
    lag_incr = .01; # could probably make this coarser as well....
    lag_range = 0.001:lag_incr:25.; # this mght need to be increased for real data?
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;

    lag_ll = 0;
    choice_ll = 0;

    trial_list = unique(data[!,:trial_num])

    ### likelihood function...

    for trial_idx in trial_list

        last_reward_obs = 1000;

        trial_data = data[data[!,:trial_num] .== trial_idx,:];

        trial_travel_key = trial_data[1,:travel_key_cond]
        travel_cost = travel_costs[trial_travel_key]
        trial_start_reward = trial_data[1,:start_reward];

        # get starting values for parameters that we'll learn from experience
        R_hat = r_hat_start_reward_weight*trial_start_reward +
                        r_hat_start_easy_weight*(trial_travel_key == "EASY");


        harvest_lag_hat = copy(harvest_lag_hat_start);

        # go through each measure in trial_data
        n_presses = size(trial_data,1);
        for press_idx in 1:n_presses

            # what if this is what makes it slow?
            press_data = trial_data[press_idx,:];

            if press_data[:phase] == "HARVEST"


                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost / harvest_lag_hat);
                E_opportunity_cost_harvest = R_hat*harvest_lag_hat;

                theta_harvest = harvest_bias + choice_beta*E_next_reward_harvest;
                theta_leave = choice_beta*E_opportunity_cost_harvest;

                theta = [theta_harvest, theta_leave]

                choice_ll = choice_ll + theta[press_data[:choice]] - logsumexp(theta);

                #println("choice: ", press_data[:choice])

                if (press_data[:choice] == 1) # choose harvest

                    vigor_cost_range = harvest_cost ./ lag_range;
                    opportunity_cost_range = R_hat .* lag_range;
                    lag_cost_range = vigor_cost_range + opportunity_cost_range;
                    lag_range_lp_unnorm = lag_beta.* -1 .* lag_cost_range;
                    this_lag = press_data[:lag]
                    harvest_lag_hat = lr_harvest_lag_hat*this_lag + (1 - lr_harvest_lag_hat)*harvest_lag_hat;
                    this_lag_idx = searchsortedfirst(lag_range, this_lag); # is this slow?

                    press_ll = lag_range_lp_unnorm[this_lag_idx] - logsumexp(lag_range_lp_unnorm);
                    lag_ll = lag_ll + press_ll;

                    this_cost = harvest_cost ./ this_lag;
                    this_reward = press_data[:reward_obs];

                    if this_reward > 0
                        last_reward_obs = this_reward;
                    end

                else # choose travel

                    vigor_cost_range = travel_cost ./ lag_range;
                    opportunity_cost_range = R_hat .* lag_range;
                    lag_cost_range = vigor_cost_range + opportunity_cost_range;
                    lag_range_lp_unnorm = lag_beta.* -1 .* lag_cost_range;
                    this_lag = press_data[:lag]
                    this_lag_idx = searchsortedfirst(lag_range, this_lag); # hope this works on real data...
                    press_ll = lag_range_lp_unnorm[this_lag_idx] - logsumexp(lag_range_lp_unnorm);
                    lag_ll = lag_ll + press_ll;

                    this_cost = travel_cost ./ this_lag;
                    this_reward = press_data[:reward_obs];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = 1000;
                end
            else # travel session - just pick a lag...
                vigor_cost_range = travel_cost ./ lag_range;
                opportunity_cost_range = R_hat .* lag_range;
                lag_cost_range = vigor_cost_range + opportunity_cost_range;
                lag_range_lp_unnorm = lag_beta.* -1 .* lag_cost_range;
                this_lag = press_data[:lag]
                this_lag_idx = searchsortedfirst(lag_range, this_lag); # hope this works on real data..., might be slow
                press_ll = lag_range_lp_unnorm[this_lag_idx] - logsumexp(lag_range_lp_unnorm);
                lag_ll = lag_ll + press_ll;

                this_cost = travel_cost ./ this_lag;
                this_reward = press_data[:reward_obs];
            end

            # update R_hat...
            this_move_rr = (this_reward - this_cost) / this_lag;
            R_hat = (1 - (1 - lr_R_hat)^this_lag)*this_move_rr + ((1 - lr_R_hat)^this_lag)*R_hat;

        end
    end
    log_lik = lag_ll + choice_ll;
    return -1*log_lik;
end

function relu_sm(x)
    # makes minimum value .0001
	#p=1/(1+exp(-10*(.0001 -b)))
	#return(p * .0001 + (1-p) * b)
    return log(1. + exp(x)) + .0001;
end

function relu_sm2(x)
    return x*erf(x)
end



function int_exp_lag_cost(R,C,B)

    #analytic integration of unnormalized lag probability
    # R is expected reward rate, C is unit cost, B is lag beta
    # requires SpecialFunctions pkg to use bessell function """

    # this fails if R goes below zero (or C)

    #println(string("R: ", relu_sm(R), " C: ", relu_sm(C), "B: ", relu_sm(B)))

    # B, R and C must be above 0...
    C_new = C;#C < .0001 ? .0001 : C;
    B_new = B# < .0001 ? .0001 : B;
    R_new = R# < .0001 ? .0001 : R;
    #R_new = R_new > 1000 ? 1000 : R_new;
    #C_new =  C_new > 1000 ? 1000 : C_new;


    #B_new = softplus(B);
    #R_new = softplus(R);

    return 2*sqrt(C_new/R_new)*besselk(1,2*B_new*sqrt(R_new*C_new))

end


# un-normalized lag probability
function u_lag_prob(R,C,B,this_lag)
    return -B.*(C./this_lag + R.*this_lag)
end

# normalization factor
function numerical_log_norm(R,C,B)
    incr = .01;
    lag = .1e-4:incr:80;
    return logsumexp(u_lag_prob(R,C,B,lag)) + log(incr);
end

function val_fail(R,C,B)
    return (R < 1e-20 || C < 1e-20 || B < 1e-20 || 2*B*sqrt(R*C) > 680)
end


function log_norm_bit(R,C,B)
    #println(R,C,B)
    #if (R < 1e-20 || C < 1e-20 || B < 1e-20 || 2*B*sqrt(R*C) > 680)
    #    return 1e7#Inf;#numerical_log_norm(R,C,B)
    #else
        return log(int_exp_lag_cost(R,C,B))
    #end
end

function log_lik_lag(R,C,B,this_lag)
    return u_lag_prob(R,C,B,this_lag) - log_norm_bit(R,C,B);
end

# this seems to be just wrong...
function log_lag_den_approx(R,C,B)
    # for log bessell:
    # https://stackoverflow.com/questions/32484696/natural-logarithm-of-bessel-function-overflow

    log_z = log(2) + log(B) + .5*log(R) + .5*log(C);
    log_bessell = .5*log(pi) + .5*log(2) - 1 - log_z;
    return log(2) + .5*log(C) - .5*log(R) + log_bessell;

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



function get_trial_exit_threshs(trial_data)

    if size(trial_data,1) > 1
        exit_tbl = by(trial_data, :round) do df
            non_missing_harvest = @where(df, .&(:phase .== "HARVEST", :reward_obs .> 0));
            if size(non_missing_harvest,1) > 0
                last_reward = non_missing_harvest.reward_true[end];
                df.reward_obs[ismissing.(df.reward_obs)] .= 0;

                travel_df = @where(df, :phase .== "TRAVEL");

                ### travel time?

                this_df = DataFrame(
                    subjectID = df.subjectID[1],
                    last_reward = last_reward,
                    start_reward = df.start_reward[1],
                    travel_key_cond = df.travel_key_cond[1],
                    );
            else
                this_df = DataFrame(
                    subjectID = [],
                    last_reward = [],
                    start_reward = [],
                    travel_key_cond = []);
            end

            return this_df
        end # this ends the do
    else
        exit_tbl = DataFrame(
        subjectID = [],
        last_reward = [],
        round = [],
        start_reward = [],
        travel_key_cond = []);
    end
    return exit_tbl
end

function make_exit_plot(sim_df)

    trial_exit_data = make_subj_trial_exit(sim_df);

    p = plot(trial_exit_data, x = :start_reward_cat, y = :exit_thresh,
    color = :travel_key_cond, Geom.line(), Geom.point(),
    Scale.x_discrete(levels = sort(unique(trial_exit_data[!,:start_reward_cat]))),
    Scale.color_discrete(levels = ["HARD", "EASY"]),

    )
    return p;
end

function make_subj_trial_exit(sim_df)
    round_exit_data = by(sim_df, [:trial_num], df -> get_trial_exit_threshs(df));

    trial_exit_data = by(round_exit_data, [:trial_num],
                start_reward = :start_reward => first,
                travel_key_cond = :travel_key_cond => first,
                exit_thresh = :last_reward  => mean
        )
    trial_exit_data[!,:start_reward_cat] = CategoricalArray(trial_exit_data[!,:start_reward])
    return trial_exit_data
end

function make_group_trial_exit(pdata)
    round_exit_data = by(pdata, [:sub, :trial_num], df -> get_trial_exit_threshs(df));

    trial_exit_data = by(round_exit_data, [:sub, :trial_num],
                start_reward = :start_reward => first,
                travel_key_cond = :travel_key_cond => first,
                exit_thresh = :last_reward  => mean
        );

    group_trial_exit = by(trial_exit_data, [:start_reward, :travel_key_cond],
        exit_thresh = :exit_thresh => mean,
        exit_sd = :exit_thresh => std)

    group_trial_exit[!,:upper] = group_trial_exit[!,:exit_thresh] + group_trial_exit[!,:exit_sd] ./ sqrt(n_subj);
    group_trial_exit[!,:lower] = group_trial_exit[!,:exit_thresh] - group_trial_exit[!,:exit_sd] ./ sqrt(n_subj);
    group_trial_exit[!,:start_reward_cat] = CategoricalArray(group_trial_exit[!,:start_reward]);
    return group_trial_exit
end

function make_group_exit_plot(pdata; title = "")
    group_trial_exit = make_group_trial_exit(pdata);
    return plot(group_trial_exit, x = :start_reward_cat, y = :exit_thresh, ymin = :lower, ymax = :upper,
        color = :travel_key_cond, Geom.line(), Geom.point(),
        Geom.errorbar(),
        Scale.x_discrete(levels = sort(unique(group_trial_exit[!,:start_reward_cat]))),
        Scale.color_discrete(levels = ["HARD", "EASY"]),
        Guide.title(title),
        Guide.xlabel("Tree First Reward", orientation = :horizontal),
        Guide.ylabel("Last Reward Before Exit"),
        Guide.colorkey(title = "Travel Key")
        )
end

function make_group_lag_df(pdata)
    new_pdata = copy(pdata);
    new_pdata = by(new_pdata, [:sub, :trial_num, :round, :phase],
            df -> df[2:end, [:lag, :lag_scale, :choice, :start_reward, :reward_obs,
                        :travel_key_cond, :subjectID, :reward_true, :upper_lag_thresh, :lower_lag_thresh,
                        :remove]]);
    new_pdata = @where(new_pdata, :lag .> :lower_lag_thresh,  :lag .< :upper_lag_thresh);
    trial_lag_df = by(new_pdata, [:sub,:trial_num, :phase],
        df -> DataFrame(start_reward = first(df.start_reward),
                travel_key_cond = first(df.travel_key_cond), lag = median(df.lag_scale),
                log_lag = mean(log.(df.lag_scale)),subjectID = df.subjectID[1]
            )
        )
    group_lag_df = by(trial_lag_df, [:start_reward, :travel_key_cond, :phase],
                        :log_lag => mean,
                        :log_lag => std)
    group_lag_df[!,:upper] = group_lag_df[!,:log_lag_mean] + group_lag_df[!,:log_lag_std] ./ sqrt(n_subj);
    group_lag_df[!,:lower] = group_lag_df[!,:log_lag_mean] - group_lag_df[!,:log_lag_std] ./ sqrt(n_subj);
    group_lag_df[!,:start_reward_cat] = CategoricalArray(group_lag_df[!,:start_reward]);
    return group_lag_df
end

function make_group_lag_plot(pdata; title = "")
    group_lag_df = make_group_lag_df(pdata);
    return plot(group_lag_df, x = :start_reward_cat, y = :log_lag_mean, color = :travel_key_cond,
        ymax = :upper, ymin = :lower, xgroup = :phase,
        Geom.subplot_grid(Geom.line(), Geom.point(), Geom.errorbar()),
        Scale.x_discrete(levels = sort(unique(group_lag_df[!,:start_reward_cat]))),
        Scale.color_discrete(levels = ["HARD", "EASY"]),
        Guide.ylabel("Log Lag"),
        Guide.xlabel("Tree First Reward by Part", orientation = :horizontal),
        Guide.colorkey(title = "Travel Key"),
        Guide.title(title))
end

function transform_lr(lr_orig)
    return .5 + .5*erf(lr_orig/sqrt(2))
end

# make a lag plot

function make_subj_lag_table(sim_df)
    trial_lag_df = by(sim_df,
        [:trial_num, :phase],
        df -> DataFrame(
            start_reward = first(df.start_reward),
            travel_key_cond = first(df.travel_key_cond),
            lag = median(df.lag_scale),
            log_lag = mean(log.(df.lag_scale)),
            subjectID = df.subjectID[1]
            )
    )
    trial_lag_df[!,:start_reward_cat] = CategoricalArray(trial_lag_df[!,:start_reward]);
    return trial_lag_df;
end


function make_lag_plot(sim_df)

    trial_lag_df = make_subj_lag_table(sim_df)

    p = plot(trial_lag_df, x = :start_reward_cat, y = :log_lag, color = :travel_key_cond,
        xgroup = :phase, Geom.subplot_grid(Geom.line(), Geom.point()),
        Scale.x_discrete(levels = sort(unique(trial_lag_df[!,:start_reward_cat]))),
        Scale.color_discrete(levels = ["HARD", "EASY"])
        )
    return p;
end

function generate_start_vals(cost_fun, nparams)
    start_x = 4*randn(nparams)
    cost_fun(start_x) > 1e8 || return start_x
    generate_start_vals(cost_fun, nparams)
end

function make_recov_df(a,param_names,param_dict, cost_fun)
    p_hat = a.minimizer;
    p_nll = a.minimum;

    p_hat_dict = Dict()

    for j in 1:length(p_hat)
        p_hat_dict[param_names[j]] = p_hat[j]
    end
    original_val = zeros(length(param_dict));
    recovered_val = zeros(length(param_dict));
    param_name = [];
    i = 1;
    for (j,v) in param_dict
        #println(j)
        push!(param_name,j);
        original_val[i] = v;
        recovered_val[i] = p_hat_dict[j]
        i = i+1;
    end

    orig_nll = cost_fun(original_val)

    rec_df = DataFrame(pname = param_name, original_val = original_val,
        recovered_val = recovered_val);
    rec_df[!,:orig_nll] .= orig_nll;
    rec_df[!,:rec_nll] .= p_nll;

    return rec_df
end

function prep_subj_data(data_in)

    #s_data = DataFrame();

    s_data = DataFrame(data_in);#copy(data_in);
    s_data[!,:lag_scale] = s_data[!,:lag];
    s_data[!,:lag] = s_data[!,:lag_scale] ./ 100;

    s_data[ismissing.(s_data[!,:reward_obs]),:reward_obs] .= 0;
    s_data[ismissing.(s_data[!,:exit]),:exit] .= 1;
    s_data[!,:choice] = s_data[!,:exit] .+ 1;
    s_data[!,:choice] = convert(Array{Int,1}, s_data[!,:choice]);

    sub_df = s_data[!, [:round, :trial_num, :lag,
    :lag_scale, :choice, :phase, :start_reward,
     :reward_obs, :travel_key_cond, :subjectID, :reward_true]];

    # pass these to the likelihood function...
    upper_lag_thresh = median(s_data[!,:lag]) + 4*mad(s_data[!,:lag])
    lower_lag_thresh = median(s_data[!,:lag]) - 4*mad(s_data[!,:lag])
    sub_df[!,:upper_lag_thresh] .= upper_lag_thresh;
    sub_df[!,:lower_lag_thresh] .= lower_lag_thresh;
    return sub_df
end
