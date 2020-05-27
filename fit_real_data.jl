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

# read in the data...
data = CSV.read("/Users/evanrussek/forage_jsp/analysis/data/run5_data.csv");

# check the travel keys are correctly labeled...
travel_keys = unique(data.travel_key)
travel_key_easy = travel_keys[1]
travel_key_hard = travel_keys[2]
travel_keys_he = [travel_key_hard travel_key_easy];
cdata, n_subj = clean_group_data(data,travel_keys_he);

show(first(cdata,1), allcols = true)

## data cols needed
# sub, trial_num, lag, choice, phase, start_reward, reward_obs,

# do you need to scale the lags in some way????????


s_data = @where(cdata, :s_num .== 1);
s_data

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
cdata[!,:sub] = cdata[!,:s_num];
pdata = by(cdata, :sub, df -> prep_subj_data(df));


param_names = ["travel_cost_easy", "travel_cost_hard", "lr_R_hat_pre", "harvest_cost"];
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


function fit_subject(s_data)
    param_names = ["travel_cost_easy", "travel_cost_hard", "lr_R_hat_pre", "harvest_cost"];
    start_p = generate_start_vals((x) -> forage_learn_lik3(x,s_data; lag_threshs = [s_data[1,:lower_lag_thresh],s_data[1,:upper_lag_thresh]],
        which_data = "both"), length(param_names));
    a = optimize((x) -> forage_learn_lik3(x,s_data; lag_threshs = lag_threshs, which_data = "both"),
            start_p, LBFGS(),Optim.Options(allow_f_increases=true, iterations = 1000, show_trace = false),
            autodiff=:forward)
    return (a.minimum, a.minimizer)
end

# fit all the subjects...
for s_idx in 1:n_subj
    print(s_idx)
    if !(s_idx in bad_subjs)
        (min_nll[s_idx], p_hat[s_idx,:]) = fit_subject(@where(pdata, :sub .== s_idx));
    end
end
# create a dict of params...

fit_dict = Dict();
for p_idx in 1:n_param
    fit_dict[param_names[p_idx]] = p_hat[:,p_idx];
end
fit_dict["nll"] = min_nll;
fit_df = DataFrame(fit_dict);
fit_df[!,:sub] = 1:n_subj;
first(fit_df,6)
for b in bad_subjs
    fit_df = fit_df[fit_df[!,:sub] .!== b,:]
end

fit_df_long = stack(fit_df, [:lr_R_hat_pre, :harvest_cost, :travel_cost_hard, :travel_cost_easy], variable_name= :param_names);
fit_df_long.param_names = string.(fit_df_long.param_names);

plot(@where(fit_df_long, :param_names .!== "lr_R_hat_pre"), x = :value, ygroup = :param_names,
    Geom.subplot_grid( Geom.histogram))

plot(@where(fit_df_long, :param_names .!== "lr_R_hat_pre"), x = :value, color = :param_names,
    Geom.density)

plot(fit_df, x = :lr_R_hat_pre, Geom.histogram, Geom.density, Coord.Cartesian(xmin = -60, xmax = 10))

###
pdata_wp = join(pdata, fit_df, on = :sub, kind = :left);
pdata_wp = @where(pdata_wp, :remove .== false);

high_lr_data = @where(pdata_wp, :lr_R_hat_pre .> -20);
low_lr_data = @where(pdata_wp, :lr_R_hat_pre .< -20);
n_high_subj = length(unique(high_lr_data.sub))
n_low_subj = length(unique(low_lr_data.sub))
learn_title = string("Learners Data, N = ", n_high_subj);
nonlearn_title = string("Non-learners Data, N = ", n_low_subj);

hlr_exit = make_group_exit_plot(high_lr_data; title= learn_title);
llr_exit = make_group_exit_plot(low_lr_data; title = nonlearn_title);
lsplit_exit_data = hstack(llr_exit, hlr_exit);

hlr_lag = make_group_lag_plot(high_lr_data; title=learn_title);
llr_lag = make_group_lag_plot(low_lr_data; title = nonlearn_title);
lsplit_lag_data = hstack(llr_lag, hlr_lag);

function make_subp_dict(s_idx)
    sub_params = @where(fit_df_long, :sub .== s_idx);
    p_hat_dict = Dict();
    for j = 1:size(median_params,1)
        p_hat_dict[median_params[j,:param_names]] = sub_params[j,:value]
    end
    return p_hat_dict
end


# re-simulate each subject....
sim_data = DataFrame();
for s_idx in unique(pdata_wp[!,:sub])
    print(s_idx)
    p_hat_dict = make_subp_dict(s_idx)
    sim_df = sim_forage2(p_hat_dict);
    sim_df[!,:sub] .= s_idx;
    sim_df[!,:subjectID] .= s_idx;
    sim_data = [sim_data; sim_df];
end
sim_data[!,:upper_lag_thresh] .= Inf;
sim_data[!,:lower_lag_thresh] .= -Inf;
sim_data[!,:remove] .= false;

exit_sim_plot = make_group_exit_plot(sim_data, title = "Simualtion")
exit_data_plot = make_group_exit_plot(pdata, title = "Data (N = 50)");
hstack(exit_sim_plot, exit_data_plot)


lag_sim_plot = make_group_lag_plot(sim_data; title = "Simulation");
lag_data_plot = make_group_lag_plot(pdata; title = "Data");
set_default_plot_size(18cm, 10cm)
hstack(lag_sim_plot, lag_data_plot)

simdata_wp = join(sim_data, fit_df, on = :sub, kind = :left);
high_lr_sim = @where(simdata_wp, :lr_R_hat_pre .> -20);
low_lr_sim = @where(simdata_wp, :lr_R_hat_pre .< -20);

learn_title = string("Learners Sim, N = ", n_high_subj);
nonlearn_title = string("Non-learners Sim, N = ", n_low_subj);
hlr_exit = make_group_exit_plot(high_lr_sim; title= learn_title);
llr_exit = make_group_exit_plot(low_lr_sim; title = nonlearn_title);

lsplit_exit_sim = hstack(llr_exit, hlr_exit);

hlr_lag = make_group_lag_plot(high_lr_sim; title= learn_title);
llr_lag = make_group_lag_plot(low_lr_sim; title = nonlearn_title);
lsplit_lag_sim = hstack(llr_lag, hlr_lag);

lsplit_exit_data
lsplit_exit_sim

lsplit_lag_data





























# simulate the median paramater values...
median_params = by(fit_df_long, :param_names, :value => median)
p_hat_dict = Dict();
for j = 1:size(median_params,1)
    p_hat_dict[median_params[j,:param_names]] = median_params[j,:value_median]
end

sim_data = DataFrame();
# simulate X datasets
for s_idx in 31:50
    print(s_idx)
    sim_df = sim_forage2(p_hat_dict);
    sim_df[!,:sub] .= s_idx;
    sim_df[!,:subjectID] .= s_idx;
    sim_data = [sim_data; sim_df];
end
sim_data[!,:upper_lag_thresh] .= Inf;
sim_data[!,:lower_lag_thresh] .= -Inf;
sim_data[!,:remove] .= false;


# simulate multiple data-sets?
exit_sim_plot = make_group_exit_plot(sim_data, title = "Simualtion (Median Parameters; N= 50)");
exit_data_plot = make_group_exit_plot(pdata, title = "Data (N = 50)");
hstack(exit_sim_plot, exit_data_plot)


lag_sim_plot = make_group_lag_plot(sim_data; title = "Simulation (Median Parameters)");
lag_data_plot = make_group_lag_plot(pdata; title = "Data");
set_default_plot_size(18cm, 10cm)
hstack(lag_sim_plot, lag_data_plot)




# add error bars... ? ...


# make the group lag plot from pdata...











p_hat = a.minimizer;

p_hat_dict = Dict()
for j in 1:length(param_names)
    p_hat_dict[param_names[j]] = p_hat[j]
end
sim_df_rec = sim_forage2(p_hat_dict);

make_exit_plot(sim_df_rec)
make_exit_plot(s_data)
make_lag_plot(sim_df_rec)
make_lag_plot(s_data)


param_names

p_hat

p_hat_dict







### no learning here...



## look at the results...
#
#
#








function inrange(x, lower,upper)
    return (x >= lower) && (x <= upper)
end


function forage_learn_lik3(param_vals, data; lag_threshs = [-Inf, Inf], which_data = "both")


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

        R_hat = convert(typeof(lag_beta),5.); # maybe you need to fit this...

        # go through each measure in trial_data
        n_presses = size(lag,1);

        first_round_harvest = true;
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(R_hat,harvest_cost, lag_beta)
                    return 1e9
                end


                current_optimal_lag = sqrt(harvest_cost / R_hat)
                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost / current_optimal_lag);
                E_opportunity_cost_harvest = R_hat*current_optimal_lag;

                theta = [choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                            lag_ll = lag_ll + log_lik_lag(R_hat,harvest_cost,lag_beta,this_lag);
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
                    if val_fail(R_hat,travel_cost, lag_beta)
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

                if val_fail(R_hat,travel_cost, lag_beta)
                    return 1e9
                end
                this_lag = lag[press_idx];
                if inrange(this_lag, lag_threshs[1],lag_threshs[2])
                    lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);
                end
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

#
