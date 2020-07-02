

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

function get_trial_exit_threshs(trial_data)

    if size(trial_data,1) > 1
        exit_tbl = by(trial_data, :round) do df
            non_missing_harvest = @where(df, .&(:phase .== "HARVEST", :reward_obs .> 0));
            if size(non_missing_harvest,1) > 0
                last_reward = non_missing_harvest.reward_true[end];
                last_reward_time = non_missing_harvest.trial_time_elapsed[end];

                df.reward_obs[ismissing.(df.reward_obs)] .= 0;

                travel_df = @where(df, :phase .== "TRAVEL");

                ### travel time?

                this_df = DataFrame(
                    subjectID = df.subjectID[1],
                    last_reward = last_reward,
                    start_reward = df.start_reward[1],
                    last_reward_time = last_reward_time,
                    travel_key_cond = df.travel_key_cond[1],
                    );
            else
                this_df = DataFrame(
                    subjectID = [],
                    last_reward = [],
                    last_reward_time = [],
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

function make_group_exit_plot(pdata; title = "", bounds = [0, 100])
    group_trial_exit = make_group_trial_exit(pdata);
    return plot(group_trial_exit, x = :start_reward_cat, y = :exit_thresh, ymin = :lower, ymax = :upper,
        color = :travel_key_cond, Geom.line(), Geom.point(),
        Geom.errorbar(),
        Scale.x_discrete(levels = sort(unique(group_trial_exit[!,:start_reward_cat]))),
        Scale.color_discrete(levels = ["HARD", "EASY"]),
        Guide.title(title),
        Guide.xlabel("Tree First Reward", orientation = :horizontal),
        Guide.ylabel("Last Reward Before Exit"),
        Guide.colorkey(title = "Travel Key"),
        Coord.Cartesian(ymin = bounds[1], ymax = bounds[2])
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

function make_group_lag_plot(pdata; title = "", side = "left")
    group_lag_df = make_group_lag_df(pdata);

    if side == "right"
        plot(group_lag_df, x = :start_reward_cat, y = :log_lag_mean, color = :travel_key_cond,
            ymax = :upper, ymin = :lower, xgroup = :phase,
            Geom.subplot_grid(Geom.line(), Geom.point(), Geom.errorbar()),
            Scale.x_discrete(levels = sort(unique(group_lag_df[!,:start_reward_cat]))),
            Scale.color_discrete(levels = ["HARD", "EASY"]),
            Guide.ylabel("Log Lag"),
            Guide.xlabel("Tree First Reward by Part", orientation = :horizontal),
            #Guide.colorkey(title = "Travel Key"),
            Guide.title(title),
            Theme(key_position = :none))
    else
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
    s_data[!,:lag_scale] = copy(s_data[!,:lag]); # is this wrong?
    s_data[!,:lag] = s_data[!,:lag_scale] ./ 100;
    
    # add trial time elapsed...
    s_data[!,:trial_time_elapsed] = zeros(length(s_data[!,:time_elapsed]));
    for t_idx in unique(s_data.trial_num)
        tdat = s_data[s_data[!,:trial_num].==t_idx,:];
        s_data[s_data[!,:trial_num].==t_idx, :trial_time_elapsed] = tdat.time_elapsed .- tdat.time_elapsed[1];
    end

    s_data[ismissing.(s_data[!,:reward_obs]),:reward_obs] .= 0;
    s_data[ismissing.(s_data[!,:exit]),:exit] .= 1;
    s_data[!,:choice] = s_data[!,:exit] .+ 1;
    s_data[!,:choice] = convert(Array{Int,1}, s_data[!,:choice]);
    
    # make a column that splits by press type
    s_data[!,:button] .= string();
    s_data[s_data[!,:phase] .== "HARVEST", :button] .= "HARVEST";
    s_data[(s_data[!,:phase] .== "TRAVEL") .& (s_data[!,:travel_key_cond] .== "HARD"), :button] .= "TRAVEL_HARD";
    s_data[(s_data[!,:phase] .== "TRAVEL") .& (s_data[!,:travel_key_cond] .== "EASY"), :button] .= "TRAVEL_EASY";
    # this uses a split of 3, but changes by button press type... 
    button_lag_thresh_df = by(s_data, :button, df -> DataFrame(upper_lag_thresh = median(df.lag) + 3*mad(df.lag), 
            lower_lag_thresh = median(df.lag) - 3*mad(df.lag))) 
    s_data = join(s_data, button_lag_thresh_df, on = :button, kind = :left);
    s_data.trial_time_sec = s_data.trial_time_elapsed./1000;
    s_data.upper_lag_thresh2 = median(s_data.lag) + 4*mad(s_data.lag);
    s_data.lower_lag_thresh2 = median(s_data.lag) - 4*mad(s_data.lag); # 4 sds...

    sub_df = s_data[!, [:round, :trial_num, :lag, :lag_scale, :choice, :phase, :start_reward,
            :reward_obs, :travel_key_cond, :subjectID, :reward_true, :trial_time_elapsed, :button, 
            :upper_lag_thresh, :lower_lag_thresh, :trial_time_sec, :upper_lag_thresh2, :lower_lag_thresh2]];

    return sub_df
end

# make the lag over time and thresh over time plots
function make_smooth_rr_DF(pdata)
    pdata_lt = @where(pdata,:lag .< :upper_lag_thresh2, :lag .> :lower_lag_thresh2)
    pdata_lt.log_lag = log.(pdata_lt.lag)
    smooth_rr_DF = DataFrame();
    for s_idx in unique(pdata_lt.sub)
        print(s_idx)
        sub_rr_DF = DataFrame();
        
        s_data = @where(pdata_lt, :sub .== s_idx);
        s_data.lag_z = zscore(s_data.log_lag);
        for t_idx in unique(pdata_lt.trial_num)
            s_trial_harvest_data = @where(s_data, :phase .== "HARVEST", :trial_num .== t_idx);
            harvest_DF= try
                harvest_model = loess(s_trial_harvest_data.trial_time_sec, s_trial_harvest_data.lag_z);
                us_harvest = range(extrema(s_trial_harvest_data.trial_time_sec)...; step = 1)
                vs_harvest = Loess.predict(harvest_model, us_harvest);
                
                DataFrame(trial_time_sec = us_harvest, lag_smooth = vs_harvest, trial_num = t_idx, sub = s_idx,
                                        start_reward = s_trial_harvest_data[1,:start_reward], 
                                        travel_key_cond = s_trial_harvest_data[1,:travel_key_cond],phase = "HARVEST");
                catch e
                  bt = backtrace()
                  msg = sprint(showerror, e, bt)
                  #println(msg)
                    DataFrame();
                end

            s_trial_travel_data = @where(s_data, :phase .== "TRAVEL", :trial_num .== t_idx);
            travel_DF = try
                
                travel_model = loess(s_trial_travel_data.trial_time_sec, s_trial_travel_data.lag_z);
                us_travel = range(extrema(s_trial_travel_data.trial_time_sec)...; step = 1)
                vs_travel = Loess.predict(travel_model, us_travel);
                DataFrame(trial_time_sec = us_travel, lag_smooth = vs_travel, trial_num = t_idx, sub = s_idx,
                                        start_reward = s_trial_travel_data[1,:start_reward], 
                                        travel_key_cond = s_trial_travel_data[1,:travel_key_cond],phase = "TRAVEL");
            catch
                DataFrame();
            end
            
            sub_rr_DF = [sub_rr_DF; harvest_DF; travel_DF];
        end # end loop over trials
        #if (any(sub_rr_DF.lag_smooth .> 8) | any(sub_rr_DF.lag_smooth .< -10))
         #   print("small_lag")
        #end
        #print(sub_rr_DF)
        if (length(unique(sub_rr_DF.trial_num)) == length(unique(pdata_lt.trial_num))) & !(any(sub_rr_DF.lag_smooth .> 8) | any(sub_rr_DF.lag_smooth .< -8))
            smooth_rr_DF = [smooth_rr_DF; sub_rr_DF];
        end
    end
    smooth_rr_DF.trial_time_sec = ceil.(smooth_rr_DF.trial_time_sec);
    return smooth_rr_DF
end

# plot group response rate over time...
function plot_group_rr_over_time(pdata_lt)
    
    smooth_rr_DF = make_smooth_rr_DF(pdata_lt);
    
    smooth_rr_means = by(smooth_rr_DF, 
        [:trial_time_sec, :start_reward, :travel_key_cond, :phase], 
        :lag_smooth => mean,
        :lag_smooth => sem);
    smooth_rr_means.upper = smooth_rr_means.lag_smooth_mean + smooth_rr_means.lag_smooth_sem;
    smooth_rr_means.lower = smooth_rr_means.lag_smooth_mean - smooth_rr_means.lag_smooth_sem;
    smooth_rr_means.start_reward_cat = CategoricalArray(smooth_rr_means.start_reward);

    p1 = plot(@where(smooth_rr_means, :phase .== "HARVEST",:travel_key_cond .== "EASY",
            :trial_time_sec .> 20,:trial_time_sec .< 120), 
        x = :trial_time_sec, y = :lag_smooth_mean, ymax = :upper, ymin = :lower,
        color = :start_reward_cat, Geom.line(), Geom.ribbon(), Guide.title("Harvest Easy"),
        Scale.color_discrete_hue(levels = [60,90,120]),
        Guide.ylabel("Response Rate"), Guide.xlabel("Time (seconds)"), Guide.colorkey(title = "Start Reward"));
        
    p1a = plot(@where(smooth_rr_means, :phase .== "HARVEST",:travel_key_cond .== "HARD",
            :trial_time_sec .> 20,:trial_time_sec .< 120), 
        x = :trial_time_sec, y = :lag_smooth_mean, ymax = :upper, ymin = :lower,
        color = :start_reward_cat, Geom.line(), Geom.ribbon(), 
        Guide.title("Harvest Hard"), Scale.color_discrete_hue(levels = [60,90,120]),
        Guide.ylabel("Response Rate"), Guide.xlabel("Time (seconds)"), Guide.colorkey(title = "Start Reward"));
    

    p2 = plot(@where(smooth_rr_means, :phase .== "TRAVEL", :travel_key_cond .== "EASY",
            :trial_time_sec .> 20,:trial_time_sec .< 110), 
        x = :trial_time_sec, y = :lag_smooth_mean, ymax = :upper, ymin = :lower,
        color = :start_reward_cat, Geom.line(), Geom.ribbon(),
        Guide.title("Travel Easy"),Scale.color_discrete_hue(levels = [60,90,120]),
        Guide.ylabel("Response Rate"), Guide.xlabel("Time (seconds)"), Guide.colorkey(title = "Start Reward"));

    
    p3 = plot(@where(smooth_rr_means, :phase .== "TRAVEL", 
            :travel_key_cond .== "HARD",:trial_time_sec .> 20,:trial_time_sec .< 120), 
        x = :trial_time_sec, y = :lag_smooth_mean, ymax = :upper, ymin = :lower, color = :start_reward_cat,
        Geom.line(),Geom.ribbon(), Guide.title("Travel Hard"),
        Scale.color_discrete_hue(levels = [60,90,120]),
        Guide.ylabel("Response Rate"), Guide.xlabel("Time (seconds)"), Guide.colorkey(title = "Start Reward"));

    draw(PNG(20cm,30cm), title(gridstack([p1 p1a; p2 p3]), string("N Subj: ", length(unique(smooth_rr_DF.sub)))));
end

# average this over subjects..
function make_smooth_thresh_DF(pdata)
    round_exit_data = by(pdata, [:sub, :trial_num], df -> get_trial_exit_threshs(df));
    round_exit_data.last_reward_sec = round_exit_data.last_reward_time./1000; # in units of 1000...
    
    smooth_thresh_DF = DataFrame();
    for s_idx in unique(round_exit_data.sub)
        sub_exit_DF = DataFrame();
        #print(s_idx)
        s_data = @where(round_exit_data, :sub .== s_idx);
        s_data.last_reward_time = float.(s_data.last_reward_time);
        s_data.last_reward = float.(s_data.last_reward);
        for t_idx in unique(pdata.trial_num)

            s_trial_data = @where(s_data,:trial_num .== t_idx);

            exit_DF = try
                exit_model_ext = LinearInterpolation(s_trial_data.last_reward_sec, s_trial_data.last_reward, extrapolation_bc=Interpolations.Flat()) # create interpolation function
                #exit_model_ext = extrapolate(interpolate(s_trial_data.last_reward_sec, s_trial_data.last_reward, scheme)
                #exit_model_ext = extrapolate(exit_model, Flat())
                #exit_model = loess(s_trial_data.last_reward_sec, s_trial_data.last_reward,span=.98);
                #us_exit = range(extrema(s_trial_data.last_reward_sec)...; step = 1)
                us_exit = range(extrema(round_exit_data.last_reward_sec)...; step = 1)
                vs_exit = exit_model_ext(us_exit);#Loess.predict(exit_model, us_exit);
                DataFrame(trial_time_sec = us_exit, thresh_smooth = vs_exit, trial_num = t_idx, sub = s_idx,
                                        start_reward = s_trial_data[1,:start_reward], 
                                        travel_key_cond = s_trial_data[1,:travel_key_cond]);
                catch e
                # print the error for 6...
                bt = backtrace()
                  msg = sprint(showerror, e, bt)
                 # println(msg)
                DataFrame()
            end
            #print(unique(exit_DF.trial_num))
            
           # if !(nrow(exit_DF) > 0)
            #    print("fail")
            #end
            # throw out subjects for which we missed some
            sub_exit_DF = [sub_exit_DF; exit_DF];
            #print(sub_exit_DF)
        end
        if (nrow(sub_exit_DF) > 1)
            if(length(unique(sub_exit_DF.trial_num)) == length(unique(pdata.trial_num)))
                smooth_thresh_DF = [smooth_thresh_DF; sub_exit_DF];
            else
               println(string(s_idx, ": fail some"))
            end
        else
            println(string(s_idx, ": fail all"))
        end
    end
    return smooth_thresh_DF;
end

# differences between these are present from the first trial basically...
function plot_group_thresh_over_time(pdata)
    
    smooth_thresh_DF = make_smooth_thresh_DF(pdata);
    smooth_thresh_DF.trial_time_sec = ceil.(smooth_thresh_DF.trial_time_sec);

    # get the group mean over time...
    # plot means and sems
    smooth_thresh_means = by(smooth_thresh_DF, 
        [:trial_time_sec, :start_reward, :travel_key_cond], 
        :thresh_smooth => mean,
        :thresh_smooth => sem);
    smooth_thresh_means.upper = smooth_thresh_means.thresh_smooth_mean + smooth_thresh_means.thresh_smooth_sem;
    smooth_thresh_means.lower = smooth_thresh_means.thresh_smooth_mean - smooth_thresh_means.thresh_smooth_sem;

    p = plot(@where(smooth_thresh_means, :trial_time_sec .> 1 , :trial_time_sec .< 130), x = :trial_time_sec, y = :thresh_smooth_mean, 
        ymin =:lower, ymax =:upper,
        xgroup = :start_reward, 
        color = :travel_key_cond,
        Geom.subplot_grid(
            Geom.line(),
            Geom.ribbon()
            ), 
        Guide.ylabel("Exit Threshold"),
        Guide.xlabel("Time (sec)"),
        Scale.xgroup(levels = [60,90,120]), 
        Scale.color_discrete_hue(levels = ["EASY", "HARD"]),
        Guide.colorkey(title = "Travel Cost"),
        Guide.title(string("N Subj: ", length(unique(smooth_thresh_DF.sub)))),
        Theme(panel_fill=colorant"white"));
    draw(PNG(),p)
end






