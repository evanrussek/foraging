
#cd("C:\\Users\\erussek\\foraging")
cd("/Users/evanrussek/foraging")

# add package
using Gadfly
using DataFrames
using Distances
using DataFramesMeta
using CategoricalArrays
#using Cairo
#using Fontconfig
using Statistics

include("forage_functions.jl")

## run through experiments..
data_exp = DataFrame();

# run this for different start rewards
for start_reward = [60.,90.,120]
    for decrement = .98
        for n_travel_states = 22
            for vigor_cost1 = 1.
                for vigor_cost2 = 1.:4.:20.
                    #println(start_reward, n_travel_states, vigor_cost2)
                    global data_exp = [data_exp; sim_forage_pi(start_reward,decrement,
                            n_travel_states, [vigor_cost1, vigor_cost2])]
                end
            end
        end
    end
end

# want to make a plot
data_part_exp = @linq data_exp |>
           transform(decrement = CategoricalArray(:decrement),
                    #start_R = CategoricalArray(:start_R),
                    n_travel = CategoricalArray(:n_travel),
                    #vigor_cost2 = CategoricalArray(:vigor_cost2),
                    pol = CategoricalArray(:pol),
                    tree_states = CategoricalArray(:tree_states))

#
# plot reward thresh as a function of start reward, splitting by time_cost
exp_sum = @linq data_part_exp |>
        by([:start_R, :vigor_cost2],
        stay_lag = mean(:stay_lag), leave_lag = mean(:leave_lag),
        exit_thresh = mean(:reward_thresh))


# these are connected in some way
exit_df = @select(exp_sum, :start_R, :vigor_cost2, :exit_thresh)
lag_df = @select(exp_sum, :start_R, :vigor_cost2, :stay_lag, :leave_lag)
lag_df = stack(lag_df, [:stay_lag, :leave_lag])

lag_df = @transform(lag_df, lag = :value)
lag_df.phase = Array{Union{Missing, String}}(missing, nrow(lag_df))
lag_df.phase[lag_df.variable .== :stay_lag] .= "HARVEST"
lag_df.phase[lag_df.variable .== :leave_lag] .= "TRAVEL"
lag_df.phase = CategoricalArray(lag_df.phase)

sim_ext_plot = plot(exit_df, y = :exit_thresh, x = :start_R,
    color = :vigor_cost2,
    Guide.ylabel("Last Reward Before Exit"),
    Guide.xlabel("Tree First Reward"),
    Guide.title("Optimal Policy Simulation"),
    Geom.line, Geom.point,
    Theme(line_width = 2pt))


sim_lag_plot = plot(lag_df, y = :lag, x = :start_R, color = :vigor_cost2,
    xgroup=:phase,
    Scale.xgroup(levels=["TRAVEL","HARVEST"]),
    Geom.subplot_grid(Geom.line,Geom.point),
    Guide.ylabel("Lag (Arbitrary Units)"),
    Guide.xlabel("Tree First Reward by Trial Part"),
    Guide.title("Optimal Policy Simulation"),
    Theme(line_width = 2pt))



#using Colors
#Gadfly.push_theme(:dark)
#plot(exp2_sum,y = :exit_thresh, x = :vigor_cost2,#
#    color = :start_R, Geom.line)
#plot(exp2_sum,y = :stay_lag, x = :vigor_cost2, color = :start_R, Geom.line,Theme(line_width = 3pt),
#            Scale.color_continuous(colormap=p->RGB(.5,p,.5)))

#plot(exp2_sum,y = :leave_lag, x = :vigor_cost2, color = :start_R,
# Geom.line, Theme(line_width = 3pt),
# Scale.color_continuous(colormap=p->RGB(.5,p,.5)))



#exit_thresh_lag = vstack(lag_plot, exit_thresh)

#draw(PDF("plots/exit_thresh_lag2.pdf", 8inch, 12inch), exit_thresh_lag)
#draw(PDF("plots/exit_thresh_lag_point2.pdf", 8inch, 12inch), lag_vs_thresh_point)
