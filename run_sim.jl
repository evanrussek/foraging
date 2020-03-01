
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
                for vigor_cost2 = 1.:5:21.
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
exp2_sum = @linq data_part_exp2 |>
        by([:start_R, :vigor_cost2],
        stay_lag = mean(:stay_lag), leave_lag = mean(:leave_lag),
        exit_thresh = mean(:reward_thresh))


plot(exp2_sum, y = :exit_thresh, x = :start_R, color = :vigor_cost2, Geom.line, Geom.point)
plot(exp2_sum, y = :stay_lag, x = :start_R, color = :vigor_cost2, Geom.line, Geom.point)
plot(exp2_sum, y = :leave_lag, x = :start_R, color = :vigor_cost2, Geom.line, Geom.point)


using Colors
Gadfly.push_theme(:dark)
plot(exp2_sum,y = :exit_thresh, x = :vigor_cost2,
    color = :start_R, Geom.line)
plot(exp2_sum,y = :stay_lag, x = :vigor_cost2, color = :start_R, Geom.line,Theme(line_width = 3pt),
            Scale.color_continuous(colormap=p->RGB(.5,p,.5)))

plot(exp2_sum,y = :leave_lag, x = :vigor_cost2, color = :start_R,
 Geom.line, Theme(line_width = 3pt),
 Scale.color_continuous(colormap=p->RGB(.5,p,.5)))



#exit_thresh_lag = vstack(lag_plot, exit_thresh)

#draw(PDF("plots/exit_thresh_lag2.pdf", 8inch, 12inch), exit_thresh_lag)
#draw(PDF("plots/exit_thresh_lag_point2.pdf", 8inch, 12inch), lag_vs_thresh_point)
