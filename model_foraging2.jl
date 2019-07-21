# tree states are 7,6,5,4,3,2,1 - 1 leads to itself
# travel states are 8,9,10,11,12

# this works

using Gadfly
using DataFrames
using Distances
using DataFramesMeta
using CategoricalArrays


function solve_policy(start_reward, decrement, n_travel_states)
    Rtree = [start_reward];

    reward = copy(start_reward)
    # figure our how many tree states we need
    current_reward = Float64(start_reward);
    while current_reward > .5
        next_reward = current_reward*decrement
        prepend!(Rtree,next_reward)
        current_reward = next_reward
    end
    n_tree_states = length(Rtree);
    n_states = n_tree_states + n_travel_states;

    next_state = zeros(Int64,n_states,2);
    # fill in for tree states
    for i = n_tree_states:-1:1
        next_state[i,1] = i - 1;
        next_state[i,2] = n_tree_states + 1;
    end
    next_state[1,1] = 1;
    Rs = [Rtree; zeros(n_travel_states)]

    # fill in for travel states
    for i = n_tree_states+1:n_states
        next_state[i,1] = i;
        next_state[i,2] = i+1
    end
    next_state[n_states,2] = n_tree_states;

    # run value iteration
    V_pi = zeros(n_states)
    sp1 = 100;
    iter = 0;
    policy = zeros(n_states);
    while sp1 >  .01
        iter += 1
        last_V_pi = copy(V_pi);
        last_sp1 = copy(sp1);
        last_pol = copy(policy);
        for i = 1:n_states
            Q1 =  .99*Rs[next_state[i,1]] + .01*Rs[next_state[i,2]] +
                    .99*last_V_pi[next_state[i,1]] + .01*last_V_pi[next_state[i,2]];
            Q2 =  .99*Rs[next_state[i,2]] + .01*Rs[next_state[i,1]] +
                    .99*last_V_pi[next_state[i,2]] + .01*last_V_pi[next_state[i,1]];

            V_pi[i] = maximum([Q1 Q2])
            policy[i] = argmax([Q1 Q2])[2]
        end

        sp1 = spannorm_dist(V_pi, last_V_pi)
    end

    #return policy, Rs
    res_df = DataFrame(state=1:n_states, pol = policy, R = Rs,
                n_travel = n_travel_states*ones(n_states),
                start_R = start_reward*ones(n_states),
                decrement = decrement*ones(n_states));
    return res_df

end

data = DataFrame();

# run this for different start rewards
for start_reward = [10. 15. 20.]
    for decrement = [.9]
        for n_travel_states = [5 10 20]
            global data = [data; solve_policy(start_reward,decrement,n_travel_states)]
        end
    end
end


# want to make a plot
data_part = @linq data |>
           transform(decrement = CategoricalArray(:decrement),
                    start_R = CategoricalArray(:start_R),
                    n_travel = CategoricalArray(:n_travel))



Gadfly.push_theme(:default)

fig1a = plot(data_part, x=:R, y=:pol,
     Geom.subplot_grid(Geom.line),
        color = :n_travel, xgroup=:start_R,
        Guide.ylabel("Policy"),
        Guide.xlabel("Last R by Start R"),
        Guide.colorkey(title = "N Travel States"),
        style(line_width = 2pt),
        )

fig1b = plot(data_part, x=:R, y=:pol,
        Geom.subplot_grid(Geom.line),
        color = :start_R, xgroup=:n_travel,
        Guide.ylabel("Policy"),
        Guide.xlabel("Last R by N Travel States"),
        Guide.colorkey(title = "Start R"),
        style(line_width = 2pt),
        #Theme(major_label_color = "white", minor_label_color = "white",
        #key_title_color = "white", key_label_color = "white")
        #Theme(background_color = "white")
        )

myplot = vstack(fig1a,fig1b)

draw(PDF("plots/optimal_no_time.pdf", 10inch, 10inch), myplot)
