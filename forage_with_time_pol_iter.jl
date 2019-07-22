
cd("C:\\Users\\erussek\\foraging")

using Gadfly
using DataFrames
using Distances
using DataFramesMeta
using CategoricalArrays
include("./forage_utils.jl")
import Cairo
import Fontconfig

function build_det_MDP(start_reward, decrement, n_travel_states)
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

    return Rs, next_state

end

function time_cost(lag, unit_cost)
    return unit_cost/lag
end

time_cost_plot = plot([x -> time_cost(x,2),
    x -> time_cost(x,6),
    x -> time_cost(x,10)],
        .1,1,
        Guide.xlabel("lag"), Guide.ylabel("time cost"),
        Guide.colorkey(title="Unit Cost",labels = ["2", "6", "10"] ))
draw(PDF("plots/time_cost_plot.pdf", 8inch, 8inch), time_cost_plot)


function evaluate_policy(Rs, next_state,vigor_cost, policy, lag)

    n_states = length(Rs);
    ref_state = n_states - 1;
    all_states = Array(1:n_states);
    all_states_not_ref = filter!(x-> x!=ref_state , all_states);

    # run value iteration
    V_pi = zeros(n_states);
    rho_pi = 0.;
    sp1 = 100;
    iter = 0;
    other_choice = [2 1];
    max_val_change = 1000;
        #while sp1 >  .01
    while max_val_change > .0001
        iter += 1
        last_V_pi = copy(V_pi);
        last_rho_pi = copy(rho_pi);

        for i = ref_state
            this_state_choice = policy[i];
            this_state_other = other_choice[policy[i]]

             rho_pi = (.99*Rs[next_state[i,this_state_choice]] + .01*Rs[next_state[i,this_state_other]] +
                      .99*last_V_pi[next_state[i,this_state_choice]] + .01*last_V_pi[next_state[i,this_state_other]])/lag
                      - vigor_cost/(lag^2);
        end

        for i = all_states_not_ref

            this_state_choice = policy[i];
            this_state_other = other_choice[policy[i]]

            V_pi[i] = .99*Rs[next_state[i,this_state_choice]] + .01*Rs[next_state[i,this_state_other]] +
                        .99*last_V_pi[next_state[i,this_state_choice]] + .01*last_V_pi[next_state[i,this_state_other]] -
                        rho_pi*lag - vigor_cost/lag;
        end

        max_val_change = maximum(abs.(V_pi - last_V_pi))
    end

    return V_pi, rho_pi
end

function improve_policy(V_pi, rho_pi, Rs, next_state, vigor_cost)
    n_states = length(V_pi);
    policy = zeros(Int64,n_states);
    lag = 1;
    # improve the choice policy - this is just based on next states
    for i = 1:n_states
        Q1 =  .99*Rs[next_state[i,1]] + .01*Rs[next_state[i,2]] +
                .99*V_pi[next_state[i,1]] + .01*V_pi[next_state[i,2]];
        Q2 =  .99*Rs[next_state[i,2]] + .01*Rs[next_state[i,1]] +
                .99*V_pi[next_state[i,2]] + .01*V_pi[next_state[i,1]];

        policy[i] = argmax([Q1 Q2])[2]
    end

    println("imp_pol: ", policy[1:25])

    if (rho_pi > 0)
        lag = sqrt(vigor_cost/rho_pi)
    else
        lag = 500;
    end

    return policy, lag

end


# policy iteration
function solve_policy(Rs, next_state, vigor_cost)
    iter = 0;
    n_states = length(Rs);
    policy = ones(Int64,n_states);
    lag = 500;
    max_change = 100;
    max_pol_change = 100;
    V_pi = zeros(n_states)
    rho_pi = 0;
    while max_change > .001
        iter += 1;
        #println(iter)
        old_policy = copy(policy);
        old_lag = copy(lag);
        V_pi, rho_pi = evaluate_policy(Rs, next_state,vigor_cost, policy, lag)
        #print(V_pi)
        policy, lag = improve_policy(V_pi, rho_pi, Rs, next_state, vigor_cost)
        println("solve_pol: ", policy[1:25])
        max_pol_change = maximum(abs.(policy - old_policy))
        #println(max_pol_change)
        lag_change = abs(lag - old_lag)
        max_change = maximum([max_pol_change lag_change])
    end
    return policy, lag, V_pi, rho_pi
    println("pot_pol: ", policy[1:25])
end

function sim_forage_pi(start_reward, decrement, n_travel_states, vigor_cost)

    Rs, next_state = build_det_MDP(start_reward,decrement,n_travel_states)
    policy, lag, V_pi, rho_pi = solve_policy(Rs, next_state, vigor_cost)

    n_states = length(Rs);
    next_R_stay = zeros(n_states);
    for i = 1:n_states
        next_R_stay[i] = Rs[next_state[i,1]];
    end

    res_df = DataFrame(state=1:n_states, pol = policy, lag = lag*ones(n_states),
                R = Rs, n_travel = n_travel_states*ones(n_states),
                start_R = start_reward*ones(n_states), next_R = next_R_stay,
                decrement = decrement*ones(n_states), V = V_pi, rho = rho_pi*ones(n_states),
                vigor_cost = vigor_cost*ones(n_states));
    return res_df
end


data = sim_forage_pi(10.,.9,10, 2)

plot(data, y = :pol, x = :next_R)


data = DataFrame();

# run this for different start rewards
for start_reward = [10. 15. 20.]
    for decrement = [.9]
        for n_travel_states = [5 10 20]
            for vigor_cost = [0.2 10. 100.]
                global data = [data; sim_forage_pi(start_reward,decrement,n_travel_states, vigor_cost)]
            end
        end
    end
end

data = DataFrame();

for vigor_cost = [0.2 10. 100.]
    global data = [data; sim_forage_pi(15.,.9,20, vigor_cost)]
end

data1 = sim_forage_pi(15.,.9,20, .5)
data2 = sim_forage_pi(15.,.9,20, 20)

# plot lag as a function of start reward, n_travel_states, vigor_cost
# plot policy as a function of start_reward, n_travel states, vigor_cost
#

# want to make a plot
data_part = @linq data |>
           transform(decrement = CategoricalArray(:decrement),
                    start_R = CategoricalArray(:start_R),
                    n_travel = CategoricalArray(:n_travel),
                    vigor_cost = CategoricalArray(:vigor_cost))



Gadfly.push_theme(:default)

fig1a = plot(data_part, x=:next_R, y=:pol,
     Geom.subplot_grid(Geom.line ),
        color = :vigor_cost, xgroup=:start_R, ygroup = :n_travel,
        Guide.ylabel("Policy"),
        Guide.xlabel("Next R by Start R"),
        Guide.colorkey(title = "N Travel States"),
        style(line_width = 2pt)
        )

# plot the value we solved for row for each policy
fig1b = plot(data_part, x=:start_R, y=:rho,
            Geom.subplot_grid(Geom.bar(position= :dodge)),
            color=:vigor_cost, ygroup = :n_travel,
            Guide.ylabel("Rho"),
            Guide.xlabel("Start R"),
            Guide.colorkey(title = "vigor cost"))

# plot the value we solved for row for each policy
fig1c = plot(data_part, x=:start_R, y=:lag,
    Geom.subplot_grid(Geom.bar(position= :dodge)),
    color=:vigor_cost, ygroup = :n_travel,
    Guide.ylabel("Lag"),
    Guide.xlabel("Start R"),
    Guide.colorkey(title = "vigor cost"))
