# tree states are 7,6,5,4,3,2,1 - 1 leads to itself
# travel states are 8,9,10,11,12

# this works

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

function evaluate_policy(Rs, next_state, policy)

    n_states = length(Rs);
    ref_state = 5;
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

            #
        for i = ref_state
            this_state_choice = policy[i];
            this_state_other = other_choice[policy[i]]

             rho_pi = .99*Rs[next_state[i,this_state_choice]] + .01*Rs[next_state[i,this_state_other]] +
                      .99*last_V_pi[next_state[i,this_state_choice]] + .01*last_V_pi[next_state[i,this_state_other]];
        end

        for i = all_states_not_ref

            this_state_choice = policy[i];
            this_state_other = other_choice[policy[i]]

            V_pi[i] = .99*Rs[next_state[i,this_state_choice]] + .01*Rs[next_state[i,this_state_other]] +
                        .99*last_V_pi[next_state[i,this_state_choice]] + .01*last_V_pi[next_state[i,this_state_other]] -
                        rho_pi;
        end

        max_val_change = maximum(abs.(V_pi - last_V_pi))
    end

    return V_pi, rho_pi
end

function improve_policy(V_pi, Rs, next_state)
    n_states = length(V_pi);
    policy = zeros(Int64,n_states);
    for i = 1:n_states
        Q1 =  .99*Rs[next_state[i,1]] + .01*Rs[next_state[i,2]] +
                .99*V_pi[next_state[i,1]] + .01*V_pi[next_state[i,2]];
        Q2 =  .99*Rs[next_state[i,2]] + .01*Rs[next_state[i,1]] +
                .99*V_pi[next_state[i,2]] + .01*V_pi[next_state[i,1]];

        policy[i] = argmax([Q1 Q2])[2]
    end
    return policy
end

# policy iteration
function solve_policy2(Rs, next_state)
    n_states = length(Rs);
    policy = ones(Int64,n_states);
    max_pol_change = 100;
    V_pi = zeros(n_states)
    rho_pi = 0;
    while max_pol_change > .001
        old_policy = copy(policy);
        V_pi, rho_pi = evaluate_policy(Rs, next_state, policy)
        #print(V_pi)
        policy = improve_policy(V_pi, Rs, next_state)
        max_pol_change = maximum(abs.(policy - old_policy))
    end
    return (policy, V_pi, rho_pi)
end

function sim_forage_pi(start_reward, decrement, n_travel_states)
    Rs, next_state = build_det_MDP(start_reward,decrement,n_travel_states)
    policy, V_pi, rho_pi = solve_policy2(Rs, next_state)
    #return policy, Rs
    n_states = length(Rs);

    next_R_stay = zeros(n_states);
    for i = 1:n_states
        next_R_stay[i] = Rs[next_state[i,1]];
    end

    res_df = DataFrame(state=1:n_states, pol = policy, R = Rs,
                n_travel = n_travel_states*ones(n_states),
                start_R = start_reward*ones(n_states), next_R = next_R_stay,
                decrement = decrement*ones(n_states), V = V_pi, rho = rho_pi*ones(n_states));
    return res_df
end


# key parameters are unit cost,
vig_cost = 5;
decrement = .9;
start_R = 20.;
n_travel_states = 20;

Rs, next_state = build_det_MDP(start_R, decrement, n_travel_states);
n_states = length(Rs);
policy = ones(Int64,n_states);

# plot the policies again to makes sure it makes sense


data = DataFrame();

# run this for different start rewards
for start_reward = [10. 15. 20.]
    for decrement = [.9]
        for n_travel_states = [5 10 20]
            global data = [data; sim_forage_pi(start_reward,decrement,n_travel_states)]
        end
    end
end


# want to make a plot
data_part = @linq data |>
           transform(decrement = CategoricalArray(:decrement),
                    start_R = CategoricalArray(:start_R),
                    n_travel = CategoricalArray(:n_travel))



Gadfly.push_theme(:default)

fig1a = plot(data_part, x=:next_R, y=:pol,
     Geom.subplot_grid(Geom.line ),
        color = :n_travel, xgroup=:start_R,
        Guide.ylabel("Policy"),
        Guide.xlabel("Next R by Start R"),
        Guide.colorkey(title = "N Travel States"),
        style(line_width = 2pt)
        )

fig1b = plot(data_part, x=:next_R, y=:pol,
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

    # plot the value we solved for row for each policy
    fig1c = plot(data_part, x=:start_R, y=:rho, color=:n_travel,
        Geom.bar(position= :dodge),
        Guide.ylabel("Rho*"),
        Guide.xlabel("Start R"),
        Guide.colorkey(title = "N Travel States"),)

myplot = vstack(fig1a, fig1c)

draw(PDF("plots/policy_iter_no_time.pdf", 8inch, 8inch), myplot)
