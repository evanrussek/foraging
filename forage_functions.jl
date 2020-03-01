
#cd("C:\\Users\\erussek\\foraging")

using Gadfly
using DataFrames
using Distances
using DataFramesMeta
using CategoricalArrays
import Cairo
import Fontconfig
using Statistics

# build the MDP (returns  S -> R and next-state mtx SXA -> S')

function build_det_MDP(start_reward, decrement, n_travel_states)
    # build the deterministic MDP
    # takes in:
    #          start_reward: doulble point value tree starts at,
    #          decrement: double / tree decay rate with each press
    #          n_travel_states: INT - number of states to transition through when traveling
    # returns:
    #        RS: (array) S -> R and  next_state: (matrix) S,A -> S')

    # takes in
    Rtree = [start_reward];

    reward = copy(start_reward)
    # figure our how many tree states we need
    current_reward = Float64(start_reward);
    while current_reward > .1
        next_reward = current_reward*decrement
        prepend!(Rtree,next_reward)
        current_reward = next_reward
    end
    prepend!(Rtree,0) # let it end at 0 rewards
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

## plot the time cost
function time_cost(lag, unit_cost)
    return unit_cost/lag
end

time_cost_plot = plot([x -> time_cost(x,2),
    x -> time_cost(x,6),
    x -> time_cost(x,10)],
        .1,1,
        Guide.xlabel("lag"), Guide.ylabel("time cost"),
        Guide.colorkey(title="Unit Cost",labels = ["2", "6", "10"] ))
#draw(PDF("plots/time_cost_plot.pdf", 8inch, 8inch), time_cost_plot)


function evaluate_policy(Rs, next_state, vigor_cost, policy, lag)
    # function to evaluate a policy by setting ref state to 0
    # evaluate each other state and reward rate w/r.t. to that
    # Takes in:
        # Rs: s -> r
        # next_state: (s,a) -> s'
        # vigor cost: a -> c (cost of a press per unit time)
        # policy: array of INT s -> a
        # lag: a -> lag (array) - current decision about how fast to transition for each action
    # Returns:
        # V_pi: state value s -> v
        # rho: average reward rate
        #

    n_states = length(Rs);
    ref_state = 15;
    all_states = Array(1:n_states);
    all_states_not_ref = filter!(x-> x!=ref_state , all_states);

    # run value iteration
    V_pi = zeros(n_states);
    rho_pi = 0.;
    iter = 0;
    other_choice = [2 1];
    max_val_change = 1000;
        #while sp1 >  .01
    while max_val_change > .00001
        iter += 1
        last_V_pi = copy(V_pi);
        last_rho_pi = copy(rho_pi);

        for i = ref_state
            this_state_choice = policy[i];
            this_state_other = other_choice[policy[i]]

             rho_pi = (.99*Rs[next_state[i,this_state_choice]] + .01*Rs[next_state[i,this_state_other]] +
                      .99*last_V_pi[next_state[i,this_state_choice]] + .01*last_V_pi[next_state[i,this_state_other]])/lag[i]
                      - vigor_cost[this_state_choice]/(lag[i]^2);
        end

        for i = all_states_not_ref

            this_state_choice = policy[i];
            this_state_other = other_choice[policy[i]]

            V_pi[i] = .99*Rs[next_state[i,this_state_choice]] + .01*Rs[next_state[i,this_state_other]] +
                        .99*last_V_pi[next_state[i,this_state_choice]] + .01*last_V_pi[next_state[i,this_state_other]] -
                        rho_pi*lag[i] - vigor_cost[this_state_choice]/lag[i];
        end

        max_val_change = maximum(abs.(V_pi - last_V_pi))
    end
    # now re-solve for the reference state (weird that you have to do this?)
    for i = ref_state
        this_state_choice = policy[i];
        this_state_other = other_choice[policy[i]]

        V_pi[i] = .99*Rs[next_state[i,this_state_choice]] + .01*Rs[next_state[i,this_state_other]] +
                    .99*V_pi[next_state[i,this_state_choice]] + .01*V_pi[next_state[i,this_state_other]] -
                    rho_pi*lag[i] - vigor_cost[this_state_choice]/lag[i];
    end

    return V_pi, rho_pi
end

function improve_policy(V_pi, rho_pi, Rs, next_state, vigor_cost)
    # returns new policy and lag based on value of current policy/lag and average reward rate
    n_states = length(V_pi);
    policy = zeros(Int64,n_states);
    lag = ones(n_states);
    # improve the choice policy - this is just based on next states
    # get the optimal lag for Q1, and the optimal lag for Q2
    for i = 1:n_states
        # lag1
        if (rho_pi > 0)
            lag1 = sqrt(vigor_cost[1]/rho_pi)
            lag2 = sqrt(vigor_cost[2]/rho_pi)
        else
            lag1 = 500;
            lag2 = 500;
        end
        these_lags = [lag1 lag2];

        Q1 =  .99*Rs[next_state[i,1]] + .01*Rs[next_state[i,2]] +
                .99*V_pi[next_state[i,1]] + .01*V_pi[next_state[i,2]] -
                these_lags[1]*rho_pi - vigor_cost[1]/these_lags[1];
        Q2 =  .99*Rs[next_state[i,2]] + .01*Rs[next_state[i,1]] +
                .99*V_pi[next_state[i,2]] + .01*V_pi[next_state[i,1]] -
                these_lags[2]*rho_pi - vigor_cost[2]/these_lags[2];

        choice = argmax([Q1 Q2])[2];
        policy[i] = choice;
        lag[i] = these_lags[choice];
    end

    return policy, lag
end

# policy iteration
function solve_policy(Rs, next_state, vigor_cost)
    iter = 0;
    n_states = length(Rs);
    policy = ones(Int64,n_states);
    lag = 500 .*ones(n_states);
    max_change = 100;
    max_pol_change = 100;
    V_pi = zeros(n_states)
    rho_pi = 0;
    while max_change > .0001
        iter += 1;
        #println(iter)
        old_policy = copy(policy);
        old_lag = copy(lag);
        V_pi, rho_pi = evaluate_policy(Rs, next_state,vigor_cost, policy, lag)
        #print(V_pi)
        policy, lag = improve_policy(V_pi, rho_pi, Rs, next_state, vigor_cost)
        #println("solve_pol: ", policy[1:25])
        max_pol_change = maximum(abs.(policy - old_policy))
        #println(max_pol_change)
        lag_change = maximum(abs.(lag - old_lag))
        max_change = maximum([max_pol_change lag_change])
    end
    #println("pot_pol: ", policy[40:51])
    return policy, lag, V_pi, rho_pi
end

function sim_forage_pi(start_reward, decrement, n_travel_states, vigor_cost)

    Rs, next_state = build_det_MDP(start_reward,decrement,n_travel_states)
    policy, lag, V_pi, rho_pi = solve_policy(Rs, next_state, vigor_cost)

    n_states = length(Rs);
    n_tree_states = n_states - n_travel_states;
    next_R_stay = zeros(n_states);
    for i = 1:n_states
        next_R_stay[i] = Rs[next_state[i,1]];
    end

    # get the lag for stay and the lag for leave
    stay_lag = lag[policy .== 1][1];
    leave_lag = lag[policy .== 2][1];

    #print(stay_lag)

    next_TR_stay = zeros(n_states);
    for i = 1:n_states
        next_TR_stay[i] = Rs[next_state[i,1]]/stay_lag - vigor_cost[1]/(stay_lag^2);
    end

    tree_states = zeros(Int64,n_states);
    tree_states[1:n_tree_states] .= 1;
    travel_states = zeros(Int64,n_states);
    travel_states[n_tree_states + 1: n_states] .= 1;

    ##
    tree_pol = policy[tree_states .== 1]
    # want first highest # state with policy = 2
    first_leave_state = maximum(findall(tree_pol .== 2))
    # get the reward in teh state after this
    reward_thresh = Rs[next_state[first_leave_state,1]];
    predicted_thresh = reward_thresh/stay_lag - vigor_cost[1]/(stay_lag^2);

    # get the exit threshold

    res_df = DataFrame(state=1:n_states,tree_states = tree_states, travel_states = travel_states,
                reward_thresh = ones(n_states)*reward_thresh, pred_thresh = ones(n_states)*predicted_thresh,
                pol = policy, lag = lag, stay_lag = stay_lag.*ones(n_states), leave_lag = leave_lag*ones(n_states),
                R = Rs, n_travel = n_travel_states*ones(n_states),
                start_R = start_reward*ones(n_states), next_R = next_R_stay, next_TR = next_TR_stay,
                decrement = decrement*ones(n_states), V = V_pi, rho = rho_pi*ones(n_states),
                vigor_cost1 = vigor_cost[1]*ones(n_states), vigor_cost2 = vigor_cost[2]*ones(n_states));
    return res_df
end
