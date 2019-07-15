# tree states are 7,6,5,4,3,2,1 - 1 leads to itself
# travel states are 8,9,10,11,12

using Gadfly
using DataFrames
using Distances


function solve_policy(start_reward, decrement, n_travel_states)
    Rtree = [start_reward];

    reward = copy(start_reward)
    # figure our how many tree states we need
    n_tree_states = 1;
    i = 1;
    current_reward = Float64(start_reward);
    while current_reward > .5
        next_reward = current_reward*decrement
        next_reward = current_reward - 1; # decrement
        prepend!(Rtree,next_reward)
        #global i =i+1
        #global current_reward = next_reward
        current_reward = next_reward

    end
    n_tree_states = length(Rtree);
    #n_travel_states = Int64(round(copy(n_tree_states)*prop_travel));
    n_states = n_tree_states + n_travel_states;
    #plot(y=Rs,Geom.line)

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
    policy = ones(n_states);
    sp1_diff = 100;
    max_pol_change = 100.;
    policy = zeros(n_states);

    while max_pol_change >  .01
        iter += 1
        last_V_pi = copy(V_pi);
        last_sp1 = copy(sp1);
        ref_val = copy(V_pi[6])
        last_pol = copy(policy);
        for i = n_states:-1:1
            Q1 =  Rs[next_state[i,1]] + last_V_pi[next_state[i,1]] - ref_val;
            Q2 =  Rs[next_state[i,2]] + last_V_pi[next_state[i,2]] - ref_val;
            V_pi[i] = maximum([Q1 Q2])
            policy[i] = argmax([Q1 Q2])[2]
        end
        #print(policy)
        #global sp1 = maximum(V_pi - last_V_pi) - minimum(V_pi - last_V_pi)
        #sp1 = maximum(V_pi - last_V_pi) - minimum(V_pi - last_V_pi)
        sp1 = spannorm_dist(V_pi, last_V_pi)
        #sp1_diff = abs.(sp1 - last_sp1)
        sp2 = argmax(V_pi - last_V_pi);

        #max_pol_change = maximum(abs.(policy - last_pol));

        if mod(iter,100) == 0
            println(sp1)
            println(sp2)
            println(policy[1])
        end
    end
    #return policy, Rs
    res_df = DataFrame(state=1:n_states, pol = policy, R = Rs,
                n_travel = n_travel_states*ones(n_states),
                start_R = start_reward*ones(n_states),
                decrement = decrement*ones(n_states));
    return res_df

end

data = solve_policy(10.,.5,10)


for start_reward = [10. 20.]
    for decrement = [.7 .9]
        for n_travel_states = [5 20]
            println(start_reward, decrement, n_travel_states)
            global data = [data; solve_policy(start_reward,decrement,n_travel_states)]
        end
    end
end


# solve this for different
#for ...
    data = [data; solve_policy(start_reward,decrement,n_travel_states)]
end



start_reward = 10.;
decrement = .9;

# proportion of travel astates
n_travel_states = 10;

res_df = solve_policy(start_reward,decrement,n_travel_states)
#n_states = length(R);
#res_df = DataFrame(state=1:n_states, pol = pol, R = R, #
#            n_travel = n_travel_states*ones(n_states),
#            start_R = start_reward*ones(n_states),
#            decrement = decrement*ones(n_states));



print(policy)

plot(y=policy)
exit_state = maximum(findall(policy[1:7].==2))
print(exit_state)
# get exit threshold -- minimum value where 2 is best response
