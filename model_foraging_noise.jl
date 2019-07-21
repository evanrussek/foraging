using Gadfly
using LinearAlgebra
using Distances

## this just does foraging, constant travel time...
# maybe make it slightly more similar to the task, w/ number of states

n_states = 12;

next_state = transpose([1 1 2 3 4 5 6 8 9 10 11 12;
                        8 8 8 8 8 8 8 9 10 11 12 7]);

## 12 elements with reward for each state
Rs = [0 1 2 3 4 5 6 0 0 0 0 0];


# run value iteration
V_pi = zeros(n_states)
sp1 = 100;
iter = 0;
policy = ones(n_states);
sp1_diff = 100;
max_pol_change = 100.;
policy = zeros(n_states);

other_action = [2 1];

#while max_pol_change >  .01
    iter += 1
    last_V_pi = copy(V_pi);
    last_sp1 = copy(sp1);
    last_pol = copy(policy);
    for i = 1:n_states
        Q1 =  .95*Rs[next_state[i,1]] + .05*Rs[next_state[i,1]] +
                .95*last_V_pi[next_state[i,1]] + .05*last_V_pi[next_state[i,2]];
        Q2 =  .95*Rs[next_state[i,2]] + .05*Rs[next_state[i,2]] +
                .95*last_V_pi[next_state[i,2]] + .05*last_V_pi[next_state[i,1]];

        V_pi[i] = maximum([Q1 Q2])
        policy[i] = argmax([Q1 Q2])[2]
    end
        #print(policy)
        #global sp1 = maximum(V_pi - last_V_pi) - minimum(V_pi - last_V_pi)
        #sp1 = maximum(V_pi - last_V_pi) - minimum(V_pi - last_V_pi)
        sp1 = spannorm_dist(V_pi, last_V_pi)
        #sp1_diff = abs.(sp1 - last_sp1)
