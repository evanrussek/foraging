using Gadfly
using LinearAlgebra

## this just does foraging, constant travel time...
# maybe make it slightly more similar to the task, w/ number of states

n_states = 12;

next_state = transpose([1 1 2 3 4 5 6 8 9 10 11 12;
                        8 8 8 8 8 8 8 9 10 11 12 7]);

## 12 elements with reward for each state
R = [0 1 2 3 4 5 6 0 0 0 0 0];


function evaluate_policy(policy,R,next_state)

    T = zeros(12,12)

    for i = 1:12
        T[i,next_state[i,Int64(policy[i])]] = .9;
        T[i,7] += .1
    end

    start_dist = ones(n_states)./n_states;
    std_mtx = T^1000;
    steady_state_dist = transpose(std_mtx)*start_dist; # normally wind up in 1, unless elsewhere
    rho_pi = Rs*steady_state_dist;

    I_mtx = Matrix{Int64}(I,12,12);
    V = inv(I_mtx[1:12,1:12] - T[1:12,1:12])*(R[1:12] .- rho_pi)
    return V
end

function improve_policy(V,R,next_state)
    policy = zeros(12);
    for i = 1:12
        Q = zeros(2);
        for a = 1:2
            Q[a] = R[next_state[i,a]] + V[next_state[i,a]]
        end
        policy[i] = argmax(Q)
    end
    return policy
end

policy_hist = zeros(200,12);
V_hist = zeros(200,12);
policy = ones(12);
p_change = 100;
for x = 1:200
    V = evaluate_policy(policy,R,next_state)
    policy_hist[x,:] = policy;
    V_hist[x,:] = V;
    new_policy = improve_policy(V,R,next_state)
    global p_change = maximum(abs.(new_policy - policy))
    global policy = copy(new_policy)
end
