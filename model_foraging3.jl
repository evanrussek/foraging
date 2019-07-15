
using Gadfly

## this just does foraging, constant travel time...
# maybe make it slightly more similar to the task, w/ number of states

n_states = 12;

next_state = transpose([1 1 2 3 4 5 6 8 9 10 11 12;
                        8 8 8 8 8 8 8 9 10 11 12 7]);

## 12 elements with reward for each state
Rs = .01*[0 1 2 3 4 5 6 0 0 0 0 0];

# relative value iteration

policy = [2 2 2 2 2 2 2 2 2 2 2 2];

# build matrix for policy stuff
X_mtx = zeros(Float16, n_states,n_states) # 5 unkn
b = zeros(Float16, n_states)
rho_idx = 1;

#### polcy evaluation
ref_state = 1;
states = collect(1:12)
filter!(states->states!=ref_state,states)

for i = states
    global X_mtx;
    X_mtx[i,i] += 1; X_mtx[i,ref_state] += 1;
    if next_state[i,policy[i]] != ref_state
        X_mtx[i,next_state[i,policy[i]]] += -1;
    end
    b[i] = Rs[next_state[i,policy[i]]];
end
X_mtx[ref_state,ref_state] += 1;
if next_state[ref_state, policy[1]] != ref_state
    X_mtx[1,next_state[1,policy[6]]] += -1;
end
b[1] = Rs[next_state[6, policy[6]]];
V = pinv(X_mtx)*b



rho_pi = V[1];
V_pi
