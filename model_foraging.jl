# tree states are 7,6,5,4,3,2,1 - 1 leads to itself
# travel states are 8,9,10,11,12

using Gadfly

## this just does foraging, constant travel time...
# maybe make it slightly more similar to the task, w/ number of states

n_states = 12;

next_state = transpose([1 2 2 3 4 5 6 8 9 10 11 12;
                        8 8 8 8 8 8 8 9 10 11 12 7]);

## 12 elements with reward for each state
Rs = [0 1 2 3 4 5 6 0 0 0 0 0];

# relative value iteration
V_pi = zeros(12)
#last_V_pi = copy(V_pi)
sp1 = 100;
iter = 0;
policy = zeros(12);


#### polcy evaluation
for i = 2:4
    X_mtx[i,i] = 1; X_mtx[i,1] = 1; X_mtx[i,next_state[i]] = -1;
    b[i] = Rs[next_state[i]];
end
X_mtx[1,1] = 1; X_mtx[1,next_state[1]] = -1; b[1] = Rs[next_state[1]];
V = inv(X_mtx)*b

rho_pi = V[1];
V_pi

while sp1 > .001
    global iter += 1
    println("iter: ", iter)
    global last_V_pi = copy(V_pi);
    for i = 1:n_states
        Q1 =  Rs[next_state[i,1]] + last_V_pi[next_state[i,1]];
        Q2 =  Rs[next_state[i,2]] + last_V_pi[next_state[i,2]];
        V_pi[i] = maximum([Q1 Q2])
        policy[i] = argmax([Q1 Q2])[2]
    end
    #print(policy)
    global sp1 = maximum(V_pi - last_V_pi) - minimum(V_pi - last_V_pi)
end

print(policy)

plot(y=policy)
exit_state = maximum(findall(policy[1:7].==2))
print(exit_state)
# get exit threshold -- minimum value where 2 is best response
