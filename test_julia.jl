using DataFrames
using GLM
using PyPlot
using Statistics

# compute the Q tables by dynamic programming

horizon = 20
# rl, ul, rr, ur, action
# action is 1 (L) or 2 (R)

Q = zeros((horizon+1,horizon+1,horizon+1,horizon+1,2))

# last step is just Q = 0 so already initialized; work backward from there

for step = horizon:-1:1
    for rl=1:step
        for ul = 1:(step + 1 - rl)
            for rr = 1:(step + 2 - rl - ul)
                ur = step + 3 - rl - ul - rr
                mur = rr / (ur + rr)
                mul = rl / (ul + rl)
                Q[rl,ul,rr,ur,1] = mul * (1 + maximum(Q[rl+1,ul,rr,ur,:])) +
                                     (1 - mul) * (0 + maximum(Q[rl,ul+1,rr,ur,:]))
                Q[rl,ul,rr,ur,2] = mur * (1 + maximum(Q[rl,ul,rr+1,ur,:])) +
                                     (1 - mur) * (0 + maximum(Q[rl,ul,rr,ur+1,:]))
            end
        end
    end
end

function playgame(horizon,Q)

    rl = ul = rr = ur = 1

    p = rand(2)

    choice = zeros(Int64,horizon)
    reward = zeros(Int64,horizon)
    rls = zeros(Int64,horizon)
    uls = zeros(Int64,horizon)
    rrs = zeros(Int64,horizon)
    urs = zeros(Int64,horizon)

    for step = 1:horizon
        choice[step] = (Q[rl,ul,rr,ur,1] > Q[rl,ul,rr,ur,2]) ? 1 : 2
        reward[step] = Int64(rand() < p[choice[step]])

        rls[step] = rl
        uls[step] = ul
        rrs[step] = rr
        urs[step] = ur

        if (choice[step] == 1 && reward[step] == 1)
            rl += 1
        elseif (choice[step] == 1 && reward[step] == 0)
            ul += 1
        elseif (choice[step] == 2 && reward[step] == 1)
            rr += 1
        elseif (choice[step] == 2 && reward[step] == 0)
            ur += 1
        else
            error("surprise!")
        end
    end

    return (DataFrame(step=1:horizon,choice=choice,reward=reward,rl=rls,ul=uls,rr=rrs,ur=urs))
end

data = playgame(horizon,Q)


for i = 1:99
    global data = [data; playgame(horizon,Q)]
end

# expected value at the start of the game, vs actual obtained value
# not sure why these don't match

display(Q[1,1,1,1,1])
#mean(data[data[:step].==20,:rl]+data[data[:step].==20,:rr])
mean(data[:reward])*20

data[:pl] = data[:rl] ./ (data[:rl] + data[:ul])
data[:pr] = data[:rr] ./ (data[:rr] + data[:ur])

data[:vl] = data[:rl] .* data[:ul] ./ ((data[:rl] .* data[:ul]).^2 .* (data[:rl] + data[:ul] .+ 1))
data[:vr] = data[:rr] .* data[:ur] ./ ((data[:rr] .* data[:ur]).^2 .* (data[:rr] + data[:ur] .+ 1))

data[:meanRminusL] = data[:pr] .- data[:pl]
data[:sdRminusL] =  sqrt.(data[:vr]) .- sqrt.(data[:vl])

data[:chooseR] = data[:choice] .- 1;

scatter(data[:meanRminusL],data[:sdRminusL])
