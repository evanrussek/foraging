using QuadGK

function opportunity_cost(R_hat, lag)
    return lag .* R_hat
end

function vigor_cost(press_cost, lag)
    return press_cost ./ lag
end

function lag_cost(R_hat, press_cost, lag)
    return opportunity_cost(R_hat, lag) .+ vigor_cost(press_cost, lag)
end

function lag_lp_unnorm(beta, R_hat, press_cost, lag)
    #  un-normalized log probability....
    return beta.*-1 .* lag_cost(R_hat, press_cost, lag)
end


function lag_prob_unnorm(beta, R_hat, press_cost, lag)
    return exp.(beta.*-1 .* lag_cost(R_hat, press_cost, lag))
end

function lag_prob(beta, R_hat, press_cost, lag)
    c = x -> lag_prob_unnorm(beta, R_hat, press_cost, x)
    lag_prob_unnorm(beta, R_hat, press_cost, lag) ./ quadgk(c, 0, Inf, rtol=1e-3)[1]
end

# uses exp-normalize trick
function sample_lag(lag_beta, R_hat, press_cost, lag_range)
    lp_un_vec = lag_lp_unnorm(lag_beta, R_hat, press_cost, lag_range);
    prob_un_vec = exp.(lp_un_vec .- maximum(lp_un_vec))
    cdf_un_vec = cumsum(prob_un_vec)
    z = cdf_un_vec[end]
    u = rand()
    lag_idx = searchsortedfirst(cdf_un_vec,u*z)
    return lag_range[lag_idx]
end

R_hat = .5
press_cost = 10.
beta = 1.

fns_to_plot = [(x) -> opportunity_cost(R_hat, x), (y) -> vigor_cost(press_cost, y)]
b = x-> opportunity_cost(R_hat, x)
c = y -> vigor_cost(press_cost, y)
d = x -> lag_cost(R_hat, press_cost, x)
e =  x -> lag_prob_unnorm(beta, R_hat, press_cost, x)
f =  x -> lag_prob(beta, R_hat, press_cost, x)

xmin = .5
xmax = 20.

p1 = plot([b,c,d], .4 , xmax, Theme(line_width = 2pt))
p2 = plot(f, 0,xmax, Theme(line_width = 2pt))

hstack([p1, p2])
