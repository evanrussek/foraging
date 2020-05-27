using Gadfly # for plotting
using SpecialFunctions # for bessell
using QuadGK # numerical integration
using StatsFuns

lag_range = .001:.01:20

R = .1; C = 3.; B = 1.;
f = (lag) -> exp.(-B*(R*lag + C/lag))
plot(f,.01,100,Guide.ylabel("Unnormalized probability"), Guide.xlabel("Lag"))

R = .1; C = 3.; B = -1.;
f = (lag) -> exp.(-B*(R*lag + C/lag))
plot(f,-20,500,Guide.ylabel("Unnormalized probability"), Guide.xlabel("Lag"))

R = -.1; C = 3.; B = 1.;
f = (lag) -> exp.(-B*(R*lag + C/lag))
plot(f,-20,500,Guide.ylabel("Unnormalized probability"), Guide.xlabel("Lag"))


R = .1; C = -3.; B = 1.;
f = (lag) -> exp.(-B*(R*lag + C/lag))
plot(f,0,5000,Guide.ylabel("Unnormalized probability"), Guide.xlabel("Lag"))


# for a range of R,C, and B, run both of these.

using



function int_exp_lag_cost(R,C,B)

    #analytic integration of unnormalized lag probability
    # R is expected reward rate, C is unit cost, B is lag beta
    # requires SpecialFunctions pkg to use bessell function """

    return 2*sqrt(C/R)*besselk(1,2*B*sqrt(R*C))
end
# un-normalized lag probability
function u_lag_prob(R,C,B,this_lag)
    return -B.*(C./this_lag + R.*this_lag)
end

# normalization factor
function numerical_den(R,C,B)
    incr = .01;
    lag = 1e-4:incr:1000;
    return logsumexp(u_lag_prob(R,C,B,lag)) + log(incr);
end

R = 1e-20; C = 1e-20; B = 1e-20;
log(int_exp_lag_cost(R,C,B))
#numerical_den(R,C,B)
#log_lag_den_approx(R,C,B)

besselk(1,680)


# if R, C, or B approach 0
R < 1e-4 || C < 1e-4 || B < 1e-4 || 2*B*sqrt(R*C) > 680 || return numerical_den(R,C,B)
return log(int_exp_lag_cost(R,C,B))


    # call numerical bessell
# if input to bessell is too large...
2*B*sqrt(R*C) > 600
# call approximate bessell...


# this is an approximation and is wrong...
function log_lag_den_approx(R,C,B)
    # for log bessell:
    # https://stackoverflow.com/questions/32484696/natural-logarithm-of-bessel-function-overflow
    log_z = log(2) + log(B) + .5*log(R) + .5*log(C);
    log_bessell = .5*log(pi) + .5*log(2) - 1 - log_z;
    return log(2) + .5*log(C) - .5*log(R) + log_bessell;
end

# want a control flow to decide what to do

function int_exp_lag_cost(R,C,beta)
    return 2*sqrt(C/R)*besselk(1,2*beta*sqrt(R*C))
end

int2 = int_exp_lag_cost(R,C,B)

println(string("\nNumeric Int: ", int1,
        "\nAnalytic Int: ", int2,
        "\nDiff: ", abs(int1 - int2)))
