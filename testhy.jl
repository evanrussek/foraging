using Gadfly
using DataFrames
using Distances
using DataFramesMeta
using CategoricalArrays

using Statistics

# %
function time_cost(lag, unit_cost)
    return unit_cost/lag
end

#%%
time_cost_plot = plot([x -> time_cost(x,2),
    x -> time_cost(x,6),
    x -> time_cost(x,10)],
        .1,1,
        Guide.xlabel("lag"), Guide.ylabel("time cost"),
        Guide.colorkey(title="Unit Cost",labels = ["2", "6", "10"] ))

time_cost_plot

# % code cell
a = 1

# % code
b = 1

# %
b =  plot([x -> time_cost(x,2),
    x -> time_cost(x,6),
    x -> time_cost(x,10)],
        .1,1,
        Guide.xlabel("lag"), Guide.ylabel("time cost"),
        Guide.colorkey(title="Unit Cost",labels = ["2", "6", "10"] ))
draw(b)

# %%
p = plot(x=collect(1:10), y=sort(rand(10)))
display(p)

using Gadfly
plot(y=[1,2,3])
