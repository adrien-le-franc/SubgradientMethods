# developed with julia 1.4.2
#
# optimization loop


using SubgradientMethods
SM = SubgradientMethods

using CSV, DataFrames
using CodecZlib, Mmap

include("oracle.jl")

DIR = @__DIR__

data = CSV.File(transcode(GzipDecompressor, Mmap.mmap(joinpath(DIR, "pv.csv.gz")))) |> DataFrame

# optimization model 

oracle = NIZxOracle(data)
projection = PolytopeProjection()
step_size(k::Int64) = 30000/k
initial_parameter = projection.project(zeros(96))
parameters = SM.Parameters(initial_parameter, 10, step_size)

# test on a single day

pv_forecast = collect(data[1, 3:end])*peak_power
update_stage_cost!(oracle.model, oracle.weights, pv_forecast)

#SM.call_oracle!(oracle, zeros(96), 1) 

SM.optimize(oracle, projection, parameters)