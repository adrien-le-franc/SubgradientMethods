# developed with julia 1.4.2


using SubgradientMethods
SM = SubgradientMethods

using ParametricMultistage
PM = ParametricMultistage


include("model.jl")


# oracle

mutable struct ParametricMultistageOracle <: SubgradientMethods.AbstractOracle
	model::ParametricMultistage.ParametricMultistageModel
end

function set_variable!(oracle::ParametricMultistageOracle, variable::Array{Float64,1})
	oracle.model.parameter = variable
end

function SubgradientMethods.call_oracle!(oracle::ParametricMultistageOracle, 
	variable::Array{Float64,1}, k::Int64)

	set_variable!(oracle, variable)
	subgradient, cost_to_go = PM.compute_a_subgradient(oracle.model, true)

	if k % 1 == 0
		println("step $(k): $(cost_to_go)")
		println("subgradient[24]: $(subgradient[24])")
		println("variable[24]: $(variable[24])")
	end
	
	return cost_to_go, subgradient

end

oracle = ParametricMultistageOracle(model)

# projection

struct DummyProjection <: SubgradientMethods.AbstractProjection end
struct HyperCubeProjection <: SubgradientMethods.AbstractProjection 
	project::Function
	coefficient::Float64
end

function HyperCubeProjection(coefficient::Float64)
	f(x::Array{Float64,1}) = max.(min.(coefficient, x), 0.)
	return HyperCubeProjection(f, coefficient) 
end

#projection = DummyProjection()
projection = HyperCubeProjection(peak_power)

# parametrers

step_size(k::Int64) = 2000/k
parameters = SubgradientMethods.Parameters(zeros(48), 50, step_size)

# let's do it !

SubgradientMethods.optimize(oracle, projection, parameters)

