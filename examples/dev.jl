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

	if k % 10 == 0
		println("step $(k): $(cost_to_go)")
	end
	
	return cost_to_go, subgradient

end

oracle = ParametricMultistageOracle(model)

# projection

struct DummyProjection <: SubgradientMethods.AbstractProjection end
projection = DummyProjection()

# parametrers

step_size(k::Int64) = 1/k
parameters = SubgradientMethods.Parameters(zeros(48), 200, step_size)

# let's do it !

SubgradientMethods.optimize(oracle, projection, parameters)

