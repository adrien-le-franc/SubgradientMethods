# developed with julia 1.4.2


call_oracle!(oracle::AbstractOracle, variable::Array{Float64,1}, k::Int64) = nothing

function make_step(variable::Array{Float64,1}, projection::AbstractProjection, 
	parameters::Parameters, subgradient::Array{Float64,1}, k::Int64)

	return variable .- parameters.step_size(k)*subgradient

	# projection ?

end

function stopping_test(variable::Array{Float64,1}, objective::Float64, 
	parameters::Parameters, k::Int64)

	if k == parameters.max_step

		println("final value: $(objective)")
		println("final variable: $(variable)")
		return true

	end

	return false

end

function optimize(oracle::AbstractOracle, projection::AbstractProjection, 
	parameters::Parameters)
	
	variable = initialization(parameters)

	for k in steps(parameters)

		objective, subgradient = call_oracle!(oracle, variable, k)

		variable = make_step(variable, projection, parameters, subgradient, k)

		if stopping_test(variable, objective, parameters, k)
			break
		end

	end

end