# developed with julia 1.4.2


call_oracle!(oracle::AbstractOracle, variable::Array{Float64,1}, k::Int64) = nothing

function make_step(variable::Array{Float64,1}, projection::AbstractProjection, 
	parameters::Parameters, subgradient::Array{Float64,1}, k::Int64)

	variable =  variable .- parameters.step_size(k)*subgradient
	return projection.project(variable)

end

function stopping_test(oracle::AbstractOracle, output::Output, parameters::Parameters)

	if (output.elapsed > parameters.max_time || output.subgradient_norm < parameters.epsilon)
		return true
	else
		return false
	end

end

function optimize!(oracle::AbstractOracle, projection::AbstractProjection, 
	parameters::Parameters)
	
	variable = initialization(parameters)
	output = Output(variable)
	starting_time = now()

	for k in steps(parameters)

		t = @elapsed objective, subgradient = call_oracle!(oracle, variable, k)
		variable = make_step(variable, projection, parameters, subgradient, k)
		update_output!(output, k, objective, subgradient, variable, starting_time, t)

		if stopping_test(oracle, output, parameters)
			return output
			break
		end

	end

	return output

end