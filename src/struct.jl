# developed with julia 1.4.2
#
# struct for optimization algorithms


abstract type AbstractOracle end
abstract type AbstractProjection end


struct Parameters
	initial_variable::Array{Float64,1}
	max_iterations::Int64
	step_size::Function
	max_time::Period
	epsilon::Float64
end

initialization(parameters::Parameters) = parameters.initial_variable
steps(parameters::Parameters) = 1:parameters.max_iterations


mutable struct Output
	value::Float64
	variable::Array{Float64,1}
	iteration::Int64
	subgradient_norm::Float64
	final_iteration::Int64
	elapsed::Period
end

Output(variable::Array{Float64,1}) = Output(Inf, variable, 0, Inf, 0, Second(0))

function update_output!(output::Output, k::Int64, objective::Float64, 
	subgradient::Array{Float64,1}, variable::Array{Float64,1}, starting_time::DateTime)

	if objective < output.value

		output.value = objective
		output.variable = variable
		output.iteration = k
		output.subgradient_norm = sqrt(subgradient'*subgradient)

	end

	output.final_iteration = k
	output.elapsed = now() - starting_time

end