# developed with julia 1.4.2
#
# struct for optimization algorithms


abstract type AbstractOracle end
abstract type AbstractProjection end


struct Parameters
	initial_variable::Array{Float64,1}
	step_size::Function
	max_iterations::Int64
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
	elapsed_per_oracle_call::Array{Float64,1}
	all_values::Array{Float64,1}
end

Output(variable::Array{Float64,1}) = Output(Inf, variable, 0, Inf, 0, Second(0), Float64[], Float64[])

function update_output!(output::Output, k::Int64, objective::Float64, 
	subgradient::Array{Float64,1}, variable::Array{Float64,1}, 
	starting_time::DateTime, elapsed_per_oracle_call::Float64)

	if objective < output.value

		output.value = objective
		output.variable = variable
		output.iteration = k
		output.subgradient_norm = norm(subgradient)

	end

	push!(output.elapsed_per_oracle_call, elapsed_per_oracle_call)
	push!(output.all_values, objective)

	output.final_iteration = k
	output.elapsed = now() - starting_time

end

