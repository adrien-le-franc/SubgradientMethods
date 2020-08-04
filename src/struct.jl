# developed with julia 1.4.2
#
# struct for optimization algorithms


abstract type AbstractOracle end


struct Parameters
	initial_variable::Array{Float64,1}
	max_step::Int64
	step_size::Function
end

initialization(parameters::Parameters) = parameters.initial_variable
steps(parameters::Parameters) = 1:parameters.max_step


abstract type AbstractProjection end

