# developed with julia 1.4.2
#
# oracle for optimal NIZ PV unit profile


using SubgradientMethods
SM = SubgradientMethods

using ParametricMultistage
PM = ParametricMultistage

using StoOpt

using JuMP
using CPLEX

include("model.jl")
include("data.jl")


# oracle 

mutable struct NIZxOracle <: SM.AbstractOracle
   model::PM.ParametricMultistageModel
   weights::Array{Float64,2}
end


function NIZxOracle(training_data::DataFrame)

	# noises (raw data is in MW)

	calibration_data = extract_pv_values_and_midnight_forecast(training_data)
	weights = fit_pv_ar_1_model(calibration_data)
	noise_data = collect_prediction_error_data(calibration_data, weights)
	noises = StoOpt.Noises(noise_data, 10) # StoOpt ??

	model = niz_pv_unit_model(noises, weights)

	return NIZxOracle(model, weights)

end

function set_variable!(oracle::NIZxOracle, variable::Array{Float64,1})
	oracle.model.parameter = variable
end

function SM.call_oracle!(oracle::NIZxOracle, variable::Array{Float64,1}, k::Int64)

	set_variable!(oracle, variable)

	# update bounds ?????
	update_bounds!(oracle.model, variable)

	subgradient, cost_to_go = PM.compute_a_subgradient(oracle.model, true)

	if k % 1 == 0
		println("step $(k): $(cost_to_go)")
		println("subgradient[50]: $(subgradient[50])")
		println("variable[50]: $(variable[50])")
	end
	
	return cost_to_go, subgradient

end

# projection

struct PolytopeProjection <: SM.AbstractProjection 
	project::Function
end

function PolytopeProjection()

	model = JuMP.Model(CPLEX.Optimizer)
	set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)

	regular_steps = vcat(collect(1:76), collect(85:96))
	peak_steps = collect(77:84)

	@variable(model, profile[1:96])
	@constraint(model, [t in regular_steps], -0.05*peak_power <= profile[t] <= peak_power)
	@constraint(model, [t in peak_steps], 0.2*peak_power <= profile[t] <= peak_power)
	@constraint(model, [t in 1:75], -0.075*peak_power <= profile[t] - profile[t+1] <= 0.075*peak_power)
	@constraint(model, [t in 76:84], -0.15*peak_power <= profile[t] - profile[t+1] <= 0.15*peak_power)
	@constraint(model, [t in 85:95], -0.075*peak_power <= profile[t] - profile[t+1] <= 0.075*peak_power)

	function project(p::Array{Float64,1})

		@objective(model, Min, (p - profile)'*(p - profile))
		optimize!(model)

		return value.(model[:profile])

	end

	return PolytopeProjection(project) 

end
