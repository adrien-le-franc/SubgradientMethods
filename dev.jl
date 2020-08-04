# developed with julia 1.4.2


using StoOpt
using JLD

include("ar.jl")


## multistage problem ##

# physical values

const peak_power = 1000. # kW
const price = 0.15 # EUR/kWh
const penalty_coefficient = 0.5

const max_battery_capacity = 1000. # kWh
const max_battery_power = 1000. # kW
const dt = 0.25 # h
const rho_c = 0.95
const rho_d = 0.95

# states

const dx = 0.1
const dg = 0.1
const states = Grid(0:dx:1, 0:dg:1, enumerate=true)

# controls

const du = 0.1
const controls = Grid(-1:du:1)

# noises

data = load("/home/SubgradientMethods/ausgrid.jld", "data")

train = 1:60
test = 61:90

weights = fit_pv_ar_1_model(data[:, train])
noise_data = pv_prediction_error_data(data, weights)
const noises = StoOpt.Noises(noise_data, 10)

# costs

function final_cost(state::Array{Float64,1}) 
	-price*state*max_battery_capacity*01 # to be tuned ?
end

# dynamics

function dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
	noise::Array{Float64,1})

	scale_factor = max_battery_power*dt/max_battery_capacity
	normalized_exchanged_power = rho_c*max(0., control[1]) - max(0., -control[1])/rho_d
    soc = state[1] + normalized_exchanged_power

    predicted_pv = min(max(weights'*[state[2], 1.] + noise[1], 0.), 1.)

    return [soc, predicted_pv]

end

# horizon

const horizon = 48

# model 

model = StoOpt.SDP(states, 
		controls,
		noises,
		nothing,
		dynamics,
		final_cost,
		horizon)


function initialize_sdp_model!(model::StoOpt.SDP, parameter::Array{Float64,2})

	function parametric_stage_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		noise::Array{Float64,1})

		power_production = min(max(weights'*[state[2], 1.] + noise[1], 0.), 1.)*peak_power
		power_delivery = power_production - control[1]*max_battery_power # in kW

		return -price*dt*parameter[t] + penalty_coefficient*price*dt*abs(power_delivery 
			- parameter[t])

	end

	model.cost = parametric_stage_cost

end


mutable struct ParametricMultistageOracle <: SubgradientMethods.AbstractOracle
	model::StoOpt.SDP
	value_functions::StoOpt.ArrayValueFunctions
end


function SubgradientMethods.call_oracle(oracle::ParametricMultistageOracle, 
	variable::Array{Float64,1}, k::Int64)
	
	initialize_sdp_model!()

	value_functions = StoOpt.compute_value_functions(oracle.model)

	subgradient = compute_subgradient()

end