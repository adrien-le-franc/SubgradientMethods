# developed with julia 1.4.2
#
# PV unit toy model 


using ParametricMultistage
PM = ParametricMultistage

using StoOpt
using JLD

directory = @__DIR__
package_directory = joinpath(directory, "..", "..")


# physical values

peak_power = 1000. # kW
price = 0.15 # EUR/kWh
penalty_coefficient = 2.

max_battery_capacity = 1000. # kWh
max_battery_power = 1000. # kW
dt = 0.25 # h
rho_c = 0.95
rho_d = 0.95

# horizon

horizon = 48

# states

dx = 0.1
dg = 0.1
states = PM.States(horizon, 0:dx:1., 0:dg:1.)

# controls

du = 0.1
controls = PM.Controls(horizon, -1:du:1)

# noises

data = load(joinpath(package_directory, "data", "data.jld"))
noises = StoOpt.Noises(data["error_data"], 10)

# indicator of control set subgradient

function indicator_of_control_set_subgradient(t::Int64, state::Array{Float64,1},
	control::Array{Float64,1}, parameter::Array{Float64,1})
	control = control[1]
	soc = state[1]
	if control*rho_c*dt*max_battery_power + (soc - 1)*max_battery_capacity == 0.
		return vcat([1., 0.], zeros(horizon))
	elseif -control*dt*max_battery_power/rho_d - soc*max_battery_capacity == 0.
		return vcat([-1., 0.], zeros(horizon))
	else
		return zeros(2+horizon)
	end
end

# dynamics

weights = data["weights"]

function dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
	noise::Array{Float64,1}, parameter::Array{Float64,1})

	scale_factor = max_battery_power*dt/max_battery_capacity
	normalized_exchanged_power = rho_c*max(0., control[1]) - max(0., -control[1])/rho_d
    soc = state[1] + normalized_exchanged_power*scale_factor

    predicted_pv = min(max(weights[t, :]'*[state[2], 1.] + noise[1], 0.), 1.)

    return [soc, predicted_pv]

end

# dynamics jacobian

function dynamics_jacobian(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
	parameter::Array{Float64,1})
	
	Axx = sparse([1. 0. ; 0. weights[t, 1]])
	Axp = spzeros(2, horizon)
	Apx = spzeros(horizon, 2)
	App = I

	return [Axx Axp ; Apx App]

end

# costs

function stage_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
	noise::Array{Float64,1}, parameter::Array{Float64,1})

	power_production = min(max(weights[t, :]'*[state[2], 1.] + noise[1], 0.), 1.)*peak_power
	power_delivery = power_production - control[1]*max_battery_power # in kW

	return -price*dt*parameter[t] + penalty_coefficient*price*dt*abs(power_delivery 
		- parameter[t])

end

function final_cost(state::Array{Float64,1}, parameter::Array{Float64,1})
	-price*state[1]*max_battery_capacity 
end

# cost subgradients

function stage_cost_subgradient(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
	noise::Array{Float64,1}, parameter::Array{Float64,1}) 

	power_production = min(max(weights[t, :]'*[state[2], 1.] + noise[1], 0.), 1.)*peak_power
	power_delivery = power_production - control[1]*max_battery_power # in kW
	subgradient_absolute_delivery_gap = sign(power_delivery - parameter[t])

	subgradient_generated_power = [0., price*dt*penalty_coefficient*(
		weights[t, 1]*peak_power*subgradient_absolute_delivery_gap)]

	subgradient_parameter = zeros(horizon)
	subgradient_parameter[t] = -price*dt*(1. + 
		penalty_coefficient*subgradient_absolute_delivery_gap)

	return vcat(subgradient_generated_power, subgradient_parameter)

end

function final_cost_subgradient(state::Array{Float64,1}, parameter::Array{Float64,1})
	return vcat([-price*max_battery_capacity, 0.], zeros(horizon))
end

# model

model = PM.ParametricMultistageModel(
	states,
	controls,
	noises,
	dynamics,
	stage_cost,
	final_cost,
	horizon,
	zeros(horizon),
	zeros(2),
	indicator_of_control_set_subgradient,
	dynamics_jacobian,
	stage_cost_subgradient,
	final_cost_subgradient)

# testing scenarios

#scenarios = data["scenarios"]