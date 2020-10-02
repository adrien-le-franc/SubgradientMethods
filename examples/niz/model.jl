# developed with julia 1.4.2
#
# NIZ PV unit model


using ParametricMultistage
PM = ParametricMultistage

using StoOpt # ??
using DataFrames

using SparseArrays
using LinearAlgebra

#include("data.jl")


# PV unit physical values

global const peak_power = 1000. # kW
global const max_battery_capacity = 1000. # kWh
global const max_battery_power = 1000. # kW

global const max_price = 0.2 # EUR/kWh
global const min_price = 0.15 # EUR/kWh
global const prices = vcat(ones(76)*min_price,
	ones(8)*max_price,
	ones(12)*min_price)

global const dt = 0.25 # h
global const horizon = 96

# cost function from CRE regulation

function cost_function(t::Int64, engaged_power::Float64, delivered_power::Float64)
    
    if delivered_power < engaged_power - 0.05*peak_power
        
        bound = engaged_power - 0.05*peak_power
        
        return prices[t]*dt/peak_power *((delivered_power - (bound))^2 - 
            0.2*peak_power*(delivered_power - bound) - delivered_power*peak_power)
        
    elseif delivered_power > engaged_power + 0.05*peak_power
        
        return 0.0
        
    else
        
        return -prices[t]*dt*delivered_power
        
    end
    
end

# model

function niz_pv_unit_model(noises::StoOpt.Noises, weights::Array{Float64,2})

	# states

	dx = 0.1
	dg = 0.05
	states = PM.States(horizon, 0:dx:1, -0.5:dg:0.5)

	# controls

	du = 0.025
	controls = PM.Controls(horizon, -1:du:1)
	
	# indicator of control set subgradient

	function indicator_of_control_set_subgradient(t::Int64, state::Array{Float64,1},
		control::Array{Float64,1}, parameter::Array{Float64,1})

		control = control[1]*max_battery_power
		soc = state[1]
		subgradient = zeros(2+horizon)

		if control*rho_c*dt + (soc - 1)*max_battery_capacity == 0.
			subgradient[1] = 0.5
		elseif -control*dt/rho_d - soc*max_battery_capacity == 0.
			subgradient[1] = -0.5
		end

		if -control - parameter[t]*peak_power - 0.05*peak_power == 0.
			subgradient[2+t] = -0.5
		end

		return subgradient

	end

	# dynamics

	function predict_error(observed_error::Float64, model_weights::Array{Float64,1}, 
		noise::Array{Float64,1})
		return min(max(model_weights'*[observed_error, 1.] + noise[1], -0.5), 0.5)
	end

	rho_c = 0.95
	rho_d = 0.95

	function dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		noise::Array{Float64,1}, parameter::Array{Float64,1})

		scale_factor = max_battery_power*dt/max_battery_capacity
		power = control[1]
    	normalized_exchanged_power = rho_c*max(0., power) - max(0., -power)/rho_d 
		soc = state[1] + normalized_exchanged_power*scale_factor

	    prediction_error = predict_error(state[2], weights[t, :], noise)
	    
	    return [soc, prediction_error]
	end

	# dynamics jacobian

	matrix_xx(t::Int64) = sparse([1. 0. ; 0. weights[t, 1]])
	matrix_xp = spzeros(2, horizon)
	matrix_px = spzeros(horizon, 2)

	function dynamics_jacobian(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		parameter::Array{Float64,1}) 
		return [matrix_xx(t) matrix_xp ; matrix_px I] 
	end
	
	# costs (the stage cost is defined later when the dayahead forecast is known)

	function final_cost(state::Array{Float64,1}, parameter::Array{Float64,1})
		return -prices[end]*state[1]*max_battery_capacity
	end

	# costs subgradient 

	function final_cost_subgradient(state::Array{Float64,1}, parameter::Array{Float64,1})
		return vcat([-prices[end]*max_battery_capacity, 0.], zeros(horizon))
	end

	# model

	model = PM.ParametricMultistageModel(
	states,
	controls,
	noises,
	dynamics,
	nothing,
	final_cost,
	horizon,
	zeros(horizon),
	zeros(2),
	indicator_of_control_set_subgradient,
	dynamics_jacobian,
	nothing,
	final_cost_subgradient)

	return model

end

function update_stage_cost!(model::PM.ParametricMultistageModel, weights::Array{Float64,2},
	pv_forecast::Array{Float64,1})

	function predict_error(observed_error::Float64, model_weights::Array{Float64,1}, 
		noise::Array{Float64,1})
	    return min(max(model_weights'*[observed_error, 1.] + noise[1], -0.5), 0.5)
	end

	function curtailment(battery_power::Float64, pv_power::Float64, engagement_power::Float64)
		return min(max(pv_power - battery_power - engagement_power - 0.05*peak_power, 0.), 
			pv_power)
	end

	function stage_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
    noise::Array{Float64,1}, parameter::Array{Float64,1})

		predicted_error = predict_error(state[2], weights[t, :], noise)
		generated_power = pv_forecast[t] + predicted_error*peak_power # in kW
		curtailed_power = curtailment(control[1]*max_battery_power, generated_power, parameter[t])
		delivered_power = generated_power - control[1]*max_battery_power - curtailed_power

		return cost_function(t, parameter[t], delivered_power)

	end

	function stage_cost_subgradient(t::Int64, state::Array{Float64,1}, 
		control::Array{Float64,1}, noise::Array{Float64,1}, parameter::Array{Float64,1})

		predicted_error = predict_error(state[2], weights[t, :], noise)
		generated_power = pv_forecast[t] + predicted_error*peak_power # in kW
		upper_bound = parameter[t] + 0.05*peak_power
		lower_bound = parameter[t] - 0.05*peak_power

		control = control[1]*max_battery_power
		subgradient = zeros(2+horizon)

		if -upper_bound <= control < generated_power - upper_bound
			subgradient[2+t] = -prices[t]*dt 
		elseif generated_power - upper_bound <= control <= generated_power - lower_bound
			subgradient[2] = -prices[t]*dt*weights[t, 1]
		elseif generated_power - lower_bound < control <= max_battery_power
			subgradient[2] = prices[t]*dt*weights[t, 1]*(-1.2 + 2*(generated_power
				- control - lower_bound)/peak_power)
			subgradient[2+t] = prices[t]*dt*(0.2 - 2*(generated_power - control 
				- lower_bound)/peak_power)
		else
			error("control value $(control) is not admissible")
		end

		return subgradient

	end

	model.cost = stage_cost
	model.cost_subgradient = stage_cost_subgradient

end

function update_bounds!(model::PM.ParametricMultistageModel, parameter::Array{Float64,1})

	# controls 

	upper_bound = [[ub] for ub in ones(horizon)*max_battery_power]
	lower_bound = [[lb] for lb in max.(-1., -(parameter .+ 0.05*peak_power)/max_battery_power)]
	bounds = PM.Bounds(upper_bound, lower_bound)
	
	du = 0.025
	controls = PM.Controls(bounds, -1:du:1)

	model.controls = controls

end