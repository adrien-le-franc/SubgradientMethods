
function NIZx.compute_control(controller::Sdp, information::NIZx.Information)

	if information.m == 1
		grid_scale_prediction_error = 0.0
	else	
		observed_pv = information.pv[end]
		predicted_pv = information.initial_forecast[information.m-1]
		prediction_error = observed_pv - predicted_pv # in kW
		grid_scale_prediction_error = min(max(prediction_error/NIZx.peak_power, -0.5), 0.5)
	end

	state = [information.soc, grid_scale_prediction_error]

	return StoOpt.compute_control(controller.model,
		information.m,
		state, 
		StoOpt.RandomVariable(controller.model.noises, information.m),
		controller.value_functions)[1]

end