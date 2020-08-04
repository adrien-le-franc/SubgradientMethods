# developed with julia 1.4.2

using LinearAlgebra


function fit_pv_ar_1_model(data::Array{Float64, 2})

	weights = zeros(48, 2)

	for t in 1:48

	   if t == 1
           # avoid fitting AR from 23:30 to 00:00
           # using that pv is null at this time
           weights[1, :] = [0., 0.]
	   else
	       x = data[t-1, :]
	       y = data[t, :]
           x = hcat(x, ones(size(x)))
           weights[t, :] = pinv(x'*x)*x'*y
       end

	end

	return weights

end

function pv_prediction_error_data(data::Array{Float64,2}, ar_model_weights::Array{Float64,2})

   n_days = size(data, 2)
   train_data = zeros(48, n_days)

   for t in 1:48

       if t == 1

           previous_pv = zeros(n_days) # pv is null at 23:45
           previous_pv = hcat(previous_pv, ones(n_days))
           predicted_pv = previous_pv*ar_model_weights[1, :]
           effective_pv = data[1, :]

       else

           previous_pv = data[t-1, :] 
           previous_pv = hcat(previous_pv, ones(n_days))
           predicted_pv = previous_pv*ar_model_weights[t, :]
           effective_pv = data[t, :]

       end

       delta_prediction = predicted_pv - effective_pv
       train_data[t, :] = delta_prediction

   end
   
   return train_data
   
end