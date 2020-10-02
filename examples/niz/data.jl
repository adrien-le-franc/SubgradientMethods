# developed with julia 1.4.2
#
# data processing for NIZ


using LinearAlgebra
using DataFrames
using Dates


function extract_pv_values_and_midnight_forecast(raw_data::DataFrame)

    days = DataFrame()

    for (k, row) in enumerate(eachrow(raw_data))

        if k >= size(raw_data, 1) - 96
            break
        end

        timestamp = row[:timestamp]

        if Dates.Time(timestamp) != Dates.Time(0, 0, 0)
            continue
        end

        today = DataFrame(timestamp=Dates.DateTime[], quarter=Int64[], pv=Float64[],
        	prediction_at_00_00=Float64[])

        for i in 0:95

            timing = timestamp + i*Dates.Minute(15)
            pv = raw_data[k+i+1, :actual_pv]

            step = string(i)
            if i < 10
                step = "0"*step
            end
            prediction = row[Symbol("pv_"*step)]

            push!(today, [timing, i+1, pv, prediction])

        end

        days = vcat(days, today)

    end
    
    return days
    
end

function fit_pv_ar_1_model(data::DataFrame)

    weights = zeros(96, 2)

    for m in 1:96

        if m == 1
        	# avoid fitting AR from 23:45 to 00:00
        	# using that pv is null at this time
        	weights[1, :] = [0., 0.]
        else
            x = data[data.quarter .== m-1, :]
            y = data[data.quarter .== m, :]

	        x = x[:, :pv] - x[:, :prediction_at_00_00]
	        y = y[:, :pv] - y[:, :prediction_at_00_00]

	        x = hcat(x, ones(size(x)))
	        weights[m, :] = pinv(x'*x)*x'*y
	    end

    end
    
    return weights
    
end

function collect_prediction_error_data(data::DataFrame, ar_model_weights::Array{Float64,2})

    n_days = size(data[data.quarter .== 1, :], 1)
    train_data = zeros(96, n_days)

    for m in 1:96

        if m == 1

            previous_error = zeros(n_days) # pv is null at 23:45
            previous_error = hcat(previous_error, ones(n_days))
            predicted_error = previous_error*ar_model_weights[1, :]
            effective_error = data[data.quarter .== 1, :pv] - data[data.quarter .== 1,
            	:prediction_at_00_00]

        else

            previous_error = data[data.quarter .== m-1, :pv] - data[data.quarter .== m-1,
            	:prediction_at_00_00] 
            previous_error = hcat(previous_error, ones(n_days))
            predicted_error = previous_error*ar_model_weights[m, :]
            effective_error = data[data.quarter .== m, :pv] - data[data.quarter .== m, 
            	:prediction_at_00_00]

        end

        delta_prediction = predicted_error - effective_error
        train_data[m, :] = delta_prediction

    end
    
    return train_data
    
end