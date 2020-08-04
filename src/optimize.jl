# developed with julia 1.4.2


function optimize(oracle::AbstractOracle, projection::AbstractProjection, 
	parameters::Parameters)
	
	variable = initialization(parameters)

	for k in steps(parameters)

		objective, subgradient = call_oracle(oracle, variable, k)
		make_step!(variable, projection, parameters)

		if stopping_test()
			break
		end

	end

end


call_oracle()
make_step!()
stopping_test()