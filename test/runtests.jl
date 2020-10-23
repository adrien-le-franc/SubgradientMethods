# developed with julia 1.4.2
#
# tests for SubgradientMethods package

using SubgradientMethods
SM = SubgradientMethods

using Test
using Dates

@testset "SubgradientMethods" begin

	mutable struct TestOracle <: SM.AbstractOracle end
	function SM.call_oracle!(oracle::TestOracle, variable::Array{Float64,1}, k::Int64)
		return sum(abs.(variable)), sign.(variable)
	end

	struct HyperCubeProjection <: SM.AbstractProjection 
		project::Function
	end

	function HyperCubeProjection()
		f(x::Array{Float64,1}) = max.(min.(1., x), -1.)
		return HyperCubeProjection(f) 
	end

	oracle = TestOracle()
	projection = HyperCubeProjection()
	step_size(k::Int64) = 1/k
	parameters = SM.Parameters(rand(10), 50, step_size, Second(10), 0.01)
    output = SM.optimize!(oracle, projection, parameters)

    @test output.iteration <= 50

end