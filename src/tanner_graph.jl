struct TannerGraph
	nc::Int
	nv::Int
	v2c::Matrix{Int}
	c2v::Matrix{Int}
	soft_constraint_start::Int # Index of the start of soft constraints. These soft constraints are used to enforce correlations.
	check_neighbors::Vector{Vector{Int}}  # Precomputed neighbors for check nodes
	vertex_neighbors::Vector{Vector{Int}} # Precomputed neighbors for variable nodes
	function TannerGraph(parity_check_matrix::Matrix{Int}, soft_constraint_start::Int=r)
		nc, nv = size(parity_check_matrix)
		v2c = transpose(parity_check_matrix)
		c2v = deepcopy(parity_check_matrix)
		check_neighbors = [findall(x -> x == 1, c2v[c, :]) for c in 1:nc]
		vertex_neighbors = [findall(x -> x == 1, v2c[v, :]) for v in 1:nv]
		new(nc, nv, v2c, c2v, soft_constraint_start, check_neighbors, vertex_neighbors)
	end
end

function print_tanner_graph(tg::TannerGraph; io::IO=stdout)
	println(io, "Number of Check Nodes: ", tg.nc)
	println(io, "Soft Constraint starting row: ", tg.soft_constraint_start)
	println(io, "Number of Variable Nodes: ", tg.nv)
	println(io, "Variable-to-Check Matrix (v2c):\n", tg.v2c)
	println(io, "Check-to-Variable Matrix (c2v):\n", tg.c2v)
	println(io, "Check Node Neighbors: ", tg.check_neighbors)
	println(io, "Variable Node Neighbors: ", tg.vertex_neighbors)
end