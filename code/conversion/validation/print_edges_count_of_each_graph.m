GRAPHS = ENZYMES
% GRAPHS = MUTAG
% GRAPHS = DD
% GRAPHS = NCI1
% GRAPHS = NCI109


f = fopen('matlab_edges_count_of_each_graph.csv', 'w');

for graph_num = 1:length(GRAPHS)
	edges_count = 0

	adj_list = GRAPHS(graph_num).al
	nodes_count = length(adj_list)
	
	for v = 1:nodes_count
		v_nbrs_count = length(adj_list{v})
		edges_count = edges_count + v_nbrs_count
	end
	
	fprintf(f, [num2str(graph_num), '; ', num2str(edges_count)])
	fprintf(f, '\n')
end

fclose(f)