% DD (unlabeled edges)
%load('DD')
%labels = ldd; % labels(x) in {1, 2} for all x
%GRAPHS = DD;

% ENZYMES (unlabeled edges)
load('ENZYMES')
labels = lenzymes; % labels(x) in {1, 2, 3, 4, 5, 6} for all x
GRAPHS = ENZYMES;

% MUTAG (labeled edges)
%load('MUTAG')
%labels = lmutag; % labels(x) in {-1, 1} for all x
%GRAPHS = MUTAG;

% NCI1 (labeled edges)
%load('NCI1')
%labels = lnci1; % labels(x) in {0, 1} for all x
%GRAPHS = NCI1;

% NCI109 (labeled edges)
%load('NCI109')
%labels = lnci109; % labels(x) in {0, 1} for all x
%GRAPHS = NCI109;

graphs_count = length(GRAPHS);

for i = 1:graphs_count
	% convert only graphs of the specified label
	% if labels(i) ~= -1
	% if labels(i) ~= 0
	if labels(i) ~= 1
	% if labels(i) ~= 2
	% if labels(i) ~= 3
	% if labels(i) ~= 4
	% if labels(i) ~= 5
	% if labels(i) ~= 6
		continue
	end
	
	disp(['Converting graph no ', num2str(i)])
	
	% create file for the i-th graph
	f = fopen(strcat(num2str(i), '.graph'), 'w');
	
	% write node labels
	fprintf(f, 'node labels\n');
	fprintf(f, num2str(transpose(GRAPHS(i).nl.values)));
	fprintf(f, '\n');
	
	% write adjacency list
	fprintf(f, 'adjacency list\n');
	nodes_count = length(GRAPHS(i).al);
	for j = 1:nodes_count
		fprintf(f, num2str(GRAPHS(i).al{j}));
		fprintf(f, '\n');
	end
	
	% write edge labels
	fprintf(f, 'edge labels\n');
	edges_count = size(GRAPHS(i).el.values, 1);
	for j = 1:edges_count
		%fprintf(f, num2str(GRAPHS(i).el.values(j,:)));
		fprintf(f, num2str(GRAPHS(i).el.values{j}));
		fprintf(f, '\n');
	end
	
	fclose(f);
end

