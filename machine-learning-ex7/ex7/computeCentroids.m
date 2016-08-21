function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


for k = 1:K,
	%idx(i) indicates the centroid index that was assigned to xi
	%get all xis assigned to centroid k
	assigned_xis = X((idx == k), :);
	%get the number of xis assigned to k
	%num_xis is a scalar
	num_xis = size(assigned_xis, 1);
	
	%get the "sum" of all assigned xis
	%note, have to prepend with a 0 row so that Octave's sum function
	%will sum column wise (in the case where rows = 1 for assigned_xis)
	%instead of summing all the numbers of the row into a scalar
	assigned_xis = [zeros(1, size(assigned_xis, 2)); assigned_xis];
	%sum_xis is a 1 x n vector
	sum_xis = sum(assigned_xis);
	
	%now, assign the centroid row to the mean of the assigned xis
	centroids(k, :) = sum_xis / num_xis;
endfor


% =============================================================


end

