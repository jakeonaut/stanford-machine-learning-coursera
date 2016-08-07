function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1); %m x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1);

%loop over every input example
for i = 1:m,
	min_dist = 9999;
	%loop over all centroids (to find the one closest to this x^(i)
	xi = X(i, :);
	for k = 1:K,
		%http://scicomp.stackexchange.com/questions/8223/euclidean-distance-in-octave
		% the euclidean distance between two vectors is the two-norm of their difference
		% note: also we are squaring distance because this is squared in the k-means algorithm
			% the teacher said something about this but i didn't write it down...
		dist = norm(xi - centroids(k, :), 2)^2;
		if dist < min_dist,
			min_dist = dist;
			idx(i) = k;
		endif
	endfor
endfor


% =============================================================

end

