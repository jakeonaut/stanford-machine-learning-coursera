function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

%Sigma here is the covariance matrix of the data X (don't ask me what that means, the teacher
% said the math here is complicated and it's not too important to understand for now)
%TODO:: take a linear algebra class refresher & advanced
Sigma = (1/m)*X'*X;

%use SVD to compute the principal components (ditto above)
[U, S, V] = svd(Sigma);





% =========================================================================

end
