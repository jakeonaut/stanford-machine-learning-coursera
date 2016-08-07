function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% assuming X has already been prepended with x0 (hence the -1)
n = size(X, 2)-1;

J = (1/m) * sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta))) + (lambda/(2*m))*sum(theta(2:end, :).^2);

%manually set gradient for theta_0 (which is grad(1) since octave 1-indexes)
%grad(1) = (1/m) * sum((sigmoid(theta'*X')'-y) .* X(:, 1));
% add the part of the sum corresponding to the derivative of the regularization term
% for all the rest of the j = 1 through j = n gradients
%grad = ((1/m) * sum((sigmoid(theta'*X')'-y) .* X(:, 2:(n+1))))' + (lambda/m)*theta(2:(n+1), :);

grad_wo_reg = ((1/m) * sum((sigmoid(theta'*X')'-y) .* X))';
grad_w_reg = ((1/m) * sum((sigmoid(theta'*X')'-y) .* X))' + (lambda/m)*theta;

grad = grad_w_reg;
grad(1) = grad_wo_reg(1);






% =============================================================

grad = grad(:);

end
