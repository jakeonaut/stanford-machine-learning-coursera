function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% without regularization...
% J(theta) = -(1/m) * [sum(m, i=1, sum(K, k=1, (yk^(i)*log(htheta(x^(i)))k + ...
%			 (1-yk^(i))*log(1-(htheta(x^(i)))k))))]

%need to add a column of 1s to X (bias unit to input)
X = [ones(m, 1) X];

DELTA_2 = zeros(size(Theta2));
DELTA_1 = zeros(size(Theta1));

%vectorized forward propagation
a1 = X;
%a1 is 5000 x 401, Theta1' is 401 x 25, z2 is 5000 x 25
z2 = a1 * Theta1';
%apply activation function to z2
a2 = sigmoid(z2);
%add bias unit column to a2, a2 is 5000 x 26
a2 = [ones(size(a2, 1), 1) a2];
%a2 is 5000 x 26, Theta2' is 26 x K, z3 is 5000 x K
z3 = a2 * Theta2';
%apply activation function to z3
a3 = sigmoid(z3);
%we now have our hypothesis, 5000 x K
h_x = a3; 

%y matrix
y_matrix = eye(num_labels)(y, :);

%compute the cost
J = -(1/m) * sum(sum(y_matrix.*log(h_x) + (1-y_matrix).*log(1-h_x)));
	
%add the regularization term to cost function

%for Theta1
reg_term1 = 0;
for j=1:hidden_layer_size,
	for k=2:input_layer_size+1, %skip first column, as this is bias unit
		reg_term1 = reg_term1 + Theta1(j, k)^2;
	endfor
endfor

%for Theta2
reg_term2 = 0;
for j=1:num_labels,
	for k=2:hidden_layer_size+1, %skip first column, as this is bias unit
		reg_term2 = reg_term2 + Theta2(j, k)^2;
	endfor
endfor

%multiply by lambda/(2*m) to give correct value of regularization term
reg_term = (reg_term1 + reg_term2) * (lambda/(2*m));

%add regularization term to J
J = J + reg_term;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%			BACK PROPAGATION                  %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%delta3 is 5000 x K
%		= a3 is 5000 x K, y_matrix is 5000 x K
delta3 = a3 - y_matrix;
%delta 2 is 5000 x 25
% 		= delta3 is 5000 x K, Theta2(:, 2:end) is K x 25, z2 is 5000 x 25
delta2 = delta3*Theta2(:, 2:end) .* sigmoidGradient(z2);

%DELTA_1 is 25 x 401
%		= delta2' is 25 x 5000, a1 is 5000 x 401
DELTA_1 = delta2'*a1;

%DELTA_2 is K x 26
%		= delta3' is K x 5000, a2 is 5000 x 26		
DELTA_2 = delta3'*a2;

%unregularized
Theta1_grad = (1/m) * DELTA_1;
Theta2_grad = (1/m) * DELTA_2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
