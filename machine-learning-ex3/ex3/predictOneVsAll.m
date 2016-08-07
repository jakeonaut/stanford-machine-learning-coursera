function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% non-vectorized implementation:
% loop through every m example in X
%for i = 1:m
	% predict the number in the i-th example of X
	% all_theta is a 10x401 matrix
		% where the each row has n+1 theta values
		% used in the equation to predict the output value
		% (note, the output value will be a vector of size 10
		%	that is used to classify the example into one of 10 classes)
	% X is a 5000 x 401 vector
		% where each row is one of m examples to be classified
		% and contains n+1 input feature values
	% to calculate the hypothesis/prediction, need to multiply
		% all_theta*X(i)', and then take the logistic function of that value
%	probabilities = sigmoid(all_theta*X(i, :)');
%	[highest_prob, prediction] = max(probabilities);
	
	% and store the prediction value in p at that index
%	p(i) = prediction;
%endfor

%vectorized implementation of above
[highest_probs, p] = max(sigmoid(all_theta*X'));

%have to take transpose because need p to be 5000 x 1, not 1x5000
p = p';



% =========================================================================


end
