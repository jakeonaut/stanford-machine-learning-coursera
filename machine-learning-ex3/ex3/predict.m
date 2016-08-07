function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% remember: Theta1 is 25 x 401, Theta2 is 10 x 26

% add column x0 to leftmost side of X
X = [ones(m, 1) X];

%vectorized implementation

%a1 is size 5000 x 401
a1 = X;	

%z2 is size (5000 x 401) x (401 x 25) = 5000 x 25
z2 = a1 * Theta1';

% a2 is size 5000 x 25
a2 = sigmoid(z2);
%need to add bias unit (5000 x 26)
a2  = [ones(m, 1) a2];

%z3 is size (5000 x 26) x (26 x 10) = 5000 x 10
z3 = a2 * Theta2';

%a3 is size 5000 x 10
a3 = sigmoid(z3);

%final prediction probabilities is 5000 x 10
% need to make predictions (classify each example into one of num_labels classes

% need to make p be a vector of 5000 x 1
% by transposing a3, max will return p vector of 1 x 5000
[highest_probs, p] = max(a3');

%now just transpose p to make it 5000 x 1
p = p';



% =========================================================================


end
