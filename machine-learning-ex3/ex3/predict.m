function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%Add the column of ones which is the bias
X = [ones(m, 1) X];
%compute the predictions from the first layer using the first layer
%parameters
%X is 5000x401 and theta1 is 25x401, transpose(theta1) = 401x25
%sig1 = 5000x25
sig1 = sigmoid(X*Theta1');




%Add the bias again 
sig1 = [ones(m,1) sig1];
%now sig1 is 5000x26 Theta2 is 10x26
%Now compute sigmoid for the output layer 
sig2 = sigmoid(sig1*Theta2');

%Now get the maximum
[val p] = max(sig2, [], 2);

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









% =========================================================================


end
