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

%STEP 1, get the first Add the column of ones which is the bias
X = [ones(m, 1) X];

%STEP 2, compute the predictions from the first layer using the first layer
%parameters
%X is 5000x401 and theta1 is 25x401, transpose(theta1) = 401x25
%sig1 = 5000x25

z2 = X*Theta1';

%size(z2)
%size(X)
%size(Theta1')

%dd
a2 = sigmoid(z2);

%STEP 3, compute the output layer, and use that for the cost function
%Add the bias again 
a2 = [ones(m,1) a2];
%now sig1 is 5000x26 Theta2 is 10x26
%Now compute sigmoid for the output layer
%hx is gonna be 5000x10.
z3 = a2*Theta2';
a3 = sigmoid(z3);

%STEP 4 Now since hx is 5000x10, y should also be 5000x10. 
different_classes = unique(y);
num_classes = size(different_classes,1);
yout = zeros(m,num_classes); %5000x10 
for c=1:num_classes
    %c
    %different_classes(c)
    yout(find(y==different_classes(c)),c) = 1;
endfor
    
%size(yout)
%size(hx)
%STEP 5 compute the cost function without regularization yet!
J = (1./m)*sum( -yout.*log(a3)-(1-yout).*log(1-a3) );
J = sum(J); 
J = J+lambda/(2*m)*( sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

a1 = X;
delt1=0;
delt2=0;
for t = 1:m
   d3 = a3(t,:)'-yout(t,:)';

   %fprintf('-----')
   %size(d3)
   %size(z2(t,:))
   %size(Theta2(:,2:end)')
   %size(Theta2'*d3)
   %size(sigmoidGradient(z2(t,:)))
   %fprintf('-----')

   %dd
   %size(Theta2(:,2:end)')
   %size(d3)
   d2 = (Theta2(:,2:end)'*d3).*sigmoidGradient(z2(t,:))';
   %d2 = d2(2:end);
   
   %size(d2)
   %size(d2*a1(t,:))
   %size(Theta1)
   %size(d3)
   %size(a2(t,:))
   %dd
   delt2 = delt2+d3*a2(t,:);
   delt1 = delt1+d2*a1(t,:);
endfor
    
%size(delt1)
%size(delt2)
%size(Theta1)
%size(Theta2)
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Theta1_grad = (1.0/m)*delt1;
Theta2_grad = (1.0/m)*delt2;


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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
