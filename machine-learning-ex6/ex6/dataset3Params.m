function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
m1 = size(C1)(2);
fprintf('\nSIZE ...%i \n',m1)
pred_error = 10000000000;
for ii = 1:m1
    for jj=1:m1
        
        %fprintf('\nSIZE ...%i \n',m1)
        %fprintf('\nI AM HERE ...%i %i \n',ii,jj);
        model= svmTrain(X, y, C1(ii), @(x1, x2) gaussianKernel(x1, x2, sigma1(jj)));
        %fprintf('\nI AM HERE ...%i %i \n',i,j);
        predictions = svmPredict(model,Xval);
        %fprintf('\nI AM HERE ...%i %i \n',i,j);
        pred_error_cur = mean(double(predictions ~= yval));
        %fprintf('\n %f %f\n',pred_error,pred_error_cur);
        
        if (pred_error_cur < pred_error)
            C = C1(ii);
            sigma=sigma1(jj);
            pred_error=pred_error_cur;
        end
end
end

fprintf('\nEND\n')
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
