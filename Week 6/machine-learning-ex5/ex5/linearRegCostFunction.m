function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% same as costFunctionReg.m from ex2
lambdaEye = eye(length(theta)) .* lambda;
lambdaEye(1,1) = 0; % exclude first element, assign that to 0
lambdaEye = sum(lambdaEye,2); % column sum

J = 1 / (2*m) * sum(((X * theta)-y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);
%note: 1/m * [100x1]'[100x3] - ''

grad = 1 / m * (X'*((X * theta)-y)) + (lambdaEye .* theta ./ m) ;
%note: 1/m * [100x3]'[100x1]


% =========================================================================

grad = grad(:);

end
