function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = 1 / (2*m) * norm(X * theta - y)^2; % vectorized result with norm()
%J = 1 / (2*m) * norm(theta' * X - y)^2; % vectorized result with norm(),
%alternative ans, but wrong! lecture note may be wrong


% =========================================================================

end
