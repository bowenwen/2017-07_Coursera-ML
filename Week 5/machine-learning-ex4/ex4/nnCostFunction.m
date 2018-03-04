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
% -------------------------------------------------------------

% =========================================================================



% FEEDFORWARD

% build recoded y matrix
yMatrix = zeros(num_labels,size(y,1))
for item = 1:size(y)
    yMatrix(y(item),item) = 1;
end
yMatrix = yMatrix'

% calculate all probabilities
X_bias = ones(m,1);
X = [X_bias,X]; % add bias column x0
A2 = sigmoid(X * Theta1');
A2_bias = ones(m,1);
A2 = [A2_bias,A2]; % add bias column a(2)0
prob_Matrix = sigmoid(A2 * Theta2'); % prob_Matrix = h

% generate classification result
[finalProb, p] = max(prob_Matrix, [], 2); % finalProb = h


% COST FUNCTION WITH REGULARIZATION

%J = 1 / m * sum(-y'*log(finalProb)-(1-y)'*log(1-finalProb)) 
% y needs additional treatment
% loop over K (num_labels)

currentJ = 0;
for example = 1:m
    currenty = yMatrix(example,:)';
    currentprob = prob_Matrix(example,:)';
    currentJ = currentJ + sum(-currenty'*log(currentprob)-(1-currenty)'*log(1-currentprob));
end

J = currentJ / m;

Theta1NoBias = Theta1(:,2:size(Theta1,2));
Theta2NoBias = Theta2(:,2:size(Theta2,2));

% add regularization
J = J + lambda / (2*m) * (sum(sum(Theta1NoBias.^2)) + sum(sum(Theta2NoBias.^2)))


% BACKPROPAGATION UPDATES
% do for each example
%DELTA_2 = zeros(size(Theta2,1), size(Theta2,2)-1)
%DELTA_1 = zeros(size(Theta1,1), size(Theta1,2)-1)
DELTA_2 = zeros(size(Theta2,1), size(Theta2,2))
DELTA_1 = zeros(size(Theta1,1), size(Theta1,2))
for t = 1:m
    % step 1: feed forward
    a1 = X(t,:); % a_1k
    a1_noBias = a1(1,2:size(a1,2)); %for later
    z2 = a1 * Theta1';
    a2_noBias = sigmoid(z2); %for later
    a2 = [1,a2_noBias];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3); % a_3k = h
    
    % step 2: find delta3
    y_k_cur = zeros(num_labels,1);
    y_k_cur(y(t), 1) = 1;
    delta_3 = a3 - y_k_cur'; % delta_k3
    
    % step 3: set delta2
    Theta2_noBias = Theta2(:,2:size(Theta2,2)); % take out first col, bias
    delta_2 = (Theta2_noBias' * delta_3')' .* sigmoidGradient(z2);
    
    % step 4: find DELTA_l and DELTA_2
    %DELTA_2 = DELTA_2 + delta_3' * a2_noBias;
    %DELTA_1 = DELTA_1 + delta_2' * a1_noBias;
    DELTA_2 = DELTA_2 + delta_3' * a2;
    DELTA_1 = DELTA_1 + delta_2' * a1;
           
end

% COMPUTE GRADIANT - see step 5
%grad = 1 / m * X' * (sigmoid(X * theta)-y) + (lambdaEye .* theta ./ m);

% step 5: compute J_Theta2 & J_Theta1
Theta2_grad = 1 / m * DELTA_2; % J_Theta2
Theta1_grad = 1 / m * DELTA_1; % J_Theta1

% finally, add regularization parameter

Theta2_grad_reg = Theta2;
Theta2_grad_reg(:,1) = 0;
Theta1_grad_reg = Theta1;
Theta1_grad_reg(:,1) = 0;
Theta2_grad = Theta2_grad + lambda / m * Theta2_grad_reg;
Theta1_grad = Theta1_grad + lambda / m * Theta1_grad_reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
