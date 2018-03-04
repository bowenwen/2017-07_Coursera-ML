function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

cList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
errMatrix = zeros(length(cList),length(sigmaList));
errMatrix(errMatrix==0)= -1; %avoid zeros when intiatializing

for i=1:length(cList)
    for j=1:length(sigmaList)
        current_C = cList(:,i);
        current_sigma = sigmaList(:,j);
        % Train using training data
        cvModel = svmTrain(X, y, current_C, ...
            @(x1, x2) gaussianKernel(x1, x2, current_sigma));
        % Cross validate using cv data
        predictions = svmPredict(cvModel, Xval);        
        errMatrix(i,j) = mean(double(predictions ~= yval));        
    end
end

%Smallest Indices of minimum error
%https://www.mathworks.com/help/matlab/ref/min.html
[M,I] = min(errMatrix(:))
[I_row, I_col] = ind2sub(size(errMatrix),I);

C = cList(:,I_row);
sigma = sigmaList(:,I_col);

% =========================================================================

end
