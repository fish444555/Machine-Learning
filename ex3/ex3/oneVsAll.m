function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
% for i = 1:num_labels
% % for i = 1:1
%     
%     initial_theta = zeros(size(X, 2), 1);
%     options = optimset('GradObj', 'on', 'MaxIter', 400);
% %     for j = 1:size(y,1)
% %         if y(j,1) == i
% %             y_l(j,1) = 1;
% %         else
% %             y_l(j,1) = 0;
% %         end
% %     end
%     [theta, J] = ...
%         fmincg(@(t)(lrCostFunction(t, X, (y==i), lambda)), ...
%         initial_theta, options);
% 
%     all_theta(i,:) = theta';
% end



for c = 1:num_labels
    % Set Initial theta
    initial_theta = zeros(n + 1, 1);
    % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    
    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost
    [all_theta(c,:)] = ...
        (fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
        initial_theta, options))';     
%    initial_theta = all_theta;
end

% =========================================================================


end


% 
% function [J, grad] = costFunctionReg(theta, X, y, lambda)
% m = length(y); % number of training examples
% J = 0;
% grad = zeros(size(theta));
% z = X*theta;
% h = sigmoid(z);
% J = (-y'*log(h)-(1-y)'*log(1-h))/m + lambda*((theta'*theta)-theta(1,1)*theta(1,1))/(2*m);
% grad = ((h-y)'*X(:,1:size(X,2)))'/m;
% grad(2:size(X,2),:) = grad(2:size(X,2),:)+lambda*theta(2:size(X,2),:)/m;
% end