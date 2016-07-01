function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
     J_history(iter) = computeCost(X, y, theta);
%     temp = theta(1);
%     theta(1) = theta(1)-sum(X(iter,1)*temp+X(iter,2)*theta(2)-y)*X(iter,1)/m;
%     theta(2) = theta(2)-sum(X(iter,1)*temp+X(iter,2)*theta(2)-y)*X(iter,2)/m;

%     for i = 1:m
        
        temp = [theta(1,1);theta(2,1)];
%         theta(1,1) = theta(1,1)-alpha*sum(X(m,1)*temp+X(m,2)*theta(2,1)-y)*X(m,1)/m;
%         theta(2,1) = theta(2,1)-alpha*sum(X(m,1)*temp+X(m,2)*theta(2,1)-y)*X(m,2)/m;
        theta(1,1) = theta(1,1)-alpha*((X*temp-y)'*X(:,1))/m;
        theta(2,1) = theta(2,1)-alpha*((X*temp-y)'*X(:,2))/m;
%     end

end

end
