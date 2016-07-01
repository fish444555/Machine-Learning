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

% X = [ones(size(X,1),1) X];
% temp = zeros(size(X,1),size(Theta1,1));
% temp = X*Theta1';
% for i = 1:size(temp,1)
%     for j = 1:size(temp,2)
%         temp(i,j) = 1/(1+exp(-temp(i,j)));
%     end
% end
% temp = [ones(size(temp,1),1) temp];
% temp2 = temp*Theta2';
% 
% for i = 1:size(temp2,1)
%     for j = 1:size(temp2,2)
%         temp2(i,j) = 1/(1+exp(-temp2(i,j)));
%     end
% end
% [k p] = max(temp2, [], 2); 



X = [ones(m,1) X];
a = sigmoid(X*Theta1');
a = [ones(size(a,1),1) a];
p = sigmoid(a*Theta2');
[temp p] = max(p,[],2);




% =========================================================================


end
