% Exercise 8: SVM Nonlinear classification

clear all; close all; clc

% Load training features and labels
[y, x] = libsvmread('ex8b.txt');

gamma = [1,10,100,1000];

% Libsvm options
% -s 0 : classification
% -t 2 : RBF kernel
% -g : gamma in the RBF kernel
for i = 1:length(gamma)
    
    model = svmtrain(y, x, sprintf('-s 0 -t 2 -g %g', gamma(i)));
    
    % Display training accuracy
    [predicted_label, accuracy, decision_values] = svmpredict(y, x, model);
    
    % Plot training data and decision boundary
%     figure(i); hold on
    plotboundary(y, x, model);
    
    title(sprintf('\\gamma = %g', gamma(i)), 'FontSize', 14);
end




