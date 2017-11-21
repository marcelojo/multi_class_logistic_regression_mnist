function [a_param] = train_regression (data, label)
% Define max interation
max_interation = 100;


% Define learning rate step
learning_rate = 0.4;

% Initialize parameters for all 10 classes
a_param = ones(10, size(data)(2) + 1);

% Loop for training
for class = 0:9
    for i = 1:max_interation
        a_param(class + 1, :) = step_gradient(a_param(class + 1,:), data, label == class, learning_rate);
    end
end
endfunction