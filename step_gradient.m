function [new_a J] = step_gradient(a, data, label, learning_rate)
    % Get data length
    data_len = size(data)(1);
    
    
    % Get x and y points from data
    x = ones(data_len, 28*28 + 1); %x0 = 1
    x(:,2:end) = data;
    
    y = label;
    
    % gradiente is defined as:
    % (1/n) * sum(i=0:n){x_m*[h - y]}  where x0 is ALWAYS 1  
    
    % h(x) = 1 / 1 + e^-z (logistic function)
    z = x * a';                 % [60000 x 785] x [785 x 1] ==> y = a0x0 + a1x1 + a2x2 + ... a4x4
    h = 1 ./ (1 + e.^-z);       % [60000 x 1]
    
    % calculate h - y which will be used in the gradient descent
    y_error = (h - y);          % [60000 x 1]
    
    temp = y_error' * x;        % [1 x 60000] x [60000 x 785] ==> sum(i=0:n){x_m*[y' - y]} where x_m can be x0, x1, x2, ... , x4
    a_temp = (temp)./data_len;
    new_a = a - (learning_rate .* a_temp);
    
    J = 1/data_len * sum(-y .* log(h) - (1 - y) .* log(1 - h));
    
    %{
    x = [60k x 5]     - 60k samples
    a = [1 x 785]     - 785 parameters
    z = [60k x 1]     - 60k results
    h = [60k x 1]     - 60k results 
    new_a = [1 x 785] - 785 new parameters
    %}
endfunction
