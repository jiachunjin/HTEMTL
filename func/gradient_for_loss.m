% gradient_for_loss: function description
function [G] = gradient_for_loss(W, X_train, y_train, cla_flag)
[num_dim, num_task] = size(W);
G = zeros(num_dim, num_task);
for t = 1:num_task
    if cla_flag
        G(:, t) = (1 / size(y_train{t}, 1)) * X_train{t}' * (sigmoid(X_train{t} * W(:, t)) - y_train{t});
    else
        G(:, t) = (1 / size(y_train{t}, 1)) * X_train{t}' * (X_train{t} * W(:, t) - y_train{t});
    end
end
end