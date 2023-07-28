%% sum_loss: function description
function [outputs] = sum_loss(X_train, y_train, W, cla_flag)
outputs = 0;
num_task = size(W,2);
for t = 1 : num_task
    if cla_flag
        p_t = sigmoid(X_train{t} * W(:, t));
        outputs = outputs - 1 / size(y_train{t}, 1) * (y_train{t}' * log(p_t) + (1 - y_train{t})' * log(1 - p_t));
    else
        outputs = outputs + 0.5 / size(y_train{t}, 1) * norm(X_train{t} * W(:, t) - y_train{t}) ^ 2;
    end
end
end