%% single_task_learn_reg: function description
function [W_single_reg] = learnW_STL_reg(X_train, y_train, X_val, y_val, hp)
% lambda_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3];
err = Inf;
list_len = length(hp.lambda_list_STL);
W = zeros(hp.num_dim, hp.num_task);
for i = 1 : list_len
    for t = 1 : hp.num_task
%         W(:, t) = inv(X_train{t}'*X_train{t} + hp.lambda_list(i) * eye(hp.num_dim)) * (X_train{t}' * y_train{t});
        W(:, t) = (X_train{t}'*X_train{t}+hp.lambda_list_STL(i)*eye(hp.num_dim)) \ (X_train{t}'*y_train{t});
    end
    [~, rmse] = calculate_RMSE_reg(X_val, y_val, W);
    if rmse < err
        err = rmse;
        W_single_reg = W;
    end
end
end