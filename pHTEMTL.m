function [W_set_pre, Z_set_pre, L_set_pre, error_per_task, error, obj] = pHTEMTL(X_train, y_train, X_val, y_val, hp)
% *************************************************************************
% Progressive Hidden Task Enhanced Multi-Task Learning (pHTEMTL)
% Input:
%      Training data (cell): X_train, y_train
%      Validation data (cell): X_val, y_val
%      Hyperparameters: hp
% Output:
%      Model parameters (cell): W_set_pre
%      Task correlation matrix (cell): Z_set_pre
%      Feature correlation matrix (cell): L_set_pre
%      Error per task: error_per_task
%      Total error: error
%      Objective values: obj
% *************************************************************************

% hyperparameters
init_pHTEMTL;

% Pre-train by HTEMTL
W_set = cell(2, hp.num_layer+1);
Z_set = cell(2, hp.num_layer);
L_set = cell(2, hp.num_layer);
for k = 1 : hp.num_layer
    if k == 1
        [W_set{1,k}, Z_set{1,k}, L_set{1,k}, ~, ~, ~] = HTEMTL(X_train, y_train, X_val, y_val, hp);
    else
        Z_set{1,k} = Z_set{1,k-1};
        L_set{1,k} = L_set{1,k-1};
    end
    W_set{1,k+1} = W_set{1, k} * Z_set{1,k} + L_set{1,k} * W_set{1, k};
end

% main loop
obj = [];
obj = [obj, obj_value_pHTEMTL(X_train, y_train, W_set(1, :), Z_set(1, :), L_set(1, :), hp)];
for iter = 2 : outer_iter_max

    % update W
    W_set{2, 1} = learnW_GD_pHTEMTL(W_set{1, 1}, Z_set(1, :), L_set(1, :), X_train, y_train, hp, para_W);

    % update Z and L
    if strcmp(hp.LR_sur, 'N')  % proximal gradient descent (PGD)
        [Z_set(2, :), L_set(2, :)] = learnZL_PGD(W_set{2, 1}, Z_set(1, :), L_set(1, :), X_train, y_train, hp, para_ZL);
    elseif strcmp(hp.LR_sur, 'F')  % gradient descent (GD)
        [Z_set(2, :), L_set(2, :)] = learnZL_GD(W_set{2, 1}, Z_set(1, :), L_set(1, :), X_train, y_train, hp, para_ZL);
    end

    % calculate the validation error
    for k = 1 : hp.num_layer
        W_set{2, k+1} = W_set{2, k} * Z_set{2, k} + L_set{2, k} * W_set{2, k};
    end

    % convegence condition
    obj = [obj, obj_value_pHTEMTL(X_train, y_train, W_set(2, :), Z_set(2,:), L_set(2,:), hp)];
    if obj_converges(obj) || iter == outer_iter_max
        W_set_pre = W_set(2, :);
        Z_set_pre = Z_set(2, :);
        L_set_pre = L_set(2, :);
        if hp.cla_flag
            [error_per_task, error_num, total_sample] = calculate_error_cla(X_val, y_val, W_set{2, end-1});
            error = sum(error_num) / total_sample;
        else
            [error_per_task, error] = calculate_RMSE_reg(X_val, y_val, W_set{2, end-1});
        end
        break;
    end
    W_set(1, :) = W_set(2, :);
    Z_set(1, :) = Z_set(2, :);
    L_set(1, :) = L_set(2, :);
end
end

%% obj_converges: function description
function [converged] = obj_converges(obj_values)
converged = false;
if length(obj_values) < 3
    converged = false;
else
    current = obj_values(length(obj_values));
    last = obj_values(length(obj_values) - 1);
    if (last - current) / last < 1e-3
        converged = true;
    end
end
end