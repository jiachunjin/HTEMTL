function [W, Z, L, error_per_task, error, obj] = HTEMTL(X_train, y_train, X_val, y_val, hp)
% *************************************************************************
% Hidden Task Enhanced Multi-Task Learning (HTEMTL)
% Input:
%      Training data (cell): X_train, y_train
%      Validation data (cell): X_val, y_val
%      Hyperparameters: hp
% Output:
%      Model parameters: W
%      Task correlation matrix: Z
%      Feature correlation matrix: L
%      Error per task: error_per_task
%      Total error: error
%      Objective values: obj
% *************************************************************************

% problem size
num_task = size(X_train, 2);
num_dim = size(X_train{1}, 2);

% hyperparameters
init_HTEMTL;

% initialization by single task learning
if hp.cla_flag
    W = learnW_STL_cla(X_train, y_train, X_val, y_val, hp, para_W);
else
    W = learnW_STL_reg(X_train, y_train, X_val, y_val, hp);
end
Z = zeros(num_task, num_task);
L = zeros(num_dim, num_dim);
if strcmp(hp.LR_sur, 'N')
    Z = learnZ_ADMM(Z, L, W, hp.lambda, hp.gamma, para_Z, num_task);
    if hp.withL
        L = learnL_ADMM(L, Z, W, hp.lambda, hp.gamma, para_L, num_dim);
    end
elseif strcmp(hp.LR_sur, 'F')
    Z = learnZ_Fro(L, W, hp.lambda, hp.gamma, num_task);
    if hp.withL
        L = learnL_Fro(Z, W, hp.lambda, hp.gamma, num_dim);
    end
end

% main loop
obj = [];
obj = [obj, obj_value_HTEMTL(X_train, y_train, W, Z, L, hp)];
for iter = 2 : outer_iter_max
    W_prev = W;
    Z_prev = Z;
    L_prev = L;

    % 1. update W
    W = learnW_GD_HTEMTL(W_prev, Z_prev, L_prev, X_train, y_train, hp, para_W);

    % 2. update Z and L
    if strcmp(hp.LR_sur, 'N')
        Z = learnZ_ADMM(Z_prev, L_prev, W, hp.lambda, hp.gamma, para_Z, num_task);
        if hp.withL
            L = learnL_ADMM(L_prev, Z, W, hp.lambda, hp.gamma, para_L, num_dim);
        end
    elseif strcmp(hp.LR_sur, 'F')
        Z = learnZ_Fro(L_prev, W, hp.lambda, hp.gamma, num_task);
        if hp.withL
            L = learnL_Fro(Z, W, hp.lambda, hp.gamma, num_dim);
        end
    end

    % 3. convegence condition
    obj = [obj, obj_value_HTEMTL(X_train, y_train, W, Z, L, hp)];
    if obj_converges(obj) || iter == outer_iter_max
        if hp.cla_flag
            [error_rate, error_num, total_sample] = calculate_error_cla(X_val, y_val, W);
            error_per_task = error_rate;
            error = sum(error_num) / total_sample;
        else
            [rmse_per_task, macro_rmse] = calculate_RMSE_reg(X_val, y_val, W);
            error_per_task = rmse_per_task;
            error = macro_rmse;
        end
        break;
    end
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