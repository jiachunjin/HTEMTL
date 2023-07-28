function [metric1, metric2, metric_list] = evaluate_HTEMTL(X_test, y_test, W, cla_flag)
if cla_flag
    [~, error_num, total_sample] = calculate_error_cla(X_test, y_test, W);
    metric1 = sum(error_num) / total_sample;               % Error Rate: 1 - Accuracy
    [metric2] = calculate_AUC_cla(X_test, y_test, W);      % AUC
    metric_list = {'ER', 'AUC'};
else
    [~, metric1] = calculate_RMSE_reg(X_test, y_test, W);  % RMSE
    metric2 = calculate_MAE_reg(X_test, y_test, W);        % MAE
    metric_list = {'RMSE', 'MAE'};
end
end