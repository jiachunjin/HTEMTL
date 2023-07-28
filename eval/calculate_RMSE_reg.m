%% calculate_error_reg: function description
function [rmse_per_task, rmse] = calculate_RMSE_reg(X_test, y_test, W)
	num_task = size(X_test, 2);
    whole_y = [];
    whole_predicted = [];
    rmse_per_task = zeros(num_task,1);
	for t = 1 : num_task
        y_predicted = X_test{t} * W(:, t);
        rmse_per_task(t) = sqrt(sum((y_predicted-y_test{t}).^2)/size(y_test{t},1));
		whole_y = [whole_y; y_test{t}];
		whole_predicted = [whole_predicted; y_predicted];
	end
	N = length(whole_y);
    rmse = sqrt(sum((whole_predicted-whole_y).^2)/N);
end
