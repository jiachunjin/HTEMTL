%% calculate_error_reg: function description
function [macro_mae] = calculate_MAE_reg(X_test, y_test, W)
	num_task = size(X_test, 2);
    whole_y = [];
    whole_predicted = [];
	for t = 1 : num_task
		whole_y = [whole_y; y_test{t}];
		whole_predicted = [whole_predicted; X_test{t} * W(:, t)];
	end
	macro_mae = 1 / length(whole_y) * sum(abs(whole_y - whole_predicted));
end
