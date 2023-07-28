%% calculate_error_cla: function description
function [error_rate, error_num, total_sample] = calculate_error_cla(X_test, y_test, W)
	num_task = size(X_test, 2);
	error_rate = zeros(num_task, 1);
	error_num = zeros(num_task, 1);
	total_sample = 0;
	for t = 1 : num_task
        N_t = size(y_test{t}, 1);
        total_sample = total_sample + N_t;
		prediction = sign(X_test{t} * W(:, t));
		prediction(prediction == -1) = 0;
		error_rate(t) = sum(~(prediction == y_test{t})) / N_t;
		error_num(t) = sum(~(prediction == y_test{t}));
	end
end