%% cal_AUC: function description
function [result] = calculate_AUC_cla(X_test, y_test, W)
	num_task = size(X_test, 2);
    Yout = cell(num_task,1);
	for t = 1 : num_task
		Yout{t} = sigmoid(X_test{t} * W(:, t));
	end
	result = evalAUC(Yout, y_test);
end