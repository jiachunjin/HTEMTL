%% learnZ_Fro: function description
function [outputs] = learnL_Fro(Z, W, lambda, gamma, num_dim)
% 	[D, T] = size(W);
	outputs = ((W - W * Z) * W') / (W * W' + (2*gamma/lambda) * eye(num_dim));
end