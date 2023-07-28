%% learnZ_Fro: function description
function [outputs] = learnZ_Fro(L, W, lambda, gamma, num_task)
% [D, T] = size(W);
% outputs = inv(lambda * W' * W + 2 * gamma * eye(size(W,2))) * (lambda * W' * (W - L * W));
% outputs = (lambda * (W' * W) + 2 * gamma * eye(size(W,2))) \ (lambda * (W' * (W - L * W)));
outputs = (W'*W + (2*gamma/lambda)*eye(num_task)) \ (W' * (W - L * W));

end