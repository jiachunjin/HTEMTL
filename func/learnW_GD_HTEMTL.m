%% learnW_GD_cla: function description
function [W] = learnW_GD_HTEMTL(W, Z, L, X_train, y_train, hp, para_W)

% initialization
num_dim = hp.num_dim;
num_task = hp.num_task;
G = zeros(num_dim, num_task); % the gradient
norm_G_column = Inf(1, num_task);

% main loop
iter = 0;
while any(norm_G_column > para_W.EPS_G) && iter < para_W.iter_max
    prev_G = G;
    % 1. gradient of loss
    G = gradient_for_loss(W, X_train, y_train, hp.cla_flag);
    % 2. gradient of self-expression
    M = W - W * Z - L * W;
    G = G + hp.lambda * ((eye(num_dim) - L)' * M - M * Z');
    % 3. gradient of regularization
    G = G + hp.eta * W;

    if iter < 2
        stepsize = para_W.alpha;
    else
        delta_G = G - prev_G;
        delta_W = W - prev_W;
        stepsize = (delta_W(:)'*delta_G(:)) ./ (delta_G(:)'*delta_G(:));
    end
    prev_W = W;
    W = W - stepsize * G;
    iter = iter + 1;
    norm_G_column = sqrt(sum(G.^2, 1));
end
end