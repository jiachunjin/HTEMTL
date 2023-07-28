%% learnW_GD_cla: function description
function [W] = learnW_GD_pHTEMTL(W_1, Z_set, L_set, X_train, y_train, hp, para_W)

% initialization
G = zeros(hp.num_dim, hp.num_task);   % the gradient
norm_G_column = Inf(1, hp.num_task);
W_set = cell(hp.num_layer+1, 1);
W_set{1} = W_1;

% main loop
iter = 0;
while any(norm_G_column > para_W.EPS_G) && iter < para_W.iter_max
    prev_G = G;

    % Update W_set in each iteration
    for k = 1 : hp.num_layer
        W_set{k+1} = W_set{k} * Z_set{k} + L_set{k} * W_set{k};
    end

    % 1. gradient of loss
    for i = hp.num_layer : -1 : 1
        if i == hp.num_layer
            G = gradient_for_loss(W_set{i}, X_train, y_train, hp.cla_flag);
        else
            G = G * Z_set{i}' + L_set{i}' * G;
        end
    end

    % 2. gradient of self-expression
    for k = 1 : hp.num_layer
        for i = k : -1 : 1
            if i == k
                Delta_W = W_set{i} - W_set{i+1};
                G_tmp = Delta_W - Delta_W * Z_set{i}' - L_set{i}' * Delta_W;
            else
                G_tmp = G_tmp * Z_set{i}' + L_set{i}' * G_tmp;
            end
        end
        G = G + hp.lambda_set(k) * G_tmp;
    end

    % 3. gradient of regularization
    G = G + hp.eta * W_set{1};

    if iter < 2
        stepsize = para_W.alpha;
    else
        delta_G = G - prev_G;
        delta_W = W_set{1} - prev_W;
        stepsize = (delta_W(:)'*delta_G(:)) ./ (delta_G(:)'*delta_G(:));
    end
    prev_W = W_set{1};
    W_set{1} = W_set{1} - stepsize * G;
    iter = iter + 1;
    norm_G_column = sqrt(sum(G.^2, 1));

end
W = W_set{1};
end