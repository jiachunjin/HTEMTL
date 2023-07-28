%% learn_W_STL_cla: function description
function [opt_W_STL] = learnW_STL_cla(X_train, y_train, X_val, y_val, hp, para_W)

% lambda_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];
lambda_len = length(hp.lambda_list_STL);
er = ones(lambda_len, 1);
W_list = cell(lambda_len,1);
W = zeros(hp.num_dim, hp.num_task);
for k = 1 : lambda_len
    lambda = hp.lambda_list_STL(k);
    iter_W = 0;
    G = zeros(size(W));
    norm_G_column = Inf(1,hp.num_task);
    while (any(norm_G_column > para_W.EPS_G)) && (iter_W < para_W.iter_max)
        prev_G = G;
        %             G = zeros(size(W));
        for t = 1 : hp.num_task
            g_loss =   X_train{t}' * ( sigmoid(X_train{t}*W(:,t)) -  y_train{t} ) * (1/length(y_train{t}));
            g_reg = lambda * W(:,t);
            G(:,t) = g_loss + g_reg;
        end
        if iter_W < 2
            stepsize = para_W.alpha;
        else
            delta_G = G - prev_G;
            delta_W = W - prev_W;
            stepsize = (delta_W(:)'*delta_G(:))./(delta_G(:)'*delta_G(:));
        end
        prev_W = W;
        W = W - stepsize*G;
        iter_W = iter_W + 1;
        norm_G_column = sqrt(sum(G.^2,1));
    end
    [~, error_num, total_sample] = calculate_error_cla(X_val, y_val, W);
    W_list{k} = W;
    er(k) = sum(error_num) / total_sample;
end
opt_W_STL = W_list{er == min(er)};
end
