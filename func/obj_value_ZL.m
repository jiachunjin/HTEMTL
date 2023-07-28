function [obj] = obj_value_ZL(X_train, y_train, W_set, Z_set, L_set, hp)
obj = 0;
obj = obj + sum_loss(X_train, y_train, W_set{hp.num_layer}, hp.cla_flag);
for k = 1 : hp.num_layer
    obj = obj + hp.lambda_set(k) / 2 * norm(W_set{k} - W_set{k+1}, 'fro') ^ 2;
    if strcmp(hp.LR_sur, 'N')
        obj = obj + hp.gamma_set(k) * (norm(svd(Z_set{k}),1) + norm(svd(L_set{k}),1));
    elseif strcmp(hp.LR_sur, 'F')
        obj = obj + hp.gamma_set(k) * (norm(Z_set{k}, 'fro') ^ 2 + norm(L_set{k}, 'fro') ^ 2);
    end
end
end
