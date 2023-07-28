%% obj_value: function description
function [obj] = obj_value_HTEMTL(X_train, y_train, W, Z, L, hp)
obj = 0;
obj = obj + sum_loss(X_train, y_train, W, hp.cla_flag);
obj = obj + hp.lambda / 2 * norm(W-W*Z-L*W, 'fro') ^ 2;
if strcmp(hp.LR_sur, 'N')
    obj = obj + hp.gamma * (norm(svd(Z),1) + norm(svd(L),1));
elseif strcmp(hp.LR_sur, 'F')
    obj = obj + hp.gamma * (norm(Z, 'fro') ^ 2 + norm(L, 'fro') ^ 2);
end
obj = obj + hp.eta / 2 * norm(W, 'fro') ^ 2;
end