%% learnW_GD_cla: function description
function [Z_set, L_set] = learnZL_PGD(W_1, Z_set, L_set, X_train, y_train, hp, para_ZL)

% initialization
W_set = cell(hp.num_layer+1, 1);
W_set{1} = W_1;
GZ = cell(hp.num_layer, 1);
G1_tmp = zeros(size(W_1));           % backwards signal for calculating G1
G2_tmp = cell(hp.num_layer, 1);      % backwards signal for calculating G2

% Initialize W_set
for k = 1 : hp.num_layer
    W_set{k+1} = W_set{k} * Z_set{k} + L_set{k} * W_set{k};
end

% main loop
obj_ZL = [];
obj_ZL = cat(1,obj_ZL,obj_value_ZL(X_train, y_train, W_set, Z_set, L_set, hp));
for iter = 1 : para_ZL.iter_max

    % Update Z and L by gradient backpropagation
    for i = hp.num_layer: -1 : 1

        % 1. Calculate the gradient for loss
        if i == hp.num_layer
            G1_Z = 0;
            if hp.withL
                G1_L = 0;
            end
        elseif i == hp.num_layer - 1
            G1_tmp = gradient_for_loss(W_set{i+1}, X_train, y_train, hp.cla_flag);
            G1_Z = W_set{i}' * G1_tmp;
            if hp.withL
                G1_L = G1_tmp * W_set{i}';
            end
        else
            G1_tmp = G1_tmp * Z_set{i+1}' + L_set{i+1}' * G1_tmp;
            G1_Z = W_set{i}' * G1_tmp;
            if hp.withL
                G1_L = G1_tmp * W_set{i}';
            end
        end

        % 2. Calculate the gradient for the HTE layer
        for j = i : hp.num_layer
            if j == i
                G2_Z = - hp.lambda_set(j) * W_set{i}' * (W_set{j} - W_set{j+1});
                if hp.withL
                    G2_L = - hp.lambda_set(j) * (W_set{j} - W_set{j+1}) * W_set{i}';
                end
            elseif j == i + 1
                Delta_W = W_set{j} - W_set{j+1};
                G2_tmp{j} = Delta_W - Delta_W * Z_set{j}' - L_set{j}' * Delta_W;
                G2_Z = G2_Z + hp.lambda_set(j) * W_set{i}' * G2_tmp{j};
                if hp.withL
                    G2_L = G2_L + hp.lambda_set(j) * G2_tmp{j} * W_set{i}';
                end
            else
                for jj = j : hp.num_layer
                    G2_tmp{jj} = G2_tmp{jj} * Z_set{jj-1}' + L_set{jj-1}' * G2_tmp{jj};
                    G2_Z = G2_Z + hp.lambda_set(jj) * W_set{i}' * G2_tmp{jj};
                    if hp.withL
                        G2_L = G2_L + hp.lambda_set(jj) * G2_tmp{jj} * W_set{i}';
                    end
                end
            end
        end

        % 3. Proximal operator
        GZ{i} = G1_Z + G2_Z;
        if hp.withL
            GL{i} = G1_L + G2_L;
        end
        Z_set{i}  = prox_SVT(Z_set{i} - para_ZL.stepsize * GZ{i}, hp.gamma_set(i)*para_ZL.stepsize);
        if hp.withL
            L_set{i}  = prox_SVT(L_set{i} - para_ZL.stepsize * GL{i}, hp.gamma_set(i)*para_ZL.stepsize);
        end

    end

    % Update W_set in each iteration
    for k = 1 : hp.num_layer
        W_set{k+1} = W_set{k} * Z_set{k} + L_set{k} * W_set{k};
    end

    % Check the convergence condition
    obj_ZL = cat(1,obj_ZL,obj_value_ZL(X_train, y_train, W_set, Z_set, L_set, hp));
    if (iter > 1 && abs(obj_ZL(iter)-obj_ZL(iter-1)) / obj_ZL(iter-1) < para_ZL.absTol) || isnan(obj_ZL(end))
        break;
    end
end
end


% function [obj] = obj_value_ZL(X_train, y_train, W_set, Z_set, L_set, hp)
% obj = 0;
% obj = obj + sum_loss(X_train, y_train, W_set{hp.num_layer}, hp.cla_flag);
% for k = 1 : hp.num_layer
%     obj = obj + hp.lambda_set(k) / 2 * norm(W_set{k} - W_set{k+1}, 'fro') ^ 2;
%     obj = obj + hp.gamma_set(k) * (norm(svd(Z_set{k}),1) + norm(svd(L_set{k}),1));
% end
% end