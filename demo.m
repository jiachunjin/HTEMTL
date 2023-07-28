clear
close all
rng('default')
addpath('func','eval')

% Select the algorithm and the number of HTE layers
select_alg = 'pHTEMTL';    % 'HTEMTL' or 'pHTEMTL'
hp.num_layer = 2;          % positive integer

% generate the synthetic dataset
generate_data_cla

% hyperparameters
hp.gamma  = 1e-1;
hp.lambda = 1e-2;
hp.eta    = 1e-2;
hp.phi    = 1e-2;
hp.withL  = true;          % true: L~=0; false: L=0
hp.LR_sur = "N";           % 'N': the trace norm; 'F': the Frobenious norm
hp.cla_flag = true;        % true: classification; false: regression

% main function
tic;
switch select_alg
    case 'pHTEMTL'
        [W_set, Z_set, L_set, error_t, error, obj] = pHTEMTL(X_train_cla, y_train_cla, X_val_cla, y_val_cla, hp);
    case 'HTEMTL'
        [W, Z, L, error_t, error, obj] = HTEMTL(X_train_cla, y_train_cla, X_val_cla, y_val_cla, hp);
end
toc;

% display results
switch select_alg
    case 'pHTEMTL'
        for k = 1 : hp.num_layer
            [metric1, metric2, metric_list] = evaluate_HTEMTL(X_test_cla, y_test_cla, W_set{k}, hp.cla_flag);
            disp(['Layer ',num2str(k),': ',metric_list{1}, '=', num2str(metric1), ', ', metric_list{2}, '=', num2str(metric2)])
            [~] = show_Z(Z_set{k});
            title(['Layer ',num2str(k)])
        end
    case 'HTEMTL'
        [metric1, metric2, metric_list] = evaluate_HTEMTL(X_test_cla, y_test_cla, W, hp.cla_flag);
        disp([metric_list{1}, '=', num2str(metric1), ', ', metric_list{2}, '=', num2str(metric2)])
        [~] = show_Z(Z);
end
[U,S,V] = svd(W_star, 'econ');
V_new = V(:, 1:16);
Z_star = V_new*V_new';
[~] = show_Z(Z_star);
title('Ground truth')
figure
plot(obj)
title('Convergence analysis')