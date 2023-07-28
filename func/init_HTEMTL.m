% hyperparameters
hp.num_task = num_task;
hp.num_dim = num_dim;
hp.lambda_list_STL = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2];

% parameters used to optimize W
para_W.iter_max = 1000; % maximal iteration numbers
para_W.alpha = 1e+1; % initial step size
para_W.EPS_G = 1e-6;
if hp.cla_flag   % classification -- logistic loss
    % parameters used to optimize Z
    para_Z.rho = 1;
    para_Z.mu = 10;
    para_Z.tau_incr = 2;
    para_Z.tau_decr = 2;
    para_Z.EPS_abs = 1e-2;
    para_Z.EPS_rel = 1e-4;
    para_Z.iter_max = 1000;
    % parameters used to optimize L
    para_L.rho = 1;
    para_L.mu = 10;
    para_L.tau_incr = 2;
    para_L.tau_decr = 2;
    para_L.EPS_abs = 1e-3;
    para_L.EPS_rel = 1e-4;
    para_L.iter_max = 1000;
    outer_iter_max = 20;
else  % regression -- squared loss
    % parameters used to optimize Z
    para_Z.rho = 1;
    para_Z.mu = 10;
    para_Z.tau_incr = 2;
    para_Z.tau_decr = 2;
    para_Z.EPS_abs = 5e-3;
    para_Z.EPS_rel = 1e-4;
    para_Z.iter_max = 1000;
    % parameters used to optimize L
    para_L.rho = 1;
    para_L.mu = 10;
    para_L.tau_incr = 2;
    para_L.tau_decr = 2;
    para_L.EPS_abs = 5e-3;
    para_L.EPS_rel = 1e-4;
    para_L.iter_max = 1000;
    outer_iter_max = 20;
end