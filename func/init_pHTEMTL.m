% hyperparameters
hp.num_task =  size(X_train, 2);
hp.num_dim = size(X_train{1}, 2);
hp.lambda_list_STL = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]; % grid for STL
lambda_set = zeros(hp.num_layer,1);
gamma_set = zeros(hp.num_layer,1);
for k = 1 : hp.num_layer
    lambda_set(k) = hp.lambda * hp.phi^(k-1);
    gamma_set(k)  = hp.gamma  * hp.phi^(k-1);
end
hp.lambda_set = lambda_set; 
hp.gamma_set = gamma_set;
% hp.lambda_set = lambda_set ./ hp.num_layer;  % gradient needs to be revised
% hp.gamma_set = gamma_set ./ hp.num_layer;

% parameters used to optimize W
para_W.iter_max = 1000; % maximal iteration numbers
para_W.alpha = 1e-0; % initial step size
para_W.EPS_G = 1e-6;

% parameters used to optimize Z and L
para_ZL.iter_max = 1000; % maximal iteration numbers
para_ZL.alpha = 1e-0; % initial step size
para_ZL.EPS_G = 1e-6;
para_ZL.stepsize = 1e-3;   % tuned for syn data, 1e-3 for cla and 1e-6 for reg (parkinsons: 1e-8)
para_ZL.absTol = 1e-4;

outer_iter_max = 20;