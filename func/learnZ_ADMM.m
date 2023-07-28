%% learnZ_ADMM:
function [Z] = learnZ_ADMM(Z, L, W, lambda, gamma, para_Z, num_task)
	debug = false;
	if debug
		disp('======================');
		nuclear_norm = gamma * (norm(svd(Z),1));
		Z_term = lambda / 2 * norm(W - W * Z - L * W, 'fro') ^ 2 + nuclear_norm;
		fprintf('-------Before updating Z: %d\n', Z_term);
	end
% 	[~, num_task] = size(W);
	U = zeros(num_task, num_task); % scaled dual varabile
	iter = 0;
	converged = false;
	s = Inf;
	while ~converged
		iter = iter + 1;
% 		Z = zeros(num_task, num_task);
		Z_prev = Z; % used to compute the dual residual
		% (1) update J via SVT
		[U_svd, S_T, V_svd] = svd(Z + U, 'econ');
		x_J = wthresh(diag(S_T), 's', gamma / para_Z.rho);
		J = U_svd * diag(x_J) * V_svd';
		% (2) update Z with closed form solution
		A = eye(num_task) + (lambda / para_Z.rho) * (W' * W);
		C = J - U + (lambda / para_Z.rho) * (W' * (W - L * W));
		Z = A \ C;
		% (3) update scaled dual variable U
		r = Z - J; % primal residual
		U = U + r;
		if iter > 1
			% compute the dual residual
			s = para_Z.rho * (Z - Z_prev);
		end
		EPS_pri = para_Z.EPS_abs * num_task + para_Z.EPS_rel * max(norm(Z), norm(J));
		EPS_dual = para_Z.EPS_abs * num_task + para_Z.EPS_rel * para_Z.rho * norm(U);

		% [TO DO] vary rho here
		if iter > 1
			% vary the penalty parameter
			if norm(r) > para_Z.mu * norm(s)
				para_Z.rho = para_Z.rho * para_Z.tau_incr;
				U = (1 / para_Z.tau_incr) * U;
			elseif norm(s) > para_Z.mu * norm(r)
				para_Z.rho = para_Z.rho / para_Z.tau_decr;
				U = para_Z.tau_decr * U;
			end
		end

		% stopping criteria
		if norm(r) < EPS_pri && norm(s) < EPS_dual
			converged = true;
		end
		if ~converged && iter >= para_Z.iter_max
			% disp('Maximum Z iterations reached') ;
			converged = true ;
		end
	end % end of the while loop
	if debug
		nuclear_norm = gamma * (norm(svd(Z),1));
		Z_term = lambda / 2 * norm(W - W * Z - L * W, 'fro') ^ 2 + nuclear_norm;
		fprintf('	   After updating Z: %d\n', Z_term);
	end
end % end of the function