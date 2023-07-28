%% learnL_ADMM: function description
function [L] = learnL_ADMM(L, Z, W, lambda, gamma, para_L, num_dim)
	debug = false;
	if debug
		disp('======================');
		nuclear_norm =norm(svd(L),1);
		L_term = lambda / 2 * norm(W - W * Z - L * W, 'fro') ^ 2 + gamma * nuclear_norm;
		fprintf('-------Before updating L: %d\n', L_term);
	end
% 	[num_dim, ~] = size(W);
	U = zeros(num_dim, num_dim); % scaled dual varabile
	iter = 0;
	converged = false;
	s = Inf;

	while ~converged
		iter = iter + 1;
% 		L = zeros(num_dim, num_dim);
		L_prev = L; % used to compute the dual residual
		% (1) update S via SVT
		[U_svd, S_T, V_svd] = svd(L + U, 'econ');
		x_S = wthresh(diag(S_T), 's', gamma / para_L.rho);
		S = U_svd * diag(x_S) * V_svd';
		% (2) update L with closed form solution
		A = eye(num_dim) + (lambda / para_L.rho) * (W * W');
		B = (lambda / para_L.rho) * ((W - W * Z) * W') + S - U;
        L = B / A;

		% (3) update scaled dual variable U
		r = L - S;
		U = U + r;
		if iter > 1
			% compute the dual residual
			s = para_L.rho * (L - L_prev);
		end
		EPS_pri = para_L.EPS_abs * num_dim + para_L.EPS_rel * max(norm(L), norm(S));
		EPS_dual = para_L.EPS_abs * num_dim + para_L.EPS_rel * para_L.rho * norm(U);
		% (4) vary the penalty parameter
		if iter > 1
			if norm(r) > para_L.mu * norm(s)
				para_L.rho = para_L.rho * para_L.tau_incr;
				U = (1 / para_L.tau_incr) * U;
			elseif norm(s) > para_L.mu * norm(r)
				para_L.rho = para_L.rho / para_L.tau_decr;
				U = para_L.tau_decr * U;
			end
		end
		% fprintf('norm(r): %d, EPS_pri: %d, norm(s): %d, EPS_dual: %d\n', norm(r), EPS_pri, norm(s), EPS_dual);
		% stopping criteria
		if norm(r) < EPS_pri && norm(s) < EPS_dual
			converged = true;
		end
		if ~converged && iter >= para_L.iter_max
			% disp('Maxipara_L.mum L iterations reached');
			converged = true ;
		end
	end % end of the while loop
	if debug
		nuclear_norm =norm(svd(L),1);
		L_term = lambda / 2 * norm(W - W * Z - L * W, 'fro') ^ 2 + gamma * nuclear_norm;
		fprintf('	   After updating L: %d\n', L_term);
	end
end % end of the function