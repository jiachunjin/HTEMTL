% define the problem size
D = 150;
num_subspace = 4;
ss_dimension = [3, 4, 4, 5];
num_task_one_ss = [4, 5, 5, 6];
num_sample_per_task = 200;
T = sum(num_task_one_ss);

% generate the basis for each subspace
B = {};
A = randn(D, D);
[Q, R] = qr(A);
tot_d = 0;
for i = 1:num_subspace
    ss_d = ss_dimension(i);
    B{i} = Q(:, tot_d+1:tot_d+ss_d);
    tot_d = tot_d + ss_d;
end

% generate the weights
W_star = zeros(D, T);
w_index = 1;

for i = 1:num_subspace
    basis = B{i};
    ss_d = size(basis, 2);
    for j = 1:num_task_one_ss(i)
        coeff = rand(ss_d, 1) - 0.5;
        W_star(:, w_index) = basis * coeff;
        w_index = w_index + 1;
    end
end

% generate data regression
for i = 1:T
    X_train_cla{i} = randn(num_sample_per_task, D);
    y_train_cla{i} = sign(X_train_cla{i} * W_star(:, i) + 0.1 * randn(num_sample_per_task, 1));
    neg_index = find(y_train_cla{i} == -1);
    y_train_cla{i}(neg_index) = 0;
    X_val_cla{i} = randn(100, D);
    X_test_cla{i} = randn(100, D);
    y_val_cla{i} = sign(X_val_cla{i} * W_star(:, i) + 0.1 * randn(100, 1));
    neg_index = find(y_val_cla{i} == -1);
    y_val_cla{i}(neg_index) = 0;

    y_test_cla{i} = sign(X_test_cla{i} * W_star(:, i)+ 0.1 * randn(100, 1));
    neg_index = find(y_test_cla{i} == -1);
    y_test_cla{i}(neg_index) = 0;
end

% save('synthetic_01.mat', 'X_train_cla', 'X_val_cla', 'X_test_cla', 'y_train_cla', 'y_val_cla', 'y_test_cla', 'W_star', 'B', 'D', 'T');
