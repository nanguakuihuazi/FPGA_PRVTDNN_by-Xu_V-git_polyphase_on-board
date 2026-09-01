%% PN-RVTDNN Polyphase (Direct Multi-Output Training)
%  目标:
%  1) 网络直接训练为多输出结构: 多样本输入 ->  16-> 16 -> 16 -> 2N
%  2) 多相分支数 N_poly, 延迟深度 M_poly = M + N_poly - 1
%  3) 输出顺序: [I(n),I(n-1),...,I(n-N+1), Q(n),Q(n-1),...,Q(n-N+1)]
% 在这个文件内测试多相里面的相位归一化的输入问题，具体来说，改变传统相位归一化的做法，对x(n-m)都分别做相位归一化，再相应反归一化

clear; clc; close all;
rng(50, 'twister');

%% ===================== 超参数设置 =====================
M = 5;
P = 3;
N_poly = 4;
M_poly = M + N_poly - 1;
hidden_sizes = [16, 16 , 16];
pn = true;

train_size = 8000;
val_size = 1000;
epochs = 3000;
batch_size = 128;
lr_init = 1e-3;
weight_decay = 1e-4;
grad_clip = 1.0;
patience_es = 50;
patience_lr = 20;
leaky_alpha = 0.125;

fprintf('========== Polyphase Direct-Train配置 ==========\n');
fprintf('M=%d, N=%d, M''=%d, 输出维度=%d, 输入复数项=%d\n', M, N_poly, M_poly, 2 * N_poly, 2*(M+N_poly));

%% ===================== 数据加载 =====================
% 数据位于本文件同级的 "Phased Array_Data" 文件夹
mat_path_primary = fullfile(fileparts(mfilename('fullpath')), 'Phased Array_Data', 'data_1011_40norm01.mat');

% 兼容旧路径作为后备（原主路径）
mat_path_fallback = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'Matlab_PRVTDNN_by Xu', 'Phased Array_Data', 'data_1011_40norm01.mat');

if exist(mat_path_primary, 'file')
    mat_path = mat_path_primary;
elseif exist(mat_path_fallback, 'file')
    mat_path = mat_path_fallback;
    fprintf('[提示] 使用后备数据路径: %s\n', mat_path);
else
    error('未找到 data_1011_40norm01.mat，请检查数据路径。');
end

data = load(mat_path);
config_norm = data.config_norm;
x_raw = data.x_50(:);
y_raw = data.y_50(:);

angle_col = config_norm(:, 2);
power_col = config_norm(:, 3);
unique_angles = unique(angle_col);
unique_powers = unique(power_col);
target_angle = unique_angles(end);
target_power = unique_powers(end);
mask = (angle_col == target_angle) & (power_col == target_power);
x_state = x_raw(mask);
y_state = y_raw(mask);

fprintf('[数据筛选] 角度=%.4f, 功率=%.4f, 样本数=%d\n', target_angle, target_power, length(x_state));

%% ===================== 多输出训练数据预处理 =====================
% X_in: (N_sample, input_dim)
% Y_out: (N_sample, 2*N_poly)
% rk_mat: (N_sample, N_poly), 每列对应一个输出分支的反归一化相位因子
[X_in, Y_out, rk_mat] = preprocess_data_pn_rvtdnn_multiout(x_state, y_state, M, P, N_poly, pn);

X_train = single(X_in(1:train_size, :));
Y_train = single(Y_out(1:train_size, :));
X_val = single(X_in(train_size+1:train_size+val_size, :));
Y_val = single(Y_out(train_size+1:train_size+val_size, :));
test_start = train_size + val_size + 1;
X_test = single(X_in(test_start:end, :));
Y_test = single(Y_out(test_start:end, :));
rk_test = rk_mat(test_start:end, :);

N_test = size(X_test, 1);

dlX_val = dlarray(X_val', 'CB');
dlY_val = dlarray(Y_val', 'CB');

%% ===================== 网络初始化 =====================
input_dim = size(X_train, 2);
output_dim = 2 * N_poly;
[params, netConfig] = initParamsMLP(input_dim, hidden_sizes, output_dim);
netConfig.leaky_alpha = leaky_alpha;
netConfig.N_poly = N_poly;

fprintf('Model Parameters: %d\n', countParamsMLP(params, netConfig));
fprintf('网络结构: %d -> 16 -> 16 -> 16 -> %d\n', input_dim, output_dim);

%% ===================== 训练循环 =====================
[avgGrad, avgSqGrad] = deal([]);
iteration = 0;
lr = lr_init;
best_val_loss = inf;
epochs_no_improve = 0;
lr_no_improve = 0;
best_params = params;
best_netConfig = netConfig;

N_train = size(X_train, 1);
num_batches = ceil(N_train / batch_size);

fprintf('========== 开始训练(多输出标签) ==========\n');
for epoch = 1:epochs
    shuffle_idx = randperm(N_train);
    epoch_loss = 0;

    for b = 1:num_batches
        idx_start = (b - 1) * batch_size + 1;
        idx_end = min(b * batch_size, N_train);
        batch_idx = shuffle_idx(idx_start:idx_end);
        cur_batch_size = length(batch_idx);

        dlX_batch = dlarray(X_train(batch_idx, :)', 'CB');
        dlY_batch = dlarray(Y_train(batch_idx, :)', 'CB');

        [loss, gradients] = dlfeval(@modelLoss, params, netConfig, dlX_batch, dlY_batch, weight_decay);
        gradients = clipGradients(gradients, grad_clip);

        iteration = iteration + 1;
        [params, avgGrad, avgSqGrad] = adamupdate(params, gradients, avgGrad, avgSqGrad, iteration, lr);

        epoch_loss = epoch_loss + extractdata(loss) * cur_batch_size;
    end

    avg_train_loss = epoch_loss / N_train;
    dlY_val_pred = modelForward(params, netConfig, dlX_val);
    val_loss = computeNmseLossDl(dlY_val_pred, dlY_val);
    val_loss_val = extractdata(val_loss);

    rk_val = rk_mat(train_size+1:train_size+val_size, :);
    val_nmse = computeNmseDbMulti(dlY_val_pred, dlY_val, rk_val, pn, N_poly);

    if mod(epoch, 10) == 1 || epoch == epochs
        fprintf('Epoch %d/%d, Train: %.6f, Val: %.6f, Val NMSE: %.4f dB\n', ...
            epoch, epochs, avg_train_loss, val_loss_val, val_nmse);
    end

    if val_loss_val < best_val_loss
        best_val_loss = val_loss_val;
        epochs_no_improve = 0;
        lr_no_improve = 0;
        best_params = params;
        best_netConfig = netConfig;
    else
        epochs_no_improve = epochs_no_improve + 1;
        lr_no_improve = lr_no_improve + 1;

        if lr_no_improve >= patience_lr
            lr = lr * 0.5;
            lr_no_improve = 0;
            fprintf('  >> 学习率衰减至 %.2e\n', lr);
        end

        if epochs_no_improve >= patience_es
            fprintf('Early stopping at epoch %d\n', epoch);
            params = best_params;
            netConfig = best_netConfig;
            break;
        end
    end
end

%% ===================== 模型保存与导出 =====================
model_dirs = {
    fullfile(fileparts(mfilename('fullpath')), 'Phased Array Model'), ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))), 'sim_All Top', 'Model')
};

for d = 1:numel(model_dirs)
    if ~exist(model_dirs{d}, 'dir')
        mkdir(model_dirs{d});
    end
end

output_order = '[I(n),I(n-1),I(n-2),I(n-3),Q(n),Q(n-1),Q(n-2),Q(n-3)]';
feature_order = '[Re(x(n..n-M_poly)/|.|), Im(x(n..n-M_poly)/|.|), |x(n..n-M_poly)|, |x(n..n-M_poly)|^2]';
rk_usage = 'Only rk_mat(:,1:N_poly) participates in output de-normalization; extra phase factors are input-side only.';
model_info = struct( ...
    'input_dim', input_dim, ...
    'output_dim', output_dim, ...
    'hidden_sizes', hidden_sizes, ...
    'M', M, ...
    'P', P, ...
    'pn', pn, ...
    'N_poly', N_poly, ...
    'M_poly', M_poly, ...
    'feature_order', feature_order, ...
    'output_order', output_order, ...
    'rk_usage', rk_usage);
for d = 1:numel(model_dirs)
    model_dir = model_dirs{d};
    poly_model_path = fullfile(model_dir, 'PRVTDNN_model_polyphase_multiout.mat');
    save(poly_model_path, 'best_params', 'best_netConfig', 'M', 'P', 'pn', ...
        'N_poly', 'M_poly', 'output_order', 'feature_order', 'rk_usage', 'model_info');
    fprintf('多输出多相模型已保存: %s\n', poly_model_path);

    export_dir_unpadded = fullfile(model_dir, 'unpadded');
    export_model_hex_q14(best_params, best_netConfig.num_layers, export_dir_unpadded, '_poly_multiout');
end

%% ===================== 零填充导出 =====================
% hidden_sizes_fpga = [16, 16, 16, 16];
% layers_fpga = [input_dim, hidden_sizes_fpga, output_dim];
% layers_actual = [input_dim, hidden_sizes, output_dim];

% do_pad = true;
% if length(hidden_sizes) ~= length(hidden_sizes_fpga)
%     do_pad = false;
% end
% for li = 1:length(hidden_sizes)
%     if hidden_sizes(li) > hidden_sizes_fpga(li)
%         do_pad = false;
%     end
% end

% if do_pad
%     num_layers_pad = length(layers_fpga) - 1;
%     params_padded = struct();

%     for i = 1:num_layers_pad
%         actual_in = layers_actual(i);
%         actual_out = layers_actual(i + 1);
%         fpga_in = layers_fpga(i);
%         fpga_out = layers_fpga(i + 1);

%         W_src = double(extractdata(best_params.(['W' num2str(i)])));
%         b_src = double(extractdata(best_params.(['b' num2str(i)])));

%         W_pad = zeros(fpga_out, fpga_in, 'double');
%         b_pad = zeros(fpga_out, 1, 'double');
%         W_pad(1:actual_out, 1:actual_in) = W_src;
%         b_pad(1:actual_out) = b_src;

%         params_padded.(['W' num2str(i)]) = dlarray(single(W_pad));
%         params_padded.(['b' num2str(i)]) = dlarray(single(b_pad));
%     end

%     netConfig_padded = best_netConfig;
%     netConfig_padded.hidden_sizes = hidden_sizes_fpga;
%     netConfig_padded.num_layers = num_layers_pad;

%     dlX_verify = dlarray(single(X_test(1:min(10, size(X_test, 1)), :))', 'CB');
%     y_orig = extractdata(modelForward(best_params, best_netConfig, dlX_verify));
%     y_pad = extractdata(modelForward(params_padded, netConfig_padded, dlX_verify));
%     max_err_pad = max(abs(y_orig(:) - y_pad(:)));
%     fprintf('[零填充验证] 最大误差: %.2e\n', max_err_pad);

%     poly_model_padded_path = fullfile(model_dir, 'PRVTDNN_model_polyphase_multiout_padded.mat');
%     save(poly_model_padded_path, 'params_padded', 'netConfig_padded', ...
%         'M', 'P', 'pn', 'N_poly', 'M_poly', 'layers_fpga', 'layers_actual', 'output_order');

%     export_dir_padded = fullfile(model_dir, 'padded');
%     export_model_hex_q14(params_padded, num_layers_pad, export_dir_padded, '_poly_multiout_padded');
%     fprintf('多输出多相零填充模型已保存: %s\n', poly_model_padded_path);
% end

%% ===================== 测试集评估 =====================
dlY_test_pred = modelForward(best_params, best_netConfig, dlarray(single(X_test)', 'CB'));
Y_test_pred = extractdata(dlY_test_pred)';

[I_pred, Q_pred] = splitIQMulti(Y_test_pred, N_poly);
[I_true, Q_true] = splitIQMulti(Y_test, N_poly);

Y_pred_c = I_pred + 1j * Q_pred;
Y_true_c = I_true + 1j * Q_true;

if pn
    Y_pred_c = Y_pred_c ./ (rk_test + 1e-10);
    Y_true_c = Y_true_c ./ (rk_test + 1e-10);
end

overall_nmse = compute_nmse(Y_true_c(:), Y_pred_c(:));
branch_nmse = zeros(N_poly, 1);
for j = 1:N_poly
    branch_nmse(j) = compute_nmse(Y_true_c(:, j), Y_pred_c(:, j));
end

fprintf('\n测试集 Overall NMSE: %.4f dB\n', overall_nmse);
for j = 1:N_poly
    fprintf('Branch-%d NMSE: %.4f dB\n', j-1, branch_nmse(j));
end

fprintf('多输出矩阵尺寸: (%d, %d)\n', N_test, size(Y_test_pred, 2));
fprintf('输出顺序: %s\n', output_order);

%% ===================== 多相定点验证 =====================
F = fimath('ProductMode', 'KeepMSB', 'SumMode', 'KeepMSB', ...
    'MaxProductWordLength', 32, 'MaxSumWordLength', 40, ...
    'ProductWordLength', 32, 'SumWordLength', 40, ...
    'ProductFractionLength', 30, 'SumFractionLength', 30);
T_uniform = numerictype('WordLength', 16, 'FractionLength', 14);

W = cell(best_netConfig.num_layers, 1);
b = cell(best_netConfig.num_layers, 1);
for i = 1:best_netConfig.num_layers
    W{i} = double(extractdata(best_params.(['W' num2str(i)])));
    b{i} = double(extractdata(best_params.(['b' num2str(i)])));
end

params_fi = struct();
for i = 1:best_netConfig.num_layers
    params_fi.(['W' num2str(i)]) = fi(W{i}, T_uniform, F);
    params_fi.(['b' num2str(i)]) = fi(b{i}, T_uniform, F);
end

Y_fixed_pred = modelForward_fixed_multi(params_fi, best_netConfig, double(X_test), F, T_uniform);
if pn
    Y_fixed_pred = Y_fixed_pred ./ (rk_test + 1e-10);
end

fixed_overall_nmse = compute_nmse(Y_true_c(:), Y_fixed_pred(:));
quant_nmse = compute_nmse(Y_pred_c(:), Y_fixed_pred(:));

fprintf('\n定点 Overall NMSE (真实 vs 定点): %.4f dB\n', fixed_overall_nmse);
fprintf('量化误差 NMSE (浮点 vs 定点): %.4f dB\n', quant_nmse);

fprintf('\n========== 全部完成 ==========\n');

%% ================================================================
%                           局部函数
%  ================================================================

function [X_in, Y_out, rk_mat] = preprocess_data_pn_rvtdnn_multiout(x, y, M, P, N_poly, pn)
    N = length(x);
    M_poly = M + N_poly - 1;

    X_delayed_full = zeros(N, M_poly + 1);
    for i = 0:M_poly
        if i < N
            X_delayed_full(i+1:end, i+1) = x(1:N-i);
        end
    end

    % 新型输入: 对每个 x(n-k) (k=0,...,M_poly) 单独做相位归一化
    % 相位归一化: x(n-k)/|x(n-k)| = e^{j*theta_k}, 保留相位 IQ 分量
    % 输入特征维度 = 2*(M+N_poly) + (M+N_poly)*P = (M+N_poly)*(2+P)
    % 注: k = N_poly,...,M_poly 的相位因子仅用于输入, 未对应任何输出分支(冗余)
    if pn
        X_norm = X_delayed_full ./ (abs(X_delayed_full) + 1e-10);  % 每个样本各自相位归一化 -> e^{j*theta_k}
    else
        X_norm = X_delayed_full;  % 不归一化时直接用原始 IQ
    end
    ReX_all = real(X_norm);   % (N, M_poly+1)
    ImX_all = imag(X_norm);   % (N, M_poly+1)

    % 幅度项
    A_matrix = [];
    R = abs(X_delayed_full);
    for p = 1:P
        A_matrix = [A_matrix, R.^p]; %#ok<AGROW>
    end
    X_in = [ReX_all, ImX_all, A_matrix];

    % 多输出标签: 每个分支一个目标 y(n-j), 共 N_poly 组 I/Q
    % rk_mat 仅保存输出分支需要的前 N_poly 个反归一化因子。
    I_targets = zeros(N, N_poly);
    Q_targets = zeros(N, N_poly);
    rk_mat = ones(N, N_poly);

    for j = 0:N_poly-1
        y_shift = zeros(N, 1);
        if j < N
            y_shift(j+1:end) = y(1:end-j);
        end

        if pn
            rk_j = conj(X_delayed_full(:, j+1)) ./ (abs(X_delayed_full(:, j+1)) + 1e-10);
            y_shift = y_shift .* rk_j;
            rk_mat(:, j+1) = rk_j;
        end

        I_targets(:, j+1) = real(y_shift);
        Q_targets(:, j+1) = imag(y_shift);
    end

    Y_out = [I_targets, Q_targets];
end

function [params, netConfig] = initParamsMLP(input_dim, hidden_sizes, output_dim)
    netConfig.hidden_sizes = hidden_sizes;
    layers = [input_dim, hidden_sizes, output_dim];
    netConfig.num_layers = length(layers) - 1;

    params = struct();
    for i = 1:netConfig.num_layers
        fan_in = layers(i);
        fan_out = layers(i+1);
        params.(['W' num2str(i)]) = dlarray(single(randn(fan_out, fan_in) * sqrt(2/fan_in)));
        params.(['b' num2str(i)]) = dlarray(single(zeros(fan_out, 1)));
    end
end

function n = countParamsMLP(params, netConfig)
    n = 0;
    for i = 1:netConfig.num_layers
        n = n + numel(params.(['W' num2str(i)])) + numel(params.(['b' num2str(i)]));
    end
end

function dlY = modelForward(params, netConfig, dlX)
    H = stripdims(dlX);
    alpha = netConfig.leaky_alpha;
    for i = 1:netConfig.num_layers - 1
        H = params.(['W' num2str(i)]) * H + params.(['b' num2str(i)]);
        H = leakyRelu(H, alpha); % 注释掉leakyRelu
       % H = tanh(H); % 新激活函数tanh
    end
    out_idx = netConfig.num_layers;
    dlY = params.(['W' num2str(out_idx)]) * H + params.(['b' num2str(out_idx)]);
end

function y = leakyRelu(x, alpha)
    y = max(x, alpha * x);
end

function loss = computeNmseLossDl(dlY_pred, dlY_true)
    dlY_true = stripdims(dlY_true);
    err = dlY_pred - dlY_true;
    loss = mean(sum(err.^2, 1)) / (mean(sum(dlY_true.^2, 1)) + 1e-10);
end

function [loss, gradients] = modelLoss(params, netConfig, dlX, dlY, weight_decay)
    dlY_pred = modelForward(params, netConfig, dlX);
    loss = computeNmseLossDl(dlY_pred, dlY);

    l2 = dlarray(single(0));
    for i = 1:netConfig.num_layers
        l2 = l2 + sum(params.(['W' num2str(i)])(:).^2);
    end
    loss = loss + (weight_decay / 2) * l2;
    gradients = dlgradient(loss, params);
end

function gradients = clipGradients(gradients, maxNorm)
    totalNormSq = 0;
    fnames = fieldnames(gradients);
    for i = 1:numel(fnames)
        g = gradients.(fnames{i});
        totalNormSq = totalNormSq + sum(extractdata(g(:)).^2);
    end
    totalNorm = sqrt(totalNormSq);

    if totalNorm > maxNorm
        scale = maxNorm / totalNorm;
        for i = 1:numel(fnames)
            gradients.(fnames{i}) = gradients.(fnames{i}) * scale;
        end
    end
end

function nmse_db = computeNmseDbMulti(dlY_pred, dlY_true, rk_val, pn, N_poly)
    Y_pred = extractdata(dlY_pred)';
    Y_true = extractdata(dlY_true)';

    [I_pred, Q_pred] = splitIQMulti(Y_pred, N_poly);
    [I_true, Q_true] = splitIQMulti(Y_true, N_poly);

    p_c = I_pred + 1j * Q_pred;
    t_c = I_true + 1j * Q_true;

    if pn
        p_c = p_c ./ (rk_val + 1e-10);
        t_c = t_c ./ (rk_val + 1e-10);
    end

    nmse_db = 10 * log10(mean(abs(p_c(:) - t_c(:)).^2) / mean(abs(t_c(:)).^2));
end

function [I_part, Q_part] = splitIQMulti(Y, N_poly)
    I_part = Y(:, 1:N_poly);
    Q_part = Y(:, N_poly+1:2*N_poly);
end

function nmse = compute_nmse(x, y)
    error = x(:) - y(:);
    nmse = 10 * log10(mean(abs(error).^2) / mean(abs(x(:)).^2));
end

function Y_pred_c = modelForward_fixed_multi(params_fi, netConfig, X_feat, F, T_uniform)
    X_fi = fi(X_feat', T_uniform, F);
    H = X_fi;
    alpha = netConfig.leaky_alpha;

    for i = 1:netConfig.num_layers - 1
        H = params_fi.(['W' num2str(i)]) * H;
        H_d = double(H) + double(params_fi.(['b' num2str(i)]));
        H = fi(H_d, T_uniform, F);

        H_d = double(H);
        neg_mask = H_d < 0;
        H_d(neg_mask) = alpha * H_d(neg_mask);
        H = fi(H_d, T_uniform, F);
    end

    i = netConfig.num_layers;
    H = params_fi.(['W' num2str(i)]) * H;
    H_d = double(H) + double(params_fi.(['b' num2str(i)]));

    % 输出 H_d 维度: (2*N_poly, N_sample)
    Y = H_d';
    N_poly = netConfig.N_poly;
    I_out = Y(:, 1:N_poly);
    Q_out = Y(:, N_poly+1:2*N_poly);
    Y_pred_c = I_out + 1j * Q_out;
end

function y = export_float_to_q14(x)
    scaled = round(x * 2^14);
    scaled = min(max(scaled, -32768), 32767);
    y = int16(scaled);
end

function export_write_hex16(filepath, data)
    fid = fopen(filepath, 'w');
    if fid < 0
        error('无法创建文件: %s', filepath);
    end
    for k = 1:numel(data)
        v = int32(data(k));
        if v < 0
            v = v + 65536;
        end
        fprintf(fid, '%04X\n', v);
    end
    fclose(fid);
end

function export_model_hex_q14(params_struct, num_layers, export_dir, suffix)
    if ~exist(export_dir, 'dir')
        mkdir(export_dir);
    end

    if nargin < 4
        suffix = '';
    end

    fprintf('\n[导出] 写入 %s\n', export_dir);
    all_weights_vec = int16([]);

    for i = 1:num_layers
        W_exp = double(extractdata(params_struct.(['W' num2str(i)])));
        b_exp = double(extractdata(params_struct.(['b' num2str(i)])));

        W_q = export_float_to_q14(W_exp);
        b_q = export_float_to_q14(b_exp);

        w_vec = reshape(W_q.', [], 1);
        b_vec = b_q(:);

        w_hex = fullfile(export_dir, sprintf('layer%d_weights%s.hex', i, suffix));
        b_hex = fullfile(export_dir, sprintf('layer%d_bias%s.hex', i, suffix));
        export_write_hex16(w_hex, w_vec);
        export_write_hex16(b_hex, b_vec);

        all_weights_vec = [all_weights_vec; w_vec; b_vec]; %#ok<AGROW>

        fprintf('  Layer %d: W(%d x %d), b(%d)\n', i, size(W_exp,1), size(W_exp,2), numel(b_vec));
    end

    all_w_hex = fullfile(export_dir, sprintf('all_weights%s.hex', suffix));
    export_write_hex16(all_w_hex, all_weights_vec);
    fprintf('[导出完成] %s, 参数总数=%d\n', all_w_hex, numel(all_weights_vec));
end
