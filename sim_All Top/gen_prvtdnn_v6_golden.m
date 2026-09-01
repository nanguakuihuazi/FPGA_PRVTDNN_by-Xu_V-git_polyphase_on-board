%% gen_prvtdnn_v6_golden.m
% 为 PRVTDNN Polyphase (45→16→16→16→8, N_POLY=4) 生成端到端仿真 Golden 数据。
%
% 功能:
%   1. 从 Input/data_in.hex 读取 128-bit 并行输入, 解码为串行 IQ 样本
%   2. 精确模拟 preprocess.v 的定点行为 (Per-Tap Phase Normalization):
%      TAP_NUM=9 延迟线 → 每 tap 独立 CORDIC VECTORING (得 R[k], θ[k])
%      → CORDIC ROTATION (1,0,θ[k]) 得单位向量 (cos θ_k, sin θ_k)
%      → 前 N_POLY=4 个 tap 额外计算 rk[j] = ROTATION(1,0,-θ[j])
%      → R²[k]=trunc(R[k]²·2^{-14}), R³[k]=trunc(R[k]·R²[k]·2^{-14})
%      → 特征向量 [Re_0..Re_8, Im_0..Im_8, R_0..R_8, R²_0..R²_8, R³_0..R³_8]
%   3. 从 Model (优先 Model/unpadded) 逐层 hex 文件加载权重/偏置
%   4. 精确匹配 FPGA 定点前向传播 (4层, int64 MAC, Leaky ReLU 右移3位):
%      Layer 1: 45→16  Leaky ReLU
%      Layer 2: 16→16  Leaky ReLU
%      Layer 3: 16→16  Leaky ReLU
%      Layer 4: 16→8   Linear (无激活, 输出层)
%   5. 模拟 phase_denorm.v 的定点运算 (N_POLY=4 路并行, 乘法+右移14+饱和截断)
%   6. 将各层 golden 输出写入 Gold/
%
% 输出文件 (Gold/):
%   preprocess_feat_golden.hex      — NUM_CYCLES×45 行 (sample-major, feat 顺序 k=0..8)
%   preprocess_rk_re_golden.hex     — NUM_CYCLES×4  行 (sample-major, rk[0..3] 实部)
%   preprocess_rk_im_golden.hex     — NUM_CYCLES×4  行 (sample-major, rk[0..3] 虚部)
%   layer1_golden.hex               — NUM_CYCLES×16 行 (sample-major, neuron 顺序)
%   layer2_golden.hex               — NUM_CYCLES×16 行
%   layer3_golden.hex               — NUM_CYCLES×16 行
%   layer4_golden.hex               — NUM_CYCLES×8  行 (Output Layer, 线性输出)
%   phase_denorm_golden_i.hex       — NUM_CYCLES×4  行 (sample-major, branch 0..3 I分量)
%   phase_denorm_golden_q.hex       — NUM_CYCLES×4  行 (sample-major, branch 0..3 Q分量)
%
% 注意:
%   所有定点仿真严格按 FPGA 整数算术, 精度与硬件完全匹配 (±0~1 LSB 截断误差)。

clear; clc;

%% ===================== 路径配置 =====================
script_dir  = fileparts(mfilename('fullpath'));
input_dir   = fullfile(script_dir, 'Input');
model_dir   = fullfile(script_dir, 'Model');
out_dir     = fullfile(script_dir, 'Gold');

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% ===================== 常量 (与 prvtdnn_defines.vh 一致) =====================
FRAC_BITS   = 14;
SCALE14     = 2^FRAC_BITS;
LEAKY_SHIFT = 3;           % alpha = 2^(-3) = 0.125

M           = 5;           % 记忆深度
N_POLY      = 4;           % 多相分支数
TAP_NUM     = M + N_POLY;  % = 9 (TAP_NUM = M_POLY + 1 = M + N_POLY - 1 + 1)
P           = 3;           % 包络阶数 (R, R², R³)
FEAT_DIM    = 5 * TAP_NUM; % = 45 (Re, Im, R, R², R³ 各9维)

% 网络结构 (与 prvtdnn_defines.vh 一致)
NUM_LAYERS  = 4;
layer_dims  = [45, 16;    % Layer 1 (HL1)
               16, 16;    % Layer 2 (HL2)
               16, 16;    % Layer 3 (HL3)
               16,  8];   % Layer 4 (OL, 输出层)
has_leaky_relu = [1, 1, 1, 0];  % 最后一层线性, 无激活

%% ===================== Step 1: 读取并解码 128-bit 输入 =====================
fprintf('Step 1: 读取并解码 128-bit 输入数据...\n');

data_in_path = fullfile(input_dir, 'data_in.hex');
[x_re_q14, x_im_q14, NUM_CYCLES] = decode_data_in_128bit(data_in_path, N_POLY);
NUM_SAMPLES = NUM_CYCLES * N_POLY;

fprintf('  读取 %d 个 FPGA 周期 (%d 个串行样本)\n', NUM_CYCLES, NUM_SAMPLES);
fprintf('  re 范围: [%d, %d], im 范围: [%d, %d]\n', ...
    min(x_re_q14), max(x_re_q14), min(x_im_q14), max(x_im_q14));

%% ===================== Step 2: 模拟 preprocess.v (Per-Tap 定点 CORDIC) =====================
fprintf('Step 2: 模拟 preprocess (Per-Tap CORDIC 定点仿真)...\n');

golden_feat    = zeros(NUM_CYCLES, FEAT_DIM, 'int16');
golden_rk_re   = zeros(NUM_CYCLES, N_POLY,   'int16');
golden_rk_im   = zeros(NUM_CYCLES, N_POLY,   'int16');

% 延迟线 (初始全零, 与 FPGA rst_n 后一致)
tap_re_buf = zeros(TAP_NUM, 1, 'int16');
tap_im_buf = zeros(TAP_NUM, 1, 'int16');

for cyc = 1:NUM_CYCLES
    % 当前周期的 N_POLY 个样本 (从 oldest 到 newest)
    % x_re_q14 排列: [oldest...newest] for each cycle
    base_idx = (cyc - 1) * N_POLY;
    
    % 更新延迟线: 先移位后插入 (与 preprocess.v shift register 行为一致)
    % tap[0..N_POLY-1] = 当前周期的 x[n]..x[n-3]
    % tap[N_POLY..TAP_NUM-1] = 上一周期的 tap[0..M-1]
    for k = TAP_NUM:-1:(N_POLY+1)
        tap_re_buf(k) = tap_re_buf(k - N_POLY);
        tap_im_buf(k) = tap_im_buf(k - N_POLY);
    end
    % 插入当前周期的 N_POLY 个样本 (newest first in tap[0])
    for k = 1:N_POLY
        % tap[k-1] = x[n-(k-1)] = x_samples[N_POLY - k + 1] (newest at k=1)
        sample_idx = base_idx + (N_POLY - k + 1);
        tap_re_buf(k) = x_re_q14(sample_idx);
        tap_im_buf(k) = x_im_q14(sample_idx);
    end

    % 对每个 tap 做 CORDIC VECTORING → (R[k], θ[k])
    R_float     = zeros(TAP_NUM, 1);
    theta_float = zeros(TAP_NUM, 1);
    for k = 1:TAP_NUM
        re_f = double(tap_re_buf(k)) / SCALE14;
        im_f = double(tap_im_buf(k)) / SCALE14;
        [R_float(k), theta_float(k)] = cordic_vectoring_fixed(re_f, im_f);
    end

    % 对每个 tap 做 CORDIC ROTATION (1, 0, +θ[k]) → (cos θ_k, sin θ_k)
    unit_re_int = zeros(TAP_NUM, 1, 'int16');
    unit_im_int = zeros(TAP_NUM, 1, 'int16');
    for k = 1:TAP_NUM
        [ure, uim] = cordic_rotation_fixed(1.0, 0.0, theta_float(k));
        unit_re_int(k) = int16(round(ure * SCALE14));
        unit_im_int(k) = int16(round(uim * SCALE14));
    end

    % 对前 N_POLY 个 tap 做 CORDIC ROTATION (1, 0, -θ[j]) → rk[j]
    rk_re_int = zeros(N_POLY, 1, 'int16');
    rk_im_int = zeros(N_POLY, 1, 'int16');
    for j = 1:N_POLY
        [rre, rim] = cordic_rotation_fixed(1.0, 0.0, -theta_float(j));
        rk_re_int(j) = int16(round(rre * SCALE14));
        rk_im_int(j) = int16(round(rim * SCALE14));
    end

    % R → Q1.14 整数值
    R_int16 = zeros(TAP_NUM, 1, 'int16');
    for k = 1:TAP_NUM
        R_int16(k) = int16(round(R_float(k) * SCALE14));
    end

    % R² = trunc(R × R / 2^14), 16-bit 绕回
    R2_int16 = zeros(TAP_NUM, 1, 'int16');
    for k = 1:TAP_NUM
        r_i = int32(R_int16(k));
        r_sq = r_i * r_i;
        r2_raw = int32(floor(double(r_sq) / SCALE14));
        R2_int16(k) = int16(mod(int64(r2_raw) + 32768, 65536) - 32768);
    end

    % R³ = trunc(R × R² / 2^14), 16-bit 绕回
    R3_int16 = zeros(TAP_NUM, 1, 'int16');
    for k = 1:TAP_NUM
        r_i  = int32(R_int16(k));
        r2_i = int32(R2_int16(k));
        r_cub = r_i * r2_i;
        r3_raw = int32(floor(double(r_cub) / SCALE14));
        R3_int16(k) = int16(mod(int64(r3_raw) + 32768, 65536) - 32768);
    end

    % 组装特征向量 (FEAT_DIM=45)
    golden_feat(cyc, 1:TAP_NUM)              = unit_re_int;
    golden_feat(cyc, TAP_NUM+1:2*TAP_NUM)   = unit_im_int;
    golden_feat(cyc, 2*TAP_NUM+1:3*TAP_NUM) = R_int16;
    golden_feat(cyc, 3*TAP_NUM+1:4*TAP_NUM) = R2_int16;
    golden_feat(cyc, 4*TAP_NUM+1:5*TAP_NUM) = R3_int16;

    golden_rk_re(cyc, :) = rk_re_int;
    golden_rk_im(cyc, :) = rk_im_int;
end

fprintf('  特征矩阵维度: [%d, %d]\n', size(golden_feat, 1), size(golden_feat, 2));
fprintf('  rk_re 范围: [%d, %d], rk_im 范围: [%d, %d]\n', ...
    min(golden_rk_re(:)), max(golden_rk_re(:)), ...
    min(golden_rk_im(:)), max(golden_rk_im(:)));

%% ===================== Step 2.5: 导出 preprocess golden =====================
fprintf('Step 2.5: 导出 preprocess golden 数据...\n');

feat_flat    = reshape(golden_feat.',   [], 1);
rk_re_flat   = reshape(golden_rk_re.', [], 1);
rk_im_flat   = reshape(golden_rk_im.', [], 1);

write_hex_int16(fullfile(out_dir, 'preprocess_feat_golden.hex'),    feat_flat);
write_hex_int16(fullfile(out_dir, 'preprocess_rk_re_golden.hex'),   rk_re_flat);
write_hex_int16(fullfile(out_dir, 'preprocess_rk_im_golden.hex'),   rk_im_flat);

fprintf('  preprocess_feat_golden:  %d 行\n', numel(feat_flat));
fprintf('  preprocess_rk_re_golden: %d 行\n', numel(rk_re_flat));
fprintf('  preprocess_rk_im_golden: %d 行\n', numel(rk_im_flat));

%% ===================== Step 3: 从 Model 逐层 hex 加载权重 =====================
fprintf('Step 3: 从 Model 逐层 hex 加载权重...\n');

[W_q, b_q] = load_model_hex(model_dir, layer_dims);
for li = 1:NUM_LAYERS
    fprintf('  W%d: [%d×%d], b%d: [%d×1]\n', li, size(W_q{li},1), size(W_q{li},2), li, numel(b_q{li}));
end

%% ===================== Step 4: 定点前向传播 (精确匹配 FPGA) =====================
fprintf('Step 4: 定点前向传播 (%d 周期, %d 层)...\n', NUM_CYCLES, NUM_LAYERS);

H = int16(golden_feat.');   % [45, NUM_CYCLES] int16 列格式

for layer = 1:NUM_LAYERS
    in_dim  = layer_dims(layer, 1);
    out_dim = layer_dims(layer, 2);

    Y = zeros(out_dim, NUM_CYCLES, 'int16');

    for n = 1:NUM_CYCLES
        x_n = int64(H(:, n));   % [in_dim, 1] int64

        for o = 1:out_dim
            % MAC: Q1.14 × Q1.14 累加到 Q2.28 (int64, 无溢出)
            acc = int64(0);
            for k = 1:in_dim
                acc = acc + int64(W_q{layer}(o, k)) * x_n(k);
            end

            % 偏置: Q1.14 左移14 → Q2.28 域加法
            sum_with_bias = acc + bitshift(int64(b_q{layer}(o)), FRAC_BITS);

            % 截断: 算术右移14 → Q1.14, 16-bit 绕回
            trunc_q14 = wrap_to_int16(bitshift(sum_with_bias, -FRAC_BITS));

            % 激活: Leaky ReLU (alpha=0.125 = 2^-3)
            if has_leaky_relu(layer) && trunc_q14 < 0
                act_q14 = wrap_to_int16(int32(bitshift(int32(trunc_q14), -LEAKY_SHIFT)));
            else
                act_q14 = trunc_q14;
            end

            Y(o, n) = int16(act_q14);
        end
    end

    % 写 golden hex
    g_flat = reshape(Y, [], 1);
    g_path = fullfile(out_dir, sprintf('layer%d_golden.hex', layer));
    write_hex_int16(g_path, g_flat);
    fprintf('  Layer %d (%d→%d): %d 值, 写入 %s\n', ...
            layer, in_dim, out_dim, numel(g_flat), g_path);

    H = Y;
end

%% ===================== Step 5: phase_denorm 定点仿真 (N_POLY=4 路并行) =====================
fprintf('Step 5: phase_denorm 定点仿真 (%d 分支并行)...\n', N_POLY);

pd_golden_i = zeros(NUM_CYCLES, N_POLY, 'int16');
pd_golden_q = zeros(NUM_CYCLES, N_POLY, 'int16');

for n = 1:NUM_CYCLES
    for jj = 1:N_POLY
        Y_I = int64(H(jj,       n));  % I 分量 (branch jj-1)
        Y_Q = int64(H(N_POLY+jj, n)); % Q 分量 (branch jj-1)
        rk_re_j = int64(golden_rk_re(n, jj));
        rk_im_j = int64(golden_rk_im(n, jj));

        % Q1.14 × Q1.14 → Q2.28 (int64)
        out_I_raw = Y_I * rk_re_j + Y_Q * rk_im_j;
        out_Q_raw = Y_Q * rk_re_j - Y_I * rk_im_j;

        % 算术右移14 → Q1.14, 饱和截断至 int16
        pd_golden_i(n, jj) = saturate_int16(bitshift(out_I_raw, -FRAC_BITS));
        pd_golden_q(n, jj) = saturate_int16(bitshift(out_Q_raw, -FRAC_BITS));
    end
end

% 写文件
pd_i_flat = reshape(pd_golden_i.', [], 1);
pd_q_flat = reshape(pd_golden_q.', [], 1);
write_hex_int16(fullfile(out_dir, 'phase_denorm_golden_i.hex'), pd_i_flat);
write_hex_int16(fullfile(out_dir, 'phase_denorm_golden_q.hex'), pd_q_flat);
fprintf('  phase_denorm golden: %d×%d=%d 值/分量\n', NUM_CYCLES, N_POLY, NUM_CYCLES*N_POLY);

%% ===================== 完成汇总 =====================
fprintf('\n========== 生成完成 ==========\n');
fprintf('输出目录: %s\n', out_dir);
files = dir(fullfile(out_dir, '*.hex'));
for k = 1:numel(files)
    finfo = dir(fullfile(out_dir, files(k).name));
    fid_c = fopen(fullfile(out_dir, files(k).name), 'r');
    cnt = 0;
    while ~feof(fid_c)
        line = fgetl(fid_c);
        if ischar(line), cnt = cnt + 1; end
    end
    fclose(fid_c);
    fprintf('  %-44s  (%5d 行, %5.1f KB)\n', files(k).name, cnt, finfo.bytes/1024);
end
fprintf('================================\n');

%% =====================================================================
%  局部辅助函数
%  =====================================================================

function [x_re_q14, x_im_q14, num_cycles] = decode_data_in_128bit(filepath, N_POLY)
% 从 data_in.hex 读取 128-bit 并行数据并解码为串行 IQ 样本 (Q1.14 int16)
%
% 位域约定 (与 prvtdnn_top.v Section 4b 一致):
%   x_re(n-k): {data_in[(24-8k)+:8], data_in[(56-8k)+:8]}, k=0..3
%   x_im(n-k): {data_in[(88-8k)+:8], data_in[(120-8k)+:8]}, k=0..3
%
% 输出按 oldest→newest 排列 (与串行样本时间顺序一致)

    fid = fopen(filepath, 'r');
    if fid < 0
        error('无法打开文件: %s', filepath);
    end
    
    lines = {};
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line) && ~isempty(strtrim(line))
            lines{end+1} = strtrim(line); %#ok<AGROW>
        end
    end
    fclose(fid);
    
    num_cycles = numel(lines);
    num_samples = num_cycles * N_POLY;
    
    x_re_q14 = zeros(num_samples, 1, 'int16');
    x_im_q14 = zeros(num_samples, 1, 'int16');
    
    for cyc = 1:num_cycles
        hex_str = lines{cyc};
        % 确保是 32 字符 (128 位)
        if length(hex_str) ~= 32
            error('第 %d 行 hex 字符数不是 32: %s', cyc, hex_str);
        end
        
        % 解析为 128-bit 整数 (MATLAB uint64 不够, 用分段处理)
        % data_in[127:0] 对应 hex_str(1:32), MSB first
        % byte[15] = hex_str(1:2) = data_in[127:120]
        % byte[0]  = hex_str(31:32) = data_in[7:0]
        bytes = zeros(16, 1, 'uint8');
        for b = 1:16
            bytes(17-b) = uint8(hex2dec(hex_str((b-1)*2+1 : b*2)));
        end
        
        % 提取 cur_re[k] 和 cur_im[k], k=0..3
        % cur_re[0] = {data_in[31:24], data_in[63:56]} = {bytes(4), bytes(8)}
        % cur_re[1] = {data_in[23:16], data_in[55:48]} = {bytes(3), bytes(7)}
        % cur_re[2] = {data_in[15:8],  data_in[47:40]} = {bytes(2), bytes(6)}
        % cur_re[3] = {data_in[7:0],   data_in[39:32]} = {bytes(1), bytes(5)}
        % cur_im[0] = {data_in[95:88], data_in[127:120]} = {bytes(12), bytes(16)}
        % cur_im[1] = {data_in[87:80], data_in[119:112]} = {bytes(11), bytes(15)}
        % cur_im[2] = {data_in[79:72], data_in[111:104]} = {bytes(10), bytes(14)}
        % cur_im[3] = {data_in[71:64], data_in[103:96]}  = {bytes(9),  bytes(13)}
        
        cur_re = zeros(4, 1, 'int16');
        cur_im = zeros(4, 1, 'int16');
        
        for k = 0:3
            % cur_re[k] = {bytes(4-k), bytes(8-k)} (MSB, LSB)
            re_msb = bytes(4 - k);
            re_lsb = bytes(8 - k);
            re_u16 = uint16(re_msb) * 256 + uint16(re_lsb);
            cur_re(k+1) = typecast(re_u16, 'int16');
            
            % cur_im[k] = {bytes(12-k), bytes(16-k)} (MSB, LSB)
            im_msb = bytes(12 - k);
            im_lsb = bytes(16 - k);
            im_u16 = uint16(im_msb) * 256 + uint16(im_lsb);
            cur_im(k+1) = typecast(im_u16, 'int16');
        end
        
        % cur_re[0] = x(n), cur_re[3] = x(n-3) (newest first)
        % 输出排列: oldest→newest, 即 x(n-3), x(n-2), x(n-1), x(n)
        base_idx = (cyc - 1) * N_POLY;
        for k = 0:3
            x_re_q14(base_idx + 4 - k) = cur_re(k+1);  % k=0 → idx=4, k=3 → idx=1
            x_im_q14(base_idx + 4 - k) = cur_im(k+1);
        end
    end
end

function [W_q, b_q] = load_model_hex(model_dir, layer_dims)
% 从 Model 逐层 hex 文件读取权重/偏置。
% 优先使用 Model/unpadded；若不存在则回退到 Model。

    src_dir = fullfile(model_dir, 'unpadded');
    if ~exist(src_dir, 'dir')
        src_dir = model_dir;
    end

    fprintf('  权重目录: %s\n', src_dir);

    num_layers = size(layer_dims, 1);
    W_q = cell(num_layers, 1);
    b_q = cell(num_layers, 1);

    for li = 1:num_layers
        w_path = fullfile(src_dir, sprintf('layer%d_weights_poly_multiout.hex', li));
        b_path = fullfile(src_dir, sprintf('layer%d_bias_poly_multiout.hex', li));

        w_flat = read_hex_int16_file(w_path);
        b_vec  = read_hex_int16_file(b_path);

        in_dim  = layer_dims(li, 1);
        out_dim = layer_dims(li, 2);

        if numel(w_flat) ~= in_dim * out_dim
            error('Layer %d 权重个数不匹配: 期望 %d, 实际 %d (%s)', ...
                li, in_dim * out_dim, numel(w_flat), w_path);
        end
        if numel(b_vec) ~= out_dim
            error('Layer %d 偏置个数不匹配: 期望 %d, 实际 %d (%s)', ...
                li, out_dim, numel(b_vec), b_path);
        end

        W_q{li} = reshape(w_flat, [in_dim, out_dim]).';
        b_q{li} = b_vec(:);
    end
end

function vals = read_hex_int16_file(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('无法打开文件: %s', path);
    end

    vals = int16([]);
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if ~ischar(line) || isempty(line)
            continue;
        end
        u16 = uint16(hex2dec(line));
        vals(end+1, 1) = typecast(u16, 'int16'); %#ok<AGROW>
    end
    fclose(fid);
end

function y = wrap_to_int16(x)
    % 将任意整数 wrap 到 int16 范围 [-32768, 32767]
    y = int32(mod(int64(x) + 32768, 65536) - 32768);
end

function y = saturate_int16(x)
    % 饱和截断到 int16 (与 phase_denorm.v 的 SAT 宏一致)
    MAX_VAL = int64(32767);
    MIN_VAL = int64(-32768);
    y = int16(min(max(int64(x), MIN_VAL), MAX_VAL));
end

function write_hex_int16(filepath, data)
    % 写 int16 向量到文件, 每行一个 4 字符大写十六进制值
    fid = fopen(filepath, 'w');
    if fid < 0
        error('无法创建输出文件: %s', filepath);
    end
    for i = 1:numel(data)
        v = int32(data(i));
        if v < 0, v = v + 65536; end
        fprintf(fid, '%04X\n', v);
    end
    fclose(fid);
end

%% ---- CORDIC Vectoring 定点仿真 (精确匹配 cordic.v MODE=0, ITERATIONS=14) ----
function [mag, angle_out] = cordic_vectoring_fixed(x_in, y_in)
    ITERATIONS = 14;
    atan_table = [12868, 7596, 4014, 2037, 1023, 512, 256, 128, 64, 32, 16, 8, 4, 2];

    scale = 2^14;
    x_int = round(x_in * scale);
    y_int = round(y_in * scale);
    x_int = max(min(x_int, 131071), -131072);
    y_int = max(min(y_int, 131071), -131072);

    PI_POS = 51472;
    PI_NEG = -51472;

    if x_in < 0
        x_int = -x_int;
        y_int = -y_int;
        if y_in >= 0
            z_int = PI_POS;
        else
            z_int = PI_NEG;
        end
    else
        z_int = 0;
    end

    for i = 0:ITERATIONS-1
        x_shift = floor(x_int / 2^i);
        y_shift = floor(y_int / 2^i);

        if y_int < 0
            x_new = x_int - y_shift;
            y_new = y_int + x_shift;
            z_new = z_int - atan_table(i+1);
        else
            x_new = x_int + y_shift;
            y_new = y_int - x_shift;
            z_new = z_int + atan_table(i+1);
        end

        x_int = max(min(x_new, 131071), -131072);
        y_int = max(min(y_new, 131071), -131072);
        z_int = max(min(z_new, 131071), -131072);
    end

    inv_k_int = 9949;
    mag_prod  = x_int * inv_k_int;
    mag_int   = floor(mag_prod / scale);
    mag_int   = mod(mag_int + 32768, 65536) - 32768;
    mag       = mag_int / scale;

    angle_out = z_int / scale;
end

%% ---- CORDIC Rotation 定点仿真 (精确匹配 cordic.v MODE=1, ITERATIONS=14) ----
function [x_out, y_out] = cordic_rotation_fixed(x_in, y_in, z_in)
    ITERATIONS = 14;
    atan_table = [12868, 7596, 4014, 2037, 1023, 512, 256, 128, 64, 32, 16, 8, 4, 2];

    scale = 2^14;
    x_int = round(x_in * scale);
    y_int = round(y_in * scale);
    z_int = round(z_in * scale);
    x_int = max(min(x_int, 131071), -131072);
    y_int = max(min(y_int, 131071), -131072);
    z_int = max(min(z_int, 131071), -131072);

    PI_POS  = 51472;
    PI_HALF = 25736;
    if z_int > PI_HALF
        x_new = -x_int;
        y_new = -y_int;
        z_int = z_int - PI_POS;
        x_int = max(min(x_new, 131071), -131072);
        y_int = max(min(y_new, 131071), -131072);
        z_int = max(min(z_int, 131071), -131072);
    elseif z_int < -PI_HALF
        x_new = -x_int;
        y_new = -y_int;
        z_int = z_int + PI_POS;
        x_int = max(min(x_new, 131071), -131072);
        y_int = max(min(y_new, 131071), -131072);
        z_int = max(min(z_int, 131071), -131072);
    end

    for i = 0:ITERATIONS-1
        x_shift = floor(x_int / 2^i);
        y_shift = floor(y_int / 2^i);

        if z_int < 0
            x_new = x_int + y_shift;
            y_new = y_int - x_shift;
            z_new = z_int + atan_table(i+1);
        else
            x_new = x_int - y_shift;
            y_new = y_int + x_shift;
            z_new = z_int - atan_table(i+1);
        end

        x_int = max(min(x_new, 131071), -131072);
        y_int = max(min(y_new, 131071), -131072);
        z_int = max(min(z_new, 131071), -131072);
    end

    inv_k_int = 9949;
    x_prod = x_int * inv_k_int;
    y_prod = y_int * inv_k_int;
    x_out_int = mod(floor(x_prod / scale) + 32768, 65536) - 32768;
    y_out_int = mod(floor(y_prod / scale) + 32768, 65536) - 32768;
    x_out = x_out_int / scale;
    y_out = y_out_int / scale;
end
