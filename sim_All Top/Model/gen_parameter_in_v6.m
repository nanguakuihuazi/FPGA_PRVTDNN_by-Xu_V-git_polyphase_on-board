%% gen_parameter_in_v6.m
% Build parameter_in.mem for top-level coeff loading.
% Source files are the per-layer Q1.14 hex files under Model/unpadded.
% Packing rule matches prvtdnn_top.v decoding:
%   k even  -> low  16 bits of one 32-bit word
%   k odd   -> high 16 bits of one 32-bit word
% Current RTL dims: 45 -> 16 -> 16 -> 16 -> 8

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
src_dir = fullfile(script_dir, 'unpadded');
if ~exist(src_dir, 'dir')
    src_dir = script_dir;
end

fprintf('Source dir: %s\n', src_dir);

% RTL target dimensions (must match prvtdnn_defines.vh)
rtl_dims = [45, 16;   % layer1: in, out
            16, 16;   % layer2
            16, 16;   % layer3
            16,  8];  % layer4

% Input file names
w_file = {
    'layer1_weights_poly_multiout.hex', ...
    'layer2_weights_poly_multiout.hex', ...
    'layer3_weights_poly_multiout.hex', ...
    'layer4_weights_poly_multiout.hex'};

b_file = {
    'layer1_bias_poly_multiout.hex', ...
    'layer2_bias_poly_multiout.hex', ...
    'layer3_bias_poly_multiout.hex', ...
    'layer4_bias_poly_multiout.hex'};

num_layers = 4;
W_actual = cell(num_layers, 1);
b_actual = cell(num_layers, 1);
actual_dims = zeros(num_layers, 2); % [in, out]

for li = 1:num_layers
    w_path = fullfile(src_dir, w_file{li});
    b_path = fullfile(src_dir, b_file{li});

    w_flat = read_hex_int16_file(w_path);
    b_vec = read_hex_int16_file(b_path);

    out_dim = numel(b_vec);
    if out_dim == 0
        error('Empty bias file: %s', b_path);
    end
    if mod(numel(w_flat), out_dim) ~= 0
        error('Weight count not divisible by bias count at layer %d: %d / %d', ...
            li, numel(w_flat), out_dim);
    end

    in_dim = numel(w_flat) / out_dim;

    W = zeros(out_dim, in_dim, 'int16');
    for n = 1:out_dim
        idx0 = (n - 1) * in_dim + 1;
        idx1 = n * in_dim;
        W(n, :) = w_flat(idx0:idx1).';
    end

    W_actual{li} = W;
    b_actual{li} = b_vec(:);
    actual_dims(li, :) = [in_dim, out_dim];

    fprintf('Layer %d actual: in=%d, out=%d, w=%d, b=%d\n', ...
        li, in_dim, out_dim, numel(w_flat), numel(b_vec));

    if any(actual_dims(li, :) ~= rtl_dims(li, :))
        error(['Layer %d dimension mismatch. Model export is [%d -> %d], ', ...
               'but RTL expects [%d -> %d]. Please regenerate per-layer hex with new topology.'], ...
               li, in_dim, out_dim, rtl_dims(li, 1), rtl_dims(li, 2));
    end
end

% Build parameter vector in neuron-major order, exactly matching RTL layout.
params_i16 = int16([]);
for li = 1:num_layers
    in_dim = rtl_dims(li, 1);
    out_dim = rtl_dims(li, 2);

    W = W_actual{li};
    b = b_actual{li};

    % Flatten in neuron-major order: neuron0 all inputs, neuron1 all inputs, ...
    for n = 1:out_dim
        params_i16 = [params_i16; W(n, 1:in_dim).']; %#ok<AGROW>
    end
    params_i16 = [params_i16; b(1:out_dim)]; %#ok<AGROW>
end

total_params = numel(params_i16);
if mod(total_params, 2) ~= 0
    error('Total parameter count must be even, got %d.', total_params);
end

expected_params = sum(rtl_dims(:, 1) .* rtl_dims(:, 2) + rtl_dims(:, 2));
if total_params ~= expected_params
    error('Total params mismatch: got %d, expected %d from RTL dims.', ...
        total_params, expected_params);
end

out_path = fullfile(script_dir, 'parameter_in.mem');
write_parameter_mem_32(out_path, params_i16);

% Strict output size check for current dims: 1416 params -> 708 32-bit words
mem_lines = count_file_lines(out_path);
expected_words = expected_params / 2;
if mem_lines ~= expected_words
    error('parameter_in.mem line count mismatch: got %d, expected %d.', ...
        mem_lines, expected_words);
end

fprintf('Generated: %s\n', out_path);
fprintf('Total 16-bit params: %d\n', total_params);
fprintf('Total 32-bit words : %d\n', total_params / 2);
fprintf('parameter_in.mem lines: %d (PASS)\n', mem_lines);

% Optional: also export native parameter stream
% native_path = fullfile(script_dir, 'parameter_in_unpadded.mem');
% write_parameter_mem_32(native_path, build_native_params(W_actual, b_actual));


function vals = read_hex_int16_file(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('Cannot open file: %s', path);
    end
    c = onCleanup(@() fclose(fid));

    vals = int16([]);
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if ~ischar(line)
            continue;
        end
        if isempty(line)
            continue;
        end
        u16 = uint16(hex2dec(line));
        vals(end+1, 1) = typecast(u16, 'int16'); %#ok<AGROW>
    end
end


function write_parameter_mem_32(path, params_i16)
    fid = fopen(path, 'w');
    if fid < 0
        error('Cannot create output file: %s', path);
    end
    c = onCleanup(@() fclose(fid));

    for k = 1:2:numel(params_i16)
        lo_u16 = typecast(int16(params_i16(k)), 'uint16');
        hi_u16 = typecast(int16(params_i16(k + 1)), 'uint16');
        fprintf(fid, '%04X%04X\n', hi_u16, lo_u16);
    end
end


function n = count_file_lines(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('Cannot open file for line count: %s', path);
    end
    c = onCleanup(@() fclose(fid));

    n = 0;
    while ~feof(fid)
        t = fgetl(fid);
        if ischar(t)
            n = n + 1;
        end
    end
end


function params = build_native_params(W_actual, b_actual)
    params = int16([]);
    for li = 1:numel(W_actual)
        W = W_actual{li};
        b = b_actual{li};
        for n = 1:size(W, 1)
            params = [params; W(n, :).']; %#ok<AGROW>
        end
        params = [params; b(:)]; %#ok<AGROW>
    end
end
