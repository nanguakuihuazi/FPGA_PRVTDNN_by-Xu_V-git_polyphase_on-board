//=============================================================================
// PRVTDNN Polyphase 预处理模块 — 全流水线架构
// 功能: 对应 MATLAB preprocess_data_pn_rvtdnn_multiout()
//
// 数据流:
//   1. 维护 x(n) ... x(n-M_poly) 的复数延迟线
//   2. 对每个延迟样本独立做 VECTORING, 得到 |x(n-k)| 和 theta_k
//   3. 将单位向量 (1,0) 旋转 theta_k, 生成 x(n-k)/|x(n-k)| 的 I/Q 特征
//   4. 对前 N_poly 个分支额外生成 rk_k = exp(-j*theta_k)
//   5. 组装特征: [Re(unit), Im(unit), R, R^2]
//
// 输出特征排列 (当前 36 维):
//   feat[ 0.. 8]: Re(x(n-k)/|x(n-k)|), k=0..8
//   feat[ 9..17]: Im(x(n-k)/|x(n-k)|), k=0..8
//   feat[18..26]: |x(n-k)|,            k=0..8
//   feat[27..35]: |x(n-k)|^2,          k=0..8
//
// rk 输出排列:
//   rk_re_flat[j] = cos(theta_j)
//   rk_im_flat[j] = -sin(theta_j)
//   j = 0..N_poly-1
//
// 延迟: 33 cycles = 15 (vectoring) + 15 (rotation) + 2 (R², R³ 对齐) + 1 (输出寄存器)
// 吞吐: 1 sample/cycle
//=============================================================================
`timescale 1ns / 1ps
`include "prvtdnn_defines.vh"

module preprocess #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 14,
    parameter M          = `M,
    parameter N_POLY     = `N_POLY,
    parameter TAP_NUM    = M + N_POLY,
    parameter FEAT_DIM   = 5 * TAP_NUM,
    parameter RK_DIM     = N_POLY
)(
    input  wire                              clk,
    input  wire                              rst_n,
    // FPGA 每周期并行接收 N_POLY=4 个采样点
    // x_re_flat[k*DATA_WIDTH +: DATA_WIDTH] = x_re(n-k), k=0..N_POLY-1
    // x_im_flat[k*DATA_WIDTH +: DATA_WIDTH] = x_im(n-k), k=0..N_POLY-1
    input  wire [N_POLY*DATA_WIDTH-1:0]      x_re_flat,
    input  wire [N_POLY*DATA_WIDTH-1:0]      x_im_flat,
    output reg signed [FEAT_DIM*DATA_WIDTH-1:0] feat_flat,
    output reg                               out_valid,
    output reg signed [RK_DIM*DATA_WIDTH-1:0]   rk_re_flat,
    output reg signed [RK_DIM*DATA_WIDTH-1:0]   rk_im_flat
);

    localparam integer CORDIC_LAT = FRAC_BITS + 1;
    localparam signed [DATA_WIDTH-1:0] ONE_Q14 = 16'sd16384;

    genvar gi, gk;
    integer i;

    // 延迟寄存器: FPGA每周期并行接收 N_POLY=4 个采样点
    // delay1: 上一FPGA周期的 4 个采样点 → 对应 x(n-4)..x(n-7)
    // delay2_re_0/delay2_im_0: 两个周期前的第 0 个采样点 → 对应 x(n-8)
    reg signed [DATA_WIDTH-1:0] delay1_re [0:N_POLY-1];
    reg signed [DATA_WIDTH-1:0] delay1_im [0:N_POLY-1];
    reg signed [DATA_WIDTH-1:0] delay2_re_0;
    reg signed [DATA_WIDTH-1:0] delay2_im_0;
    reg signed [DATA_WIDTH-1:0] delay3_re_0;
    reg signed [DATA_WIDTH-1:0] delay3_im_0;

    integer d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (d = 0; d < N_POLY; d = d + 1) begin
                delay1_re[d] <= 0;
                delay1_im[d] <= 0;
            end
            delay2_re_0 <= 0;
            delay2_im_0 <= 0;
            delay3_re_0 <= 0;
            delay3_im_0 <= 0;
        end else begin
            // FPGA每周期均有有效数据, 无条件更新
            for (d = 0; d < N_POLY; d = d + 1) begin
                delay1_re[d] <= x_re_flat[d*DATA_WIDTH +: DATA_WIDTH];
                delay1_im[d] <= x_im_flat[d*DATA_WIDTH +: DATA_WIDTH];
            end
            delay2_re_0 <= delay1_re[0];
            delay2_im_0 <= delay1_im[0];
            // Align oldest tap with golden sequence (one extra register stage).
            delay3_re_0 <= delay2_re_0;
            delay3_im_0 <= delay2_im_0;
        end
    end

    wire signed [DATA_WIDTH-1:0] tap_re [0:TAP_NUM-1];
    wire signed [DATA_WIDTH-1:0] tap_im [0:TAP_NUM-1];
    // tap[0..N_POLY-1]: 当前FPGA周期的 4 个采样点 x(n)..x(n-3)
    generate
        for (gi = 0; gi < N_POLY; gi = gi + 1) begin : gen_taps_curr
            assign tap_re[gi] = x_re_flat[gi*DATA_WIDTH +: DATA_WIDTH];
            assign tap_im[gi] = x_im_flat[gi*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate
    // tap[N_POLY..2*N_POLY-1]: 上一FPGA周期的 4 个采样点 x(n-4)..x(n-7)
    generate
        for (gi = 0; gi < N_POLY; gi = gi + 1) begin : gen_taps_prev
            assign tap_re[N_POLY + gi] = delay1_re[gi];
            assign tap_im[N_POLY + gi] = delay1_im[gi];
        end
    endgenerate
    // tap[TAP_NUM-1] = tap[8]: 最老抽头增加1拍，与黄金特征窗口对齐。
    assign tap_re[TAP_NUM-1] = delay3_re_0;
    assign tap_im[TAP_NUM-1] = delay3_im_0;

    wire                         stgA_valid [0:TAP_NUM-1];
    wire signed [DATA_WIDTH-1:0] stgA_R     [0:TAP_NUM-1];
    wire signed [17:0]           stgA_theta [0:TAP_NUM-1];

    generate
        for (gi = 0; gi < TAP_NUM; gi = gi + 1) begin : gen_stgA_vec
            cordic #(
                .DATA_WIDTH (DATA_WIDTH),
                .ITERATIONS (FRAC_BITS),
                .Z_WIDTH    (18),
                .MODE       (0)
            ) u_stgA_vec (
                .clk       (clk),
                .rst_n     (rst_n),
                .in_valid  (1'b1),
                .in_x      (tap_re[gi]),
                .in_y      (tap_im[gi]),
                .in_z      (18'sd0),
                .out_valid (stgA_valid[gi]),
                .out_x     (stgA_R[gi]),
                .out_y     (),
                .out_z     (stgA_theta[gi])
            );
        end
    endgenerate

    wire                         stgB_unit_valid [0:TAP_NUM-1];
    wire signed [DATA_WIDTH-1:0] stgB_unit_re    [0:TAP_NUM-1];
    wire signed [DATA_WIDTH-1:0] stgB_unit_im    [0:TAP_NUM-1];

    generate
        for (gi = 0; gi < TAP_NUM; gi = gi + 1) begin : gen_stgB_unit
            cordic #(
                .DATA_WIDTH (DATA_WIDTH),
                .ITERATIONS (FRAC_BITS),
                .Z_WIDTH    (18),
                .MODE       (1)
            ) u_stgB_unit (
                .clk       (clk),
                .rst_n     (rst_n),
                .in_valid  (stgA_valid[gi]),
                .in_x      (ONE_Q14),
                .in_y      (16'sd0),
                .in_z      (stgA_theta[gi]),
                .out_valid (stgB_unit_valid[gi]),
                .out_x     (stgB_unit_re[gi]),
                .out_y     (stgB_unit_im[gi]),
                .out_z     ()
            );
        end
    endgenerate

    wire                         stgB_rk_valid [0:RK_DIM-1];
    wire signed [DATA_WIDTH-1:0] stgB_rk_re    [0:RK_DIM-1];
    wire signed [DATA_WIDTH-1:0] stgB_rk_im    [0:RK_DIM-1];

    generate
        for (gi = 0; gi < RK_DIM; gi = gi + 1) begin : gen_stgB_rk
            cordic #(
                .DATA_WIDTH (DATA_WIDTH),
                .ITERATIONS (FRAC_BITS),
                .Z_WIDTH    (18),
                .MODE       (1)
            ) u_stgB_rk (
                .clk       (clk),
                .rst_n     (rst_n),
                .in_valid  (stgA_valid[gi]),
                .in_x      (ONE_Q14),
                .in_y      (16'sd0),
                .in_z      (-stgA_theta[gi]),
                .out_valid (stgB_rk_valid[gi]),
                .out_x     (stgB_rk_re[gi]),
                .out_y     (stgB_rk_im[gi]),
                .out_z     ()
            );
        end
    endgenerate

    wire stgB_out_valid = stgB_unit_valid[0];

    reg signed [DATA_WIDTH-1:0] sr_R_align [0:TAP_NUM-1][0:CORDIC_LAT-1];

    generate
        for (gi = 0; gi < TAP_NUM; gi = gi + 1) begin : gen_sr_R_align
            always @(posedge clk) begin
                sr_R_align[gi][0] <= stgA_R[gi];
            end

            for (gk = 1; gk < CORDIC_LAT; gk = gk + 1) begin : gen_sr_R_align_stage
                always @(posedge clk) begin
                    sr_R_align[gi][gk] <= sr_R_align[gi][gk-1];
                end
            end
        end
    endgenerate

    wire signed [DATA_WIDTH-1:0] stgB_R_aligned [0:TAP_NUM-1];
    generate
        for (gi = 0; gi < TAP_NUM; gi = gi + 1) begin : gen_sr_R_align_out
            assign stgB_R_aligned[gi] = sr_R_align[gi][CORDIC_LAT-1];
        end
    endgenerate

    reg signed [DATA_WIDTH-1:0] pipe_R_d0  [0:TAP_NUM-1];
    reg signed [DATA_WIDTH-1:0] pipe_R2_d0 [0:TAP_NUM-1];
    reg signed [DATA_WIDTH-1:0] pipe_R_d1  [0:TAP_NUM-1];
    reg signed [DATA_WIDTH-1:0] pipe_R2_d1 [0:TAP_NUM-1];
    reg signed [DATA_WIDTH-1:0] pipe_R3_d1 [0:TAP_NUM-1];

    generate
        for (gi = 0; gi < TAP_NUM; gi = gi + 1) begin : gen_stageC_power
            always @(posedge clk) begin : blk_stageC_power
                reg signed [2*DATA_WIDTH-1:0] r_sq;
                reg signed [2*DATA_WIDTH-1:0] r_cub;

                pipe_R_d0[gi] <= stgB_R_aligned[gi];
                r_sq = stgB_R_aligned[gi] * stgB_R_aligned[gi];
                pipe_R2_d0[gi] <= r_sq[FRAC_BITS + DATA_WIDTH - 1 : FRAC_BITS];
                pipe_R_d1[gi] <= pipe_R_d0[gi];
                pipe_R2_d1[gi] <= pipe_R2_d0[gi];
                // R³ = R × R²，利用 d0 级已就绪的 R 和 R² 计算，结果注册到 d1 与 R、R² 对齐
                r_cub = pipe_R_d0[gi] * pipe_R2_d0[gi];
                pipe_R3_d1[gi] <= r_cub[FRAC_BITS + DATA_WIDTH - 1 : FRAC_BITS];
            end
        end
    endgenerate

    reg signed [DATA_WIDTH-1:0] unit_re_d [0:TAP_NUM-1][0:1];
    reg signed [DATA_WIDTH-1:0] unit_im_d [0:TAP_NUM-1][0:1];
    reg signed [DATA_WIDTH-1:0] rk_re_d   [0:RK_DIM-1][0:1];
    reg signed [DATA_WIDTH-1:0] rk_im_d   [0:RK_DIM-1][0:1];

    generate
        for (gi = 0; gi < TAP_NUM; gi = gi + 1) begin : gen_stageC_unit_delay
            always @(posedge clk) begin
                unit_re_d[gi][0] <= stgB_unit_re[gi];
                unit_im_d[gi][0] <= stgB_unit_im[gi];
                unit_re_d[gi][1] <= unit_re_d[gi][0];
                unit_im_d[gi][1] <= unit_im_d[gi][0];
            end
        end

        for (gi = 0; gi < RK_DIM; gi = gi + 1) begin : gen_stageC_rk_delay
            always @(posedge clk) begin
                rk_re_d[gi][0] <= stgB_rk_re[gi];
                rk_im_d[gi][0] <= stgB_rk_im[gi];
                rk_re_d[gi][1] <= rk_re_d[gi][0];
                rk_im_d[gi][1] <= rk_im_d[gi][0];
            end
        end
    endgenerate

    reg stgC_valid_d0, stgC_valid_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stgC_valid_d0 <= 1'b0;
            stgC_valid_d1 <= 1'b0;
        end else begin
            stgC_valid_d0 <= stgB_out_valid;
            stgC_valid_d1 <= stgC_valid_d0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_flat  <= 0;
            rk_re_flat <= 0;
            rk_im_flat <= 0;
            out_valid  <= 1'b0;
        end else begin
            out_valid <= stgC_valid_d1;
            if (stgC_valid_d1) begin
                for (i = 0; i < TAP_NUM; i = i + 1) begin
                    // R=0 时 unit 方向未定义, 门控为 0, 避免 CORDIC 对 (0,0) 产生乱数角度
                    feat_flat[i*DATA_WIDTH +: DATA_WIDTH] <= (pipe_R_d1[i] == 0) ? {DATA_WIDTH{1'b0}} : unit_re_d[i][1];
                    feat_flat[(TAP_NUM + i)*DATA_WIDTH +: DATA_WIDTH] <= (pipe_R_d1[i] == 0) ? {DATA_WIDTH{1'b0}} : unit_im_d[i][1];
                    feat_flat[(2*TAP_NUM + i)*DATA_WIDTH +: DATA_WIDTH] <= pipe_R_d1[i];
                    feat_flat[(3*TAP_NUM + i)*DATA_WIDTH +: DATA_WIDTH] <= pipe_R2_d1[i];
                    feat_flat[(4*TAP_NUM + i)*DATA_WIDTH +: DATA_WIDTH] <= pipe_R3_d1[i];
                end

                for (i = 0; i < RK_DIM; i = i + 1) begin
                    rk_re_flat[i*DATA_WIDTH +: DATA_WIDTH] <= (pipe_R_d1[i] == 0) ? {DATA_WIDTH{1'b0}} : rk_re_d[i][1];
                    rk_im_flat[i*DATA_WIDTH +: DATA_WIDTH] <= (pipe_R_d1[i] == 0) ? {DATA_WIDTH{1'b0}} : rk_im_d[i][1];
                end
            end
        end
    end

endmodule
