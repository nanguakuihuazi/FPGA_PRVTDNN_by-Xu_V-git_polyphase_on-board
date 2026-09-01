`timescale 1ns / 1ps
`include "prvtdnn_defines.vh"

//=============================================================================
// 相位反归一化模块 (Phase De-normalization)
// 功能: 将多输出神经网络的结果 [I_0..I_{N-1}, Q_0..Q_{N-1}] 恢复到原始相位空间
// 算法: (out_I + j*out_Q) = (rk_re - j*rk_im) * (Y_I + j*Y_Q)
//       out_I = rk_re * Y_I + rk_im * Y_Q
//       out_Q = rk_re * Y_Q - rk_im * Y_I
//
// 延迟: 2 cycles (乘法 1 拍 + 加减与饱和截断 1 拍)
// 吞吐: 1 sample/cycle
// 输出 out_i_flat总共128位，其中[15:0]位为第0项y(n)，[31:16]位为第1项y(n+1)，以此类推；out_q_flat同理
//=============================================================================
module phase_denorm #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 14,
    parameter INT_BITS   = 1,
    parameter N_POLY     = `N_POLY
)(
    input  wire                                clk,
    input  wire                                rst_n,
    input  wire signed [N_POLY*DATA_WIDTH-1:0] nn_out_i_flat,
    input  wire signed [N_POLY*DATA_WIDTH-1:0] nn_out_q_flat,
    input  wire                                nn_valid,
    input  wire signed [N_POLY*DATA_WIDTH-1:0] rk_re_flat,
    input  wire signed [N_POLY*DATA_WIDTH-1:0] rk_im_flat,

    output reg signed [N_POLY*DATA_WIDTH-1:0]  out_i_flat,
    output reg signed [N_POLY*DATA_WIDTH-1:0]  out_q_flat,
    output reg                                 out_valid
);

    localparam PROD_FRAC    = 2 * FRAC_BITS;
    localparam PROD_WIDTH   = 2 * DATA_WIDTH;
    localparam SIGN_BIT_POS = PROD_FRAC + INT_BITS;

    reg signed [PROD_WIDTH-1:0] mult_i_re [0:N_POLY-1];
    reg signed [PROD_WIDTH-1:0] mult_i_im [0:N_POLY-1];
    reg signed [PROD_WIDTH-1:0] mult_q_re [0:N_POLY-1];
    reg signed [PROD_WIDTH-1:0] mult_q_im [0:N_POLY-1];
    reg                         valid_d1;

    wire signed [DATA_WIDTH-1:0] nn_i [0:N_POLY-1];
    wire signed [DATA_WIDTH-1:0] nn_q [0:N_POLY-1];
    wire signed [DATA_WIDTH-1:0] rk_r [0:N_POLY-1];
    wire signed [DATA_WIDTH-1:0] rk_i [0:N_POLY-1];
    wire signed [PROD_WIDTH:0]   sum_i [0:N_POLY-1];
    wire signed [PROD_WIDTH:0]   sum_q [0:N_POLY-1];

    genvar gi;
    generate
        for (gi = 0; gi < N_POLY; gi = gi + 1) begin : gen_unpack
            assign nn_i[gi] = nn_out_i_flat[gi*DATA_WIDTH +: DATA_WIDTH];
            assign nn_q[gi] = nn_out_q_flat[gi*DATA_WIDTH +: DATA_WIDTH];
            assign rk_r[gi] = rk_re_flat[gi*DATA_WIDTH +: DATA_WIDTH];
            assign rk_i[gi] = rk_im_flat[gi*DATA_WIDTH +: DATA_WIDTH];

            assign sum_i[gi] = {mult_i_re[gi][PROD_WIDTH-1], mult_i_re[gi]} +
                               {mult_q_im[gi][PROD_WIDTH-1], mult_q_im[gi]};
            assign sum_q[gi] = {mult_q_re[gi][PROD_WIDTH-1], mult_q_re[gi]} -
                               {mult_i_im[gi][PROD_WIDTH-1], mult_i_im[gi]};
        end
    endgenerate

    function [DATA_WIDTH-1:0] sat_trunc;
        input signed [PROD_WIDTH:0] value;
        begin
            if ((~value[PROD_WIDTH]) && (|value[PROD_WIDTH-1:SIGN_BIT_POS]))
                sat_trunc = {1'b0, {(DATA_WIDTH-1){1'b1}}};
            else if (value[PROD_WIDTH] && (~&value[PROD_WIDTH-1:SIGN_BIT_POS]))
                sat_trunc = {1'b1, {(DATA_WIDTH-1){1'b0}}};
            else
                sat_trunc = value[SIGN_BIT_POS -: DATA_WIDTH];
        end
    endfunction

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1  <= 1'b0;
            out_i_flat <= 0;
            out_q_flat <= 0;
            out_valid <= 1'b0;
            for (i = 0; i < N_POLY; i = i + 1) begin
                mult_i_re[i] <= 0;
                mult_i_im[i] <= 0;
                mult_q_re[i] <= 0;
                mult_q_im[i] <= 0;
            end
        end else begin
            valid_d1 <= nn_valid;
            out_valid <= valid_d1;

            for (i = 0; i < N_POLY; i = i + 1) begin
                mult_i_re[i] <= nn_i[i] * rk_r[i];
                mult_i_im[i] <= nn_i[i] * rk_i[i];
                mult_q_re[i] <= nn_q[i] * rk_r[i];
                mult_q_im[i] <= nn_q[i] * rk_i[i];
            end

            if (valid_d1) begin
                for (i = 0; i < N_POLY; i = i + 1) begin
                    out_i_flat[i*DATA_WIDTH +: DATA_WIDTH] <= sat_trunc(sum_i[i]);
                    out_q_flat[i*DATA_WIDTH +: DATA_WIDTH] <= sat_trunc(sum_q[i]);
                end
            end
        end
    end

endmodule