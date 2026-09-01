//=============================================================================
// Hidden_Layers — PRVTDNN 隐藏层串联包装模块
//
// 功能:
//   将 3 个隐藏层 (Hidden_Layer1~3_Dense) 串联为一个整体，
//   对外暴露统一的数据通路接口和各层权重/偏置加载接口。
//
// 数据流:
//   io_inAct (45×16-bit, 来自预处理)
//     → Hidden_Layer1_Dense (45→16, LeakyReLU)
//     → Hidden_Layer2_Dense (16→16, LeakyReLU)
//     → Hidden_Layer3_Dense (16→16, LeakyReLU)
//   io_outAct (16×16-bit, 发往 Output_Layer_Dense)
//
// 整体三段式架构:
//   preprocess.v  →  Hidden_Layers.v  →  Output_Layer_Dense.v
//    (45维特征输出)   (45→16→16→16)      (16→8, [I0..I3,Q0..Q3])
//
// 权重/偏置加载接口命名规则:
//   hl1_*  : 第一隐藏层 (45→16)
//   hl2_*  : 第二隐藏层 (16→16)
//   hl3_*  : 第三隐藏层 (16→16)
//=============================================================================

`include "prvtdnn_defines.vh"

module Hidden_Layers #(
    // --- 精度与位宽参数 ---
    parameter ACT_WIDTH    = 16,
    parameter WEIGHT_WIDTH = 16,
    parameter ACC_WIDTH    = 48, // 匹配 DSP48
    parameter BIAS_WIDTH   = 16,
    parameter FIXED_POINT  = 28, // 累加器定点位置 (Q1.14 × Q1.14 = Q?.28)
    parameter BIAS_FP      = 14, // 偏置定点位置 (Q1.14)
    parameter INT_NUM      = 1,  // 整数位宽

    // --- 激活函数参数 ---
    parameter LEAKY_SHIFT  = 3,  // LeakyReLU斜率 = 2^(-LEAKY_SHIFT) = 0.125

    // --- 各隐藏层维度参数 ---
    parameter HL1_IN_DIM   = `FC1_IN_DIM,
    parameter HL1_NEURONS  = `FC1_OUT_DIM,
    parameter HL2_NEURONS  = `FC2_OUT_DIM,
    parameter HL3_NEURONS  = `FC3_OUT_DIM
)(
    input  wire clock,
    input  wire reset,

    // ==========================================================
    // 数据通路接口
    // ==========================================================

    // 数据输入: 来自预处理模块, HL1_IN_DIM×16-bit 特征向量
    input  wire [HL1_IN_DIM*ACT_WIDTH-1:0]  io_inAct,

    // 数据输出: 发往输出层, HL3_NEURONS×16-bit 激活向量
    output wire signed [HL3_NEURONS*ACT_WIDTH-1:0]  io_outAct,

    // ==========================================================
    // 第一隐藏层权重/偏置加载接口 (45→16)
    // ==========================================================
    input  wire [HL1_NEURONS*ACC_WIDTH-1:0]        hl1_inWt_top,
    input  wire [HL1_NEURONS-1:0]                  hl1_inWtValid,
    input  wire signed [HL1_NEURONS*BIAS_WIDTH-1:0] hl1_inBias,
    input  wire                                     hl1_inBiasValid,

    // ==========================================================
    // 第二隐藏层权重/偏置加载接口 (16→16)
    // ==========================================================
    input  wire [HL2_NEURONS*ACC_WIDTH-1:0]        hl2_inWt_top,
    input  wire [HL2_NEURONS-1:0]                  hl2_inWtValid,
    input  wire signed [HL2_NEURONS*BIAS_WIDTH-1:0] hl2_inBias,
    input  wire                                     hl2_inBiasValid,

    // ==========================================================
    // 第三隐藏层权重/偏置加载接口 (16→16)
    // ==========================================================
    input  wire [HL3_NEURONS*ACC_WIDTH-1:0]        hl3_inWt_top,
    input  wire [HL3_NEURONS-1:0]                  hl3_inWtValid,
    input  wire signed [HL3_NEURONS*BIAS_WIDTH-1:0] hl3_inBias,
    input  wire                                     hl3_inBiasValid
);

    // ==========================================================
    // 层间数据连线
    // ==========================================================

    // HL1 输出 → HL2 输入: HL1_NEURONS×16-bit
    wire signed [HL1_NEURONS*ACT_WIDTH-1:0] hl1_to_hl2;

    // HL2 输出 → HL3 输入: HL2_NEURONS×16-bit
    wire signed [HL2_NEURONS*ACT_WIDTH-1:0] hl2_to_hl3;

    // ==========================================================
    // 第一隐藏层实例化: 45→16, Leaky ReLU
    // ==========================================================
    Hidden_Layer1_Dense #(
        .INPUT_DIM   (HL1_IN_DIM),
        .NEURONS     (HL1_NEURONS),
        .ACT_WIDTH   (ACT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .BIAS_WIDTH  (BIAS_WIDTH),
        .FIXED_POINT (FIXED_POINT),
        .BIAS_FP     (BIAS_FP),
        .INT_NUM     (INT_NUM),
        .ACT_FUNC    (2),            // LeakyReLU
        .LEAKY_SHIFT (LEAKY_SHIFT)
    ) hl1_inst (
        .clock          ( clock ),
        .reset          ( reset ),
        .io_inAct       ( io_inAct ),
        .io_inWt_top    ( hl1_inWt_top ),
        .io_inWtValid   ( hl1_inWtValid ),
        .io_inBias      ( hl1_inBias ),
        .io_inBiasValid ( hl1_inBiasValid ),
        .io_outAct      ( hl1_to_hl2 ),
        .io_outSigAddr  (  ),            // 悬空
        .io_inSigData   ( {HL1_NEURONS*ACT_WIDTH{1'b0}} )
    );

    // ==========================================================
    // 第二隐藏层实例化: 16→16, Leaky ReLU
    // ==========================================================
    Hidden_Layer2_Dense #(
        .INPUT_DIM   (HL1_NEURONS),  // 输入维度 = 上一层神经元数
        .NEURONS     (HL2_NEURONS),
        .ACT_WIDTH   (ACT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .BIAS_WIDTH  (BIAS_WIDTH),
        .FIXED_POINT (FIXED_POINT),
        .BIAS_FP     (BIAS_FP),
        .INT_NUM     (INT_NUM),
        .ACT_FUNC    (2),            // LeakyReLU
        .LEAKY_SHIFT (LEAKY_SHIFT)
    ) hl2_inst (
        .clock          ( clock ),
        .reset          ( reset ),
        .io_inAct       ( hl1_to_hl2 ),
        .io_inWt_top    ( hl2_inWt_top ),
        .io_inWtValid   ( hl2_inWtValid ),
        .io_inBias      ( hl2_inBias ),
        .io_inBiasValid ( hl2_inBiasValid ),
        .io_outAct      ( hl2_to_hl3 ),
        .io_outSigAddr  (  ),            // 悬空
        .io_inSigData   ( {HL2_NEURONS*ACT_WIDTH{1'b0}} )
    );

    // ==========================================================
    // 第三隐藏层实例化: 16→16, Leaky ReLU
    // ==========================================================
    Hidden_Layer3_Dense #(
        .INPUT_DIM   (HL2_NEURONS),  // 输入维度 = 上一层神经元数
        .NEURONS     (HL3_NEURONS),
        .ACT_WIDTH   (ACT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .BIAS_WIDTH  (BIAS_WIDTH),
        .FIXED_POINT (FIXED_POINT),
        .BIAS_FP     (BIAS_FP),
        .INT_NUM     (INT_NUM),
        .ACT_FUNC    (2),            // LeakyReLU
        .LEAKY_SHIFT (LEAKY_SHIFT)
    ) hl3_inst (
        .clock          ( clock ),
        .reset          ( reset ),
        .io_inAct       ( hl2_to_hl3 ),
        .io_inWt_top    ( hl3_inWt_top ),
        .io_inWtValid   ( hl3_inWtValid ),
        .io_inBias      ( hl3_inBias ),
        .io_inBiasValid ( hl3_inBiasValid ),
        .io_outAct      ( io_outAct ),
        .io_outSigAddr  (  ),            // 悬空
        .io_inSigData   ( {HL3_NEURONS*ACT_WIDTH{1'b0}} )
    );

endmodule
