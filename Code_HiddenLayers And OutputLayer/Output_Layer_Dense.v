`include "prvtdnn_defines.vh"

//Output_Layer_Dense 时钟周期分析
//加载参数（权重）：INPUT_DIM 个时钟周期。权重从脉动阵列顶部顺序进入（io_inWt_top），
//                对于当前网络(16→8)需要从第 0 行流到第 15 行，valid_delay_reg 在每行产生 1 个周期的延迟。
//加载偏置：1 个时钟周期。偏置通过 io_inBias 和 io_inBiasValid 直接加载到 BiasActivation 模块的寄存器中。

module Output_Layer_Dense #(
    // --- 核心网络维度参数 ---
    parameter INPUT_DIM    = `FC5_IN_DIM,
    parameter NEURONS      = `FC5_OUT_DIM,

    // --- 精度与位宽参数 ---
    parameter ACT_WIDTH    = 16,
    parameter WEIGHT_WIDTH = 16,
    parameter ACC_WIDTH    = 48, // 匹配 DSP48
    parameter BIAS_WIDTH   = 16,
    parameter FIXED_POINT  = 28, // 累加器定点位置 (Q1.14 × Q1.14 = Q?.28)
    parameter BIAS_FP      = 14, // 偏置定点位置 (Q1.14)
    parameter INT_NUM      = 1,  // 整数位宽

    // --- 激活函数选择 ---
    parameter ACT_FUNC     = 0,  // 0: Linear — 输出层无激活函数
    parameter LEAKY_SHIFT  = 3   // 此层不使用
)(
    input  wire clock,
    input  wire reset,

    // ==========================================
    // 1. 数据输入接口 (来自第三隐藏层输出)
    // ==========================================
    // INPUT_DIM 个通道的激活值输入流，平铺为一维总线 (INPUT_DIM*ACT_WIDTH 位)
    input  wire [INPUT_DIM*ACT_WIDTH-1:0]  io_inAct,

    // ==========================================
    // 2. 权重配置接口 (顶部灌入)
    // ==========================================
    // NEURONS 根复用总线，负责灌入输出层的权重，平铺为一维总线
    input  wire [NEURONS*ACC_WIDTH-1:0]  io_inWt_top,
    input  wire [NEURONS-1:0]            io_inWtValid,

    // ==========================================
    // 3. 偏置配置接口
    // ==========================================
    // NEURONS 个神经元的偏置，拼接成 NEURONS*16 bit 的总线
    input  wire signed [NEURONS*BIAS_WIDTH-1:0] io_inBias,
    input  wire                                 io_inBiasValid,

    // ==========================================
    // 4. 最终网络输出 [I0..I(N-1), Q0..Q(N-1)] (相位归一化空间)
    // ==========================================
    // NEURONS 个神经元的线性输出，拼接成 NEURONS*16 bit 的总线
    output wire signed [NEURONS*ACT_WIDTH-1:0]  io_outAct,

    // ==========================================
    // 5. Sigmoid 外部 LUT 接口 (如果不用 Sigmoid 可悬空)
    // ==========================================
    output wire        [NEURONS*ACT_WIDTH-1:0]  io_outSigAddr,
    input  wire signed [NEURONS*ACT_WIDTH-1:0]  io_inSigData
);

    // ==========================================
    // 内部连线
    // ==========================================
    // 脉动阵列输出：2 个 48 位部分和
    wire [NEURONS*ACC_WIDTH-1:0] array_out_sum;

    // 偏置激活原始输出（尚未对齐）
    wire [NEURONS*ACT_WIDTH-1:0] raw_outAct;

    // ==========================================
    // 模块实例化 1: 4x2 的脉动阵列 (矩阵乘法核)
    // ==========================================
    SystolicArray #(
        .ROWS        (INPUT_DIM),
        .COLS        (NEURONS),
        .ACT_WIDTH   (ACT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH)
    ) systolic_array_inst (
        .clock          ( clock ),
        .reset          ( reset ),
        .io_inAct_left  ( io_inAct ),     // 前层输出已为错位格式，无需重新错位
        .io_inWtPS_top  ( io_inWt_top ),
        .io_inwtValid   ( io_inWtValid ),
        .io_outWtPS_bot ( array_out_sum )
    );

    // ==========================================
    // 模块实例化 2: 2通道的偏置与激活模块 (线性输出，无激活函数)
    // ==========================================
    BiasActivation #(
        .ACC_WIDTH   (ACC_WIDTH),
        .ACT_WIDTH   (ACT_WIDTH),
        .BIAS_WIDTH  (BIAS_WIDTH),
        .FIXED_POINT (FIXED_POINT),
        .BIAS_FP     (BIAS_FP),
        .MESH_COLUMNS(NEURONS),
        .INT_NUM     (INT_NUM),
        .ACT_FUNC    (ACT_FUNC),    // Linear: 直通，无激活函数
        .LEAKY_SHIFT (LEAKY_SHIFT)
    ) bias_act_inst (
        .clk            ( clock ),
        .reset          ( reset ),
        .io_inSum       ( array_out_sum ),
        .io_inBias      ( io_inBias ),
        .io_inBiasValid ( io_inBiasValid ),
        .io_outAct      ( raw_outAct )      // 输出到中间线，待重对齐
    );

    // ==========================================
    // 模块实例化 3: 输出重对齐 (Output De-skewing)
    // ==========================================
    ShiftRegisterSkew #(
        .NUM_CH    (NEURONS),
        .DATA_WIDTH(ACT_WIDTH),
        .DIRECTION (1)           // 1 = 降序延迟 (输出重对齐)
    ) output_deskew_inst (
        .clock  ( clock ),
        .reset  ( reset ),
        .io_in  ( raw_outAct ),
        .io_out ( io_outAct )
    );

    // 当前BiasActivation未使用Sigmoid LUT接口，置零避免悬空
    assign io_outSigAddr = {NEURONS*ACT_WIDTH{1'b0}};

endmodule
