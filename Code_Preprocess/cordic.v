//=============================================================================
// CORDIC 处理器 (全流水线, 14级迭代)
//
// 支持两种模式:
//   MODE=0 (VECTORING): 输入(x,y) → 输出(magnitude, angle)
//     将向量(x,y)旋转到 y=0, 得到 x_final = K*sqrt(x²+y²), z_final = atan2(y,x)，也就是记录了最初的向量角度信息 
//   MODE=1 (ROTATION):  输入(x,y,angle) → 输出(x_rot, y_rot)
//     将向量(x,y)旋转给定角度, x_out = K*(x*cos(z)-y*sin(z))
//
// CORDIC 增益 K ≈ 1.6468, 补偿在最后一级乘以 1/K ≈ 0.6073
//
// 输入/输出: Q1.14 (16-bit signed), 内部用 18-bit 保护精度
// 角度格式: Q2.14 (16-bit signed), 范围 [-π, π) → [-3.14, 3.14)
//
// 延迟: ITERATIONS + 1 = 15 个时钟周期 (pipeline)
// 吞吐: 1 sample/cycle
//=============================================================================

`timescale 1ns / 1ps
module cordic #(
    parameter DATA_WIDTH  = 16,
    parameter ITERATIONS  = 14,
    parameter Z_WIDTH     = 18,     // 【新增】相位专用位宽，默认为 18
    parameter MODE        = 0       // 0=VECTORING, 1=ROTATION
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          in_valid,  //输入握手，每拍拉高表示有新数据
    input  wire signed [DATA_WIDTH-1:0]  in_x,      //复数向量的实部，I 分量
    input  wire signed [DATA_WIDTH-1:0]  in_y,      //复数向量的虚部，Q 分量
    input  wire signed [Z_WIDTH-1:0]     in_z,       // 仅 ROTATION 模式使用，目标旋转角度（Q2.14）
    
    output wire                          out_valid,
    output wire signed [DATA_WIDTH-1:0]  out_x,      // VECTORING: magnitude 
    output wire signed [DATA_WIDTH-1:0]  out_y,      // VECTORING: ≈0; ROTATION: y_rotated
    output wire signed [Z_WIDTH-1:0]     out_z       // VECTORING: angle; ROTATION: ≈0
);

    // 内部位宽 (多 2 位保护位)
    // CORDIC 每次迭代做加减法，中间结果可能溢出。输入最大值约 ±1.0（Q1.14），经过增益 K≈1.6468 倍放大后，
    // 内部幅度最大约 ±1.65。Q1.14 的整数部分只有 1 位，无法表示 1.65，因此扩展 2 位，变为 Q3.14（18-bit），整数位范围扩大到 ±4，保证全程不溢出。
    localparam IW = DATA_WIDTH + 2;  // 18，即 Q3.14（18-bit internal），整数位范围扩大到 ±4，保证全程不溢出
    
    // atan lookup table (Q2.14)
    // atan 查找表 (Q2.14 格式, rad) 存储每级的固定旋转角度
    // atan(2^-i) * 2^14
    // i=0: atan(1) = 0.7854 → 12868
    // i=1: atan(0.5) = 0.4636 → 7596
    // ... 以此类推
    wire signed [DATA_WIDTH-1:0] atan_table [0:ITERATIONS-1];
    assign atan_table[0]  = 16'sd12868;  // atan(2^0)  = 0.78540 rad
    assign atan_table[1]  = 16'sd7596;   // atan(2^-1) = 0.46365 rad
    assign atan_table[2]  = 16'sd4014;   // atan(2^-2) = 0.24498 rad
    assign atan_table[3]  = 16'sd2037;   // atan(2^-3) = 0.12435 rad
    assign atan_table[4]  = 16'sd1023;   // atan(2^-4) = 0.06242 rad
    assign atan_table[5]  = 16'sd512;    // atan(2^-5) = 0.03124 rad
    assign atan_table[6]  = 16'sd256;    // atan(2^-6) = 0.01562 rad
    assign atan_table[7]  = 16'sd128;    // atan(2^-7) = 0.00781 rad
    assign atan_table[8]  = 16'sd64;     // atan(2^-8) = 0.00391 rad
    assign atan_table[9]  = 16'sd32;     // atan(2^-9) = 0.00195 rad
    assign atan_table[10] = 16'sd16;     // atan(2^-10)
    assign atan_table[11] = 16'sd8;      // atan(2^-11)
    assign atan_table[12] = 16'sd4;      // atan(2^-12)
    assign atan_table[13] = 16'sd2;      // atan(2^-13)
    
    // Pipeline registers
    //=========================================================================
    // 流水线寄存器: x, y, z, valid 每级一组
    // 每一级迭代的结果存入下一组寄存器。数据在第 0 拍进入 *_pipe[0]，第 14 拍从 *_pipe[14] 输出，形成一条 15 级深的寄存器流水线
    //=========================================================================
    reg signed [IW-1:0] x_pipe [0:ITERATIONS];
    reg signed [IW-1:0] y_pipe [0:ITERATIONS];
    reg signed [IW-1:0] z_pipe [0:ITERATIONS];
    reg                 v_pipe [0:ITERATIONS];
    
    // 首先做符号扩展：将 16-bit Q1.14 输入扩展到 18-bit Q3.14。
    // {(IW-DATA_WIDTH){in_x[DATA_WIDTH-1]}} 是符号填充，把最高位（符号位）复制 2 次填入高位，保持负数的补码语义。
    wire signed [IW-1:0] x_ext = {{(IW-DATA_WIDTH){in_x[DATA_WIDTH-1]}}, in_x};
    wire signed [IW-1:0] y_ext = {{(IW-DATA_WIDTH){in_y[DATA_WIDTH-1]}}, in_y};

    // 【修改】因为现在 in_z 已经是 Z_WIDTH (18位)，直接赋给 z_ext，不需要再做符号扩展
    wire signed [IW-1:0] z_ext = in_z;
    
    // π in Q2.14 ≈ 51472 (3.14159 * 2^14), 但 16-bit 最大 32767...
    // 使用 18-bit 内部表示: π = 51472
    // π in Q2.14 ≈ 51472 (3.14159 * 2^14)
    localparam signed [IW-1:0] PI_POS      = 51472;
    localparam signed [IW-1:0] PI_NEG      = -51472;
    // 新增：π/2 的边界值，用于 ROTATION 全象限预处理
    localparam signed [IW-1:0] PI_HALF_POS = 25736;  
    localparam signed [IW-1:0] PI_HALF_NEG = -25736;
    
    //=========================================================================
    // 第 0 级: 输入预处理 — 将输入象限映射到第一象限
    // 对于 VECTORING: 确保 x >= 0 (若 x < 0, 取反 x,y 并补偿角度 ±π)
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                x_pipe[0] <= 0;
                y_pipe[0] <= 0;
                z_pipe[0] <= 0;
                v_pipe[0] <= 0;
            end else begin
                v_pipe[0] <= in_valid;
                if (MODE == 0) begin
                    // VECTORING: 预旋转确保 x >= 0
                    if (in_x < 0) begin
                        x_pipe[0] <= -x_ext;
                        y_pipe[0] <= -y_ext;
                        z_pipe[0] <= (in_y >= 0) ? PI_POS : PI_NEG;
                    end else begin
                        x_pipe[0] <= x_ext;
                        y_pipe[0] <= y_ext;
                        z_pipe[0] <= 0;
                    end
                end else begin
                    // ROTATION: 全象限预处理 (破除 100° 旋转极限)
                    if (z_ext > PI_HALF_POS) begin
                        x_pipe[0] <= -x_ext;
                        y_pipe[0] <= -y_ext;
                        z_pipe[0] <= z_ext - PI_POS; 
                    end else if (z_ext < PI_HALF_NEG) begin
                        x_pipe[0] <= -x_ext;
                        y_pipe[0] <= -y_ext;
                        z_pipe[0] <= z_ext - PI_NEG;
                    end else begin
                        x_pipe[0] <= x_ext;
                        y_pipe[0] <= y_ext;
                        z_pipe[0] <= z_ext;
                    end
                end
            end
        end
    
    //=========================================================================
    // 迭代级 1 ~ ITERATIONS
    //=========================================================================
    genvar i;
    generate
        for (i = 0; i < ITERATIONS; i = i + 1) begin : cordic_stage
            wire signed [IW-1:0] x_shift = x_pipe[i] >>> i;//等效于乘以 2^-i
            wire signed [IW-1:0] y_shift = y_pipe[i] >>> i;
            wire                 sigma;
            
            // 旋转方向判定
            if (MODE == 0) begin : vectoring_dir
                // VECTORING: 使 y → 0, 若 y < 0 则顺时针旋转 (sigma=0)
                // 取 y 的第17位（符号位），取 y 的符号，若 y < 0 （sigma = 1），逆时针把 y 往上推，若 y > 0（sigma = 0），顺时针把 y 往下压
                assign sigma = y_pipe[i][IW-1]; // y < 0 → sigma=1 (逆时针 → 需要加x_shift)
            end else begin : rotation_dir
                // ROTATION: 使 z → 0, 若 z ≥ 0 则逆时针旋转使 z 减小
                // 取 z 的第17位（符号位）取反: z ≥ 0 时符号位=0, ~0=1
                assign sigma = ~z_pipe[i][IW-1]; // z ≥ 0 → sigma=1 (逆时针, 减小z); z < 0 → sigma=0 (顺时针, 增大z)
            end
            
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    x_pipe[i+1] <= 0;
                    y_pipe[i+1] <= 0;
                    z_pipe[i+1] <= 0;
                    v_pipe[i+1] <= 0;
                end else begin
                    v_pipe[i+1] <= v_pipe[i];
                   
                    if (sigma) begin
                        // y < 0 (vectoring) or z ≥ 0 (rotation): 逆时针旋转
                        x_pipe[i+1] <= x_pipe[i] - y_shift; // x' = x - σ·2^{-i}·y
                        y_pipe[i+1] <= y_pipe[i] + x_shift; // y' = y + σ·2^{-i}·x
                        z_pipe[i+1] <= z_pipe[i] - {{(IW-DATA_WIDTH){atan_table[i][DATA_WIDTH-1]}}, atan_table[i]}; // z' = z - σ·arctan(2^{-i})
                    end else begin
                        // 顺时针旋转
                        x_pipe[i+1] <= x_pipe[i] + y_shift;
                        y_pipe[i+1] <= y_pipe[i] - x_shift;
                        z_pipe[i+1] <= z_pipe[i] + {{(IW-DATA_WIDTH){atan_table[i][DATA_WIDTH-1]}}, atan_table[i]};
                    end
                end
            end
        end
    endgenerate
    
    //=========================================================================
    // 输出: CORDIC 增益补偿 (乘以 1/K)
    // 1/K ≈ 0.6073 → Q1.14 = 9949
    // 对于 VECTORING 的 x 输出 (magnitude) 需要补偿
    // 对于 ROTATION 的 x, y 输出都需要补偿
    //=========================================================================
    localparam signed [DATA_WIDTH-1:0] INV_K = 16'sd9949; // 0.6073 in Q1.14
    
    wire signed [IW-1:0] x_final = x_pipe[ITERATIONS];
    wire signed [IW-1:0] y_final = y_pipe[ITERATIONS];
    wire signed [IW-1:0] z_final = z_pipe[ITERATIONS];
    
    // 增益补偿乘法
    wire signed [IW+DATA_WIDTH-1:0] x_comp = x_final * INV_K;
    wire signed [IW+DATA_WIDTH-1:0] y_comp = y_final * INV_K;
    
    // 截断回 DATA_WIDTH: 取乘积的 [ITERATIONS+DATA_WIDTH-1 : ITERATIONS]
    // 因为 x_final 是 Q3.14 (IW=18), INV_K 是 Q1.14
    // 乘积是 Q4.28 (34 bits), 需要截取到 Q1.14 (16 bits)
    assign out_x     = x_comp[ITERATIONS + DATA_WIDTH - 1 : ITERATIONS];
    assign out_y     = y_comp[ITERATIONS + DATA_WIDTH - 1 : ITERATIONS];
    assign out_z     = z_final; 
    assign out_valid = v_pipe[ITERATIONS];

endmodule
