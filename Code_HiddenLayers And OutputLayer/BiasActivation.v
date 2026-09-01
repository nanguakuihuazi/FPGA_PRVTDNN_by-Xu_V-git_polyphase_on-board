module BiasActivation #(
    parameter ACC_WIDTH    = 48,//指脉动阵列输出的“部分和”以及偏置加法结果的位宽 。
    parameter ACT_WIDTH    = 16,//输入神经元的特征图（Feature Map）以及最终输出结果的位宽 。
    parameter BIAS_WIDTH   = 16,//存储在偏置寄存器（biasReg）中的偏置参数位宽 。

    parameter FIXED_POINT  = 28, // 神经网络权重的总定点位置
    parameter BIAS_FP      = 14, // 偏置参数的定点位置
    parameter MESH_COLUMNS = 16,  // 脉动阵列的列数 (即并行计算通道数)
    parameter INT_NUM      = 1,  // 整数部分的位宽
    parameter ACT_FUNC     = 1,  // 0: Linear, 1: ReLU, 2: LeakyReLU
    parameter LEAKY_SHIFT  = 3   // LeakyReLU斜率 = 2^(-LEAKY_SHIFT), 默认 0.125
)(
    input  wire                                     clk,
    input  wire                                     reset,
    
    // 从脉动阵列输入的部分和 (平铺为一维数组传递)
    input  wire signed [MESH_COLUMNS*ACC_WIDTH-1:0] io_inSum,
    
    // 偏置加载端口
    input  wire signed [MESH_COLUMNS*BIAS_WIDTH-1:0] io_inBias,//偏置加载端口,通过 io_inBiasValid 信号控制何时更新内部的偏置寄存器
    input  wire                                      io_inBiasValid,
    
    // 最终输出的激活值
    // 总位宽 = neurons数量 × 16 bits/neuron 
    output wire signed [MESH_COLUMNS*ACT_WIDTH-1:0]  io_outAct//最终输出的激活值（16 位），准备送入下一层
);

    localparam SIGN_BIT = FIXED_POINT + INT_NUM;        //确定符号位（Sign Bit）在宽位宽数据中的绝对位置。
    localparam SHIFT_AMOUNT = FIXED_POINT - BIAS_FP;    //计算偏置（Bias）需要左移的位数，以实现“小数点对齐”。
    localparam ACT_LINEAR = 0;
    localparam ACT_RELU = 1;
    localparam ACT_LEAKY_RELU = 2;

    // 内部寄存器阵列声明
    reg signed [BIAS_WIDTH-1:0] biasReg [0:MESH_COLUMNS-1];//存储当前神经网络层的偏置 (Bias) 参数。
    reg signed [ACC_WIDTH-1:0]  plusReg [0:MESH_COLUMNS-1];//存储 部分和（来自PE输出） 与偏置相加后的中间结果。
    reg signed [ACT_WIDTH-1:0]  actReg  [0:MESH_COLUMNS-1];//存储经过非线性激活函数处理后的最终输出值。
    reg                         sumReg  [0:MESH_COLUMNS-1]; // 专门用来记录输入部分和（inSum）在加法运算前的原始符号位。
    reg                         validReg;

    wire signed [ACT_WIDTH-1:0] truncWire [0:MESH_COLUMNS-1];
    wire signed [ACC_WIDTH-1:0] inSumWire  [0:MESH_COLUMNS-1];
    wire signed [ACC_WIDTH-1:0] biasShiftWire [0:MESH_COLUMNS-1];

    genvar k;
    generate
        for (k = 0; k < MESH_COLUMNS; k = k + 1) begin : signed_align
            // Keep sum/bias math in ACC_WIDTH signed domain to avoid partial-select unsigned promotion.
            assign inSumWire[k] = $signed(io_inSum[k*ACC_WIDTH +: ACC_WIDTH]);
            assign biasShiftWire[k] =
                ($signed({{(ACC_WIDTH-BIAS_WIDTH){biasReg[k][BIAS_WIDTH-1]}}, biasReg[k]}) <<< SHIFT_AMOUNT);
        end
    endgenerate

    integer i;

    always @(posedge clk) begin
        if (reset) begin
            validReg <= 1'b0;
            for (i = 0; i < MESH_COLUMNS; i = i + 1) begin
                biasReg[i] <= 0;
                plusReg[i] <= 0;
                sumReg[i]  <= 0;
                actReg[i]  <= 0;
            end
        end else begin
            validReg <= io_inBiasValid;
            
            for (i = 0; i < MESH_COLUMNS; i = i + 1) begin
                // 1. 如果有效，锁存新的偏置数据
                if (validReg) begin
                    // 从一维总线中拆解对应通道的 bias 数据
                    biasReg[i] <= io_inBias[i*BIAS_WIDTH +: BIAS_WIDTH]; 
                end
                
                // 2. 定点数对齐与加偏置: 对 Bias 左移以匹配 Sum 的定点位置
                plusReg[i] <= inSumWire[i] + biasShiftWire[i];
                
                // 获取原输入数据的符号位（位于指定的 Sign Bit 处）以备用作饱和截断判断
                sumReg[i]  <= inSumWire[i][SIGN_BIT];
                
                // 3. 激活函数逻辑 (时序逻辑)
                if (ACT_FUNC == ACT_RELU) begin
                    // ReLU 激活
                    actReg[i] <= (truncWire[i] >= 0) ? truncWire[i] : {ACT_WIDTH{1'b0}};
                end
                else if (ACT_FUNC == ACT_LEAKY_RELU) begin
                    // Leaky ReLU 激活: 负值算术右移 LEAKY_SHIFT 位
                    actReg[i] <= truncWire[i][ACT_WIDTH-1] ? (truncWire[i] >>> LEAKY_SHIFT) : truncWire[i];
                end 
                else begin
                    // Linear (直通)
                    actReg[i] <= truncWire[i];
                end
            end
        end
    end

    // 组合逻辑: 溢出 (Overflow) 与下溢 (Underflow) 检查及数据截断
    genvar j;
    generate
        for (j = 0; j < MESH_COLUMNS; j = j + 1) begin : trunc_block
            wire underflow = (~plusReg[j][SIGN_BIT]) & sumReg[j] & biasReg[j][BIAS_WIDTH-1];
            wire overflow  = plusReg[j][SIGN_BIT] & (~sumReg[j]) & (~biasReg[j][BIAS_WIDTH-1]);

            assign truncWire[j] = underflow ? 16'h7FFF : // 饱和到正最大值
                                  overflow  ? 16'h8000 : // 饱和到负最大值
                                  // 正常截取: 从SIGN_BIT开始向下截取ACT_WIDTH位
                                  plusReg[j][SIGN_BIT -: ACT_WIDTH]; //从 SIGN_BIT（符号位所在位置）开始，向低位方向连续截取 ACT_WIDTH（16 位）数据。

            // 对外连线映射
            assign io_outAct[j*ACT_WIDTH +: ACT_WIDTH] = actReg[j];
        end
    endgenerate

endmodule