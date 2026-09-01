module PE #(
    parameter ACT_WIDTH    = 16, // 激活值位宽
    parameter WEIGHT_WIDTH = 16, // 权重位宽
    parameter ACC_WIDTH    = 48, // 累加器（部分和）位宽
    parameter ROW_INDEX    = 0   // 当前PE所在行号，用于按行选择对应的权重加载脉冲
)(
    input  wire                        clock,
    input  wire                        reset,         // 同步复位
    input  wire                        io_inwtValid,  // 权重加载使能信号
    input  wire signed [ACT_WIDTH-1:0] io_inAct,      // 从左边的PE来的 输入的激活值/数据

    input  wire signed [ACC_WIDTH-1:0] io_inWtPS,     // 从上边传来的 权重 和 部分和 的复用输入总线
    output wire signed [ACC_WIDTH-1:0] io_outWtPS,    // 传给下边的 权重 和 部分和 的复用输出总线

    output wire signed [ACT_WIDTH-1:0] io_outAct      // 传给右边PE的数据
);

    // 内部寄存器
    reg signed [WEIGHT_WIDTH-1:0] weightReg;  // 记住属于自己的权重参数
    reg signed [ACT_WIDTH-1:0]    actReg;     // 暂存刚收到的激活值
    reg signed [ACC_WIDTH-1:0]    mulReg;     // 乘法流水级寄存器
    reg signed [ACC_WIDTH-1:0]    parSumReg;  // 暂存刚算好的乘加结果（部分和）
    reg [7:0]                     loadPulseCnt; // 统计本PE看到的权重加载脉冲个数
    reg signed [ACC_WIDTH-1:0]    wtLoadHoldReg; // 保持上一拍加载数据，供下一行在下一拍采样
    reg                           wtLoadHoldVld;

    always @(posedge clock) begin
        if (reset) begin
            weightReg     <= 0;
            actReg        <= 0;
            mulReg        <= 0;
            parSumReg     <= 0;
            loadPulseCnt  <= 8'd0;
            wtLoadHoldReg <= 0;
            wtLoadHoldVld <= 1'b0;
        end else begin
            // 在时钟上升沿 锁存激活值输入
            actReg <= io_inAct;
            
            // MAC 两级流水：第1级做乘法并缓存对应部分和，第2级做加法。
            // 这样避免“乘法+加法”单拍长路径，提升时序收敛能力。
            mulReg     <= weightReg * actReg;
            parSumReg  <= io_inWtPS + mulReg;

            // 将加载通道数据保持 1 拍，支持 valid 脉冲逐行下传时的数据对齐。
            wtLoadHoldVld <= io_inwtValid;
            if (io_inwtValid) begin
                wtLoadHoldReg <= io_inWtPS;
            end
            
            // 稀疏行推送加载：第 ROW_INDEX 次有效脉冲到达本PE时，捕获该行权重。
            if (io_inwtValid) begin
                if (loadPulseCnt == ROW_INDEX[7:0]) begin
                    weightReg <= io_inWtPS[WEIGHT_WIDTH-1:0];
                end
                loadPulseCnt <= loadPulseCnt + 1'b1;
            end
        end
    end

    // 激活数据直接向右级联
    assign io_outAct = actReg;
    
    // 输出 MUX（组合逻辑）：
    //   1) 上一拍加载有效: 透传保持寄存器 wtLoadHoldReg
    //   2) 其余: 输出计算结果 parSumReg
    // 不再在 io_inwtValid=1 的当前拍直通 io_inWtPS，避免跨PE组合长路径。
    assign io_outWtPS = wtLoadHoldVld ? wtLoadHoldReg : parSumReg;

endmodule