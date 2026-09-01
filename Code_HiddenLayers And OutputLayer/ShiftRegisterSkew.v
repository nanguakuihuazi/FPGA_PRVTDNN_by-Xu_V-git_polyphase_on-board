//=============================================================================
// ShiftRegisterSkew — 输入错位 / 输出重对齐模块
//
// 功能:
//   将 NUM_CH 路并行数据通道在时间上做渐进式延迟,
//   使脉动阵列可实现每周期一个样本的全流水线吞吐。
//
// 参数 DIRECTION:
//   0 = 升序延迟 (输入错位, Input Skewing)
//       channel i 延迟 i 个时钟周期
//       用于脉动阵列左侧输入, 使 Row r 的数据晚 r 拍到达。
//
//   1 = 降序延迟 (输出重对齐, Output De-skewing)
//       channel i 延迟 (NUM_CH-1-i) 个时钟周期
//       用于脉动阵列底部输出, 将各列时间差对齐到同一拍。
//
// 资源:
//   总寄存器数 = NUM_CH*(NUM_CH-1)/2 个 DATA_WIDTH 位寄存器
//=============================================================================
module ShiftRegisterSkew #(
    parameter NUM_CH     = 28,   // 通道数
    parameter DATA_WIDTH = 16,   // 每通道数据位宽
    parameter DIRECTION  = 0     // 0: 升序 (输入错位), 1: 降序 (输出重对齐)
)(
    input  wire clock,
    input  wire reset,
    input  wire [NUM_CH*DATA_WIDTH-1:0] io_in,
    output wire [NUM_CH*DATA_WIDTH-1:0] io_out
);

    genvar ch;
    generate
        for (ch = 0; ch < NUM_CH; ch = ch + 1) begin : skew_ch

            // 根据方向计算当前通道需要的延迟级数
            // DIRECTION=0: DELAY = ch
            // DIRECTION=1: DELAY = NUM_CH - 1 - ch
            localparam DELAY = (DIRECTION == 0) ? ch : (NUM_CH - 1 - ch);

            if (DELAY == 0) begin : zero_delay
                // 零延迟：直接连通
                assign io_out[ch*DATA_WIDTH +: DATA_WIDTH] =
                       io_in [ch*DATA_WIDTH +: DATA_WIDTH];
            end else begin : has_delay

                // 移位寄存器链：DELAY 级深度
                // sr[0..DELAY-1]：DELAY 个 16-bit 寄存器，例如 ch=5，DELAY=5，则 sr[0..4] 共 5 级
                reg [DATA_WIDTH-1:0] sr [0:DELAY-1];

                integer d;
                always @(posedge clock) begin
                    if (reset) begin
                        for (d = 0; d < DELAY; d = d + 1)
                            sr[d] <= {DATA_WIDTH{1'b0}};
                    end else begin
                        sr[0] <= io_in[ch*DATA_WIDTH +: DATA_WIDTH];//将当前通道的输入数据通过动态位宽截取的方式存入移位寄存器链的首级 sr[0]
                        for (d = 1; d < DELAY; d = d + 1)
                            sr[d] <= sr[d-1];//通过 for 循环，每个时钟周期将前一级的数据（sr[d-1]）传递给后一级（sr[d]）
                    end
                end

                assign io_out[ch*DATA_WIDTH +: DATA_WIDTH] = sr[DELAY-1];
            end
        end
    endgenerate

endmodule
