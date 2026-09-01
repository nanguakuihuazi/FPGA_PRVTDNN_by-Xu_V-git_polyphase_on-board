module SystolicArray #(
    parameter ROWS         = 28, // 矩阵行数 (对应输入特征图的维度)
    parameter COLS         = 16, // 矩阵列数 (对应当前层的神经元个数)
    parameter ACT_WIDTH    = 16, // 激活值位宽
    parameter WEIGHT_WIDTH = 16, // 权重位宽
    parameter ACC_WIDTH    = 48  // 累加器位宽 (匹配 DSP48 原语)
)(
    input  wire clock,
    input  wire reset,

    // 左侧：ROWS 个激活值输入，平铺成一维总线 (ROWS*ACT_WIDTH 位)
    input  wire [ROWS*ACT_WIDTH-1:0]   io_inAct_left,

    // 顶部：COLS 个权重/部分和复用总线输入，平铺 (COLS*ACC_WIDTH 位)
    input  wire [COLS*ACC_WIDTH-1:0]   io_inWtPS_top,

    // 顶部：COLS 个权重加载使能信号，平铺 (COLS 位)
    input  wire [COLS-1:0]             io_inwtValid,

    // 底部：COLS 个最终部分和输出，平铺 (COLS*ACC_WIDTH 位)
    output wire [COLS*ACC_WIDTH-1:0]   io_outWtPS_bot
);

    // COLS+1 列用于激活线网（横向传播需要 COLS+1 列出口）
    localparam COLS1 = COLS + 1;

    // ==========================================
    // 内部平铺互连线网
    // ==========================================
    //   act_wire[r][c]  : ROWS 行 × COLS1 列，每元素 ACT_WIDTH 位
    //   行主序访问: act_wire[r][c] → act_wire[(r*COLS1+c)*ACT_WIDTH +: ACT_WIDTH]
    wire [ROWS*COLS1*ACT_WIDTH-1:0]    act_wire;

    //   wtps_wire[r][c] : (ROWS+1) 行 × COLS 列，每元素 ACC_WIDTH 位
    //   行主序访问: wtps_wire[r][c] → wtps_wire[(r*COLS+c)*ACC_WIDTH +: ACC_WIDTH]
    wire [(ROWS+1)*COLS*ACC_WIDTH-1:0] wtps_wire;

    //   valid_wire[r][c] : (ROWS+1) 行 × COLS 列，每元素 1 位
    //   行主序访问: valid_wire[r][c] → valid_wire[r*COLS+c]
    wire [(ROWS+1)*COLS-1:0]           valid_wire;

    // ==========================================
    // 边界输入 / 输出赋值
    // ==========================================
    genvar i;
    generate
        // 左边界：将外部激活输入绑定到每行第 0 列
        for (i = 0; i < ROWS; i = i + 1) begin : act_in_assign
            assign act_wire[i*COLS1*ACT_WIDTH +: ACT_WIDTH] =
                   io_inAct_left[i*ACT_WIDTH +: ACT_WIDTH];
        end

        // 顶部边界：权重/部分和与有效信号绑定到第 0 行
        // 底部边界：最终部分和从第 ROWS 行取出
        for (i = 0; i < COLS; i = i + 1) begin : top_in_assign
            assign wtps_wire[i*ACC_WIDTH +: ACC_WIDTH] =
                   io_inWtPS_top[i*ACC_WIDTH +: ACC_WIDTH];
            assign valid_wire[i] = io_inwtValid[i];

            assign io_outWtPS_bot[i*ACC_WIDTH +: ACC_WIDTH] =
                   wtps_wire[(ROWS*COLS + i)*ACC_WIDTH +: ACC_WIDTH];
        end
    endgenerate

    // ==========================================
    // PE 矩阵实例化 + valid 流水线
    // ==========================================
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : row_gen
            for (c = 0; c < COLS; c = c + 1) begin : col_gen

                PE #(
                    .ACT_WIDTH   (ACT_WIDTH),
                    .WEIGHT_WIDTH(WEIGHT_WIDTH),
                    .ACC_WIDTH   (ACC_WIDTH),
                    .ROW_INDEX   (r)
                ) pe_inst (
                    .clock       ( clock ),
                    .reset       ( reset ),
                    // 横向：激活值从左向右传递
                    .io_inAct    ( act_wire  [(r*COLS1 + c    )*ACT_WIDTH +: ACT_WIDTH] ),// 从左边的PE来的 输入的激活值/数据
                    .io_outAct   ( act_wire  [(r*COLS1 + c + 1)*ACT_WIDTH +: ACT_WIDTH] ),// 传给右边PE的数据
                    // 纵向：权重/部分和从上向下传递
                    .io_inWtPS   ( wtps_wire [(r*COLS   + c  )*ACC_WIDTH  +: ACC_WIDTH] ),// 从上边传来的 权重 和 部分和 的复用输入总线
                    .io_outWtPS  ( wtps_wire [((r+1)*COLS+c  )*ACC_WIDTH  +: ACC_WIDTH] ),// 传给下边的 权重 和 部分和 的复用输出总线
                    // 权重加载使能
                    .io_inwtValid( valid_wire[r*COLS + c] )// 权重加载使能信号
                );

                // valid 信号流水线：每经过一行延迟一拍
                reg valid_delay_reg;
                always @(posedge clock) begin
                    if (reset)
                        valid_delay_reg <= 1'b0;
                    else
                        valid_delay_reg <= valid_wire[r*COLS + c];
                end
                assign valid_wire[(r+1)*COLS + c] = valid_delay_reg;

            end
        end
    endgenerate

endmodule