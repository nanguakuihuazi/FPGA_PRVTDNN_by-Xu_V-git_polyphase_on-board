`timescale 1ns / 1ps
`include "prvtdnn_defines.vh"

//=============================================================================
// prvtdnn_top — PRVTDNN DPD 推理引擎顶层模块
//
// 功能:
//   将预处理、三层隐藏层、输出层和相位反归一化串联为全流水线 PRVTDNN
//   数字预失真推理引擎。网络结构为 45 → 16 → 16 → 16 → 8；8 路输出
//   分别是 N_POLY=4 个相位归一化支路的 I/Q 分量。
//
// 数据流（每个 clk 周期输入并行的 4 个连续复样本）:
//   data_in → preprocess → Hidden_Layers → Output_Layer_Dense → phase_denorm → data_out
//              33 cycles        NN_LATENCY cycles                    2 cycles
//   NN_LATENCY 由本文件局部参数计算；当前配置下为 115 cycles，端到端
//   数据延迟为 33 + 115 + 2 = 150 cycles。流水线充满后吞吐率为
//   1 组四相复样本/clk（即 4 个复样本/clk）。
//
// 参数管理:
//   外部在 AXI_clk_i 域通过 coeff_i/coeff_addr_i/coeff_en_i 写入 708 个
//   32-bit 参数字（共 1416 个 Q1.14 参数）。最后一个字写入并撤销
//   coeff_en_i 后，跨时钟域请求触发内部加载状态机，按 HL1→HL2→HL3→OL
//   的顺序将权重和偏置装载至各层脉动阵列。load_done 是内部 1-cycle
//   完成脉冲，供仿真层级观测；开始推理前应完成一次完整参数加载。
//
// 复位规则:
//   顶层使用高有效同步复位 (reset)；
//   preprocess / phase_denorm 内部为低有效 rst_n，通过 (~reset) 桥接。
//
// 时序对齐:
//   preprocess 在 33 cycles 后同拍给出 feat_flat、pp_out_valid 和 rk。
//   rk 与 valid 经过 NN_LATENCY 级移位寄存器后，和输出层 ol_out 对齐，
//   再共同进入 phase_denorm。
//
// 文件依赖 (需在 Vivado 工程中包含以下源文件):
//   Code_Preprocess/preprocess.v
//   Code_Preprocess/cordic.v
//   Code_HiddenLayers And OutputLayer/Hidden_Layers.v
//   Code_HiddenLayers And OutputLayer/Hidden_Layer1_Dense.v
//   Code_HiddenLayers And OutputLayer/Hidden_Layer2_Dense.v
//   Code_HiddenLayers And OutputLayer/Hidden_Layer3_Dense.v
//   Code_HiddenLayers And OutputLayer/Output_Layer_Dense.v
//   Code_HiddenLayers And OutputLayer/SystolicArray.v
//   Code_HiddenLayers And OutputLayer/PE.v
//   Code_HiddenLayers And OutputLayer/BiasActivation.v
//   Code_HiddenLayers And OutputLayer/ShiftRegisterSkew.v
//   Code_PhaseDenorm/phase_denorm.v
//   Code_TOP/prvtdnn_defines.vh
//=============================================================================

module prvtdnn_top (
    input  wire        clk,               // JESD 数据通路/推理时钟
    input  wire        AXI_clk_i,         // 参数写入时钟
    input  wire        reset,             // 高有效同步复位

    // 128-bit 并行输入：包含 x(n)..x(n-3) 的 4 路 I/Q；本核按每拍有效处理
    input  wire [127:0]   data_in,
    input  wire           vio_wdpd_i,     // DPD使能 (高有效; 0=旁路输出原始 data_in)

    input   [31:0]    coeff_i,            // 两个打包的 16-bit Q1.14 权重/偏置参数
    input   [15:0]    coeff_addr_i,       // 32-bit 参数字地址：0..707
    input             coeff_en_i,         // 参数字写使能（AXI_clk_i 域）

    // 推理输出 (Q1.14 signed, 每个分支一组 I/Q)
    output wire [127:0]       data_out,
    output wire               out_valid,  // DPD 输出有效；与 y_i/y_q 对齐
    output wire [15:0]        sample0_r_ila, // ILA：DPD 开启时为 y_i[0]，旁路时为 x_re(n-3)
    output wire [15:0]        sample0_i_ila  // ILA：DPD 开启时为 y_q[0]，旁路时为 x_im(n-3)
);
    // 权重加载控制
    wire        load_start;  // 高电平 1 cycle 脉冲触发权重加载
    reg load_done;   // 加载完成 (1 cycle 脉冲, 每次加载完毕后产生)
    wire signed [(`N_POLY*16)-1:0] y_i;
    wire signed [(`N_POLY*16)-1:0] y_q;

    // DPD 旁路: vio_wdpd_i=1 输出 DPD 结果, =0 原样透传 data_in
    assign data_out = vio_wdpd_i ? {y_i, y_q} : data_in;

    assign sample0_r_ila = vio_wdpd_i ? y_i[15:0] : {data_in[7:0], data_in[39:32]};
    assign sample0_i_ila = vio_wdpd_i ? y_q[15:0] : {data_in[71:64], data_in[103:96]};

// ============================================================
// 1. 局部参数
// ============================================================

    // 各层维度 (与 prvtdnn_defines.vh 保持一致)
    localparam HL1_IN = `FC1_IN_DIM;   localparam HL1_N = `FC1_OUT_DIM;
    localparam HL2_IN = `FC2_IN_DIM;   localparam HL2_N = `FC2_OUT_DIM;
    localparam HL3_IN = `FC3_IN_DIM;   localparam HL3_N = `FC3_OUT_DIM;
    localparam OL_IN  = `FC5_IN_DIM;   localparam OL_N  = `FC5_OUT_DIM;

    // Hidden_Layers / Output_Layer_Dense 内部 DSP48E2 累加器位宽
    // (模块默认参数 ACC_WIDTH=48, 与 prvtdnn_defines.vh 的 ACC_WIDTH=40 不同)
    localparam INNER_ACC_WIDTH = 48;

    // 各层权重加载完成后的脉动阵列排空周期
    localparam HL1_DRAIN_CYC = HL1_IN + HL1_N;
    localparam HL2_DRAIN_CYC = HL2_IN + HL2_N;
    localparam HL3_DRAIN_CYC = HL3_IN + HL3_N;
    localparam OL_DRAIN_CYC  = OL_IN + OL_N;

    // NN 流水线延迟 (Hidden_Layers + Output_Layer_Dense)
    // 当前 PE 中部分和通路不再额外插入对齐寄存器，按每行额外 0 拍建模。
    localparam PE_ROW_EXTRA_CYC = 0;
    // 每个 Dense 层输出边界额外 1 拍（仿真观测: L1/L2/L3/L4 累计分别慢 1/2/3/4 拍）。
    localparam DENSE_STAGE_EXTRA_CYC = 1;
    localparam DENSE_STAGE_COUNT = 4;
    localparam NN_LATENCY_BASE = (HL1_IN + HL1_N + 1) + HL2_N + 3 + HL3_N + 3 + OL_N + 3;
    localparam NN_LATENCY = NN_LATENCY_BASE +
                            PE_ROW_EXTRA_CYC * (HL1_IN + HL2_IN + HL3_IN + OL_IN) +
                            DENSE_STAGE_EXTRA_CYC * DENSE_STAGE_COUNT;

    // (旧的 pipe_cnt 时序常量已移除, 改为状态机控制)

    // coeff_data 参数布局（16-bit 参数索引）：
    //   HL1_WT → HL1_BIAS → HL2_WT → HL2_BIAS → HL3_WT → HL3_BIAS → OL_WT → OL_BIAS。
    // 两个连续参数打包为一个 32-bit 字：偶数索引参数位于 [15:0]，奇数索引
    // 参数位于 [31:16]；参数 k 的字地址为 k >> 1。
    localparam HL1_WT_OFF   = 0;                                   // 720个 (16×45)
    localparam HL1_BIAS_OFF = HL1_WT_OFF   + HL1_N * HL1_IN;      // 720
    localparam HL2_WT_OFF   = HL1_BIAS_OFF + HL1_N;               // 736
    localparam HL2_BIAS_OFF = HL2_WT_OFF   + HL2_N * HL2_IN;      // 992
    localparam HL3_WT_OFF   = HL2_BIAS_OFF + HL2_N;               // 1008
    localparam HL3_BIAS_OFF = HL3_WT_OFF   + HL3_N * HL3_IN;      // 1264
    localparam OL_WT_OFF    = HL3_BIAS_OFF + HL3_N;               // 1280
    localparam OL_BIAS_OFF  = OL_WT_OFF    + OL_N  * OL_IN;       // 1408
    localparam TOTAL_PARAMS = OL_BIAS_OFF  + OL_N;                // 1416
    localparam COEFF_WORDS  = TOTAL_PARAMS / 2;                   // 708 (整除)

// ============================================================
// 2. 权重/偏置参数存储器 (coeff_data)
//
// coeff_data[0..COEFF_WORDS-1] 是 32-bit 参数存储阵列；外部在 AXI_clk_i
// 域按 coeff_addr_i=0..COEFF_WORDS-1 写入。加载状态机在 clk 域顺序读取
// 该阵列，并将读取结果送入各层的权重/偏置加载接口。
// ============================================================

    // coeff_data：32-bit 宽、地址从 0 开始；[15:0] 为偶数参数索引，
    // [31:16] 为奇数参数索引。
    (* ram_style = "block" *) reg [31:0] coeff_data [0:COEFF_WORDS-1];

    // 外部写入: coeff_addr_i 作为写地址 (0..COEFF_WORDS-1), coeff_i 为32位数据
    // 注意这里的时钟是慢速时钟AXI，和后面的高速时钟clk不一样
    always @(posedge AXI_clk_i) 
    begin
        if(coeff_en_i == 1)
        begin
          if (coeff_addr_i < COEFF_WORDS)
            coeff_data[coeff_addr_i] <= coeff_i;
        end
    end

// ============================================================
// 2.5 权重加载触发控制 — 检测权重全部写入完毕后产生 load_start 脉冲
//
// 当 coeff_addr_i 达到最后一个有效地址 (COEFF_WORDS-1) 且 coeff_en_i=1 时，
// 表示最后一个权重字已写入。下一拍当 coeff_en_i 从 1 变为 0 时，
// 产生 1 cycle 的 load_start 脉冲触发权重加载流水线。
// ============================================================

    // AXI 写入域内检测“最后一拍写入完成”，并翻转请求位。
    // 在 clk 域做 2FF 同步后用异或生成 1 拍 load_start 脉冲，避免跨域毛刺/亚稳态。
    reg coeff_en_prev;        // 前一拍的 coeff_en_i (AXI 时钟域)
    reg last_coeff_written;   // 标记本次事务是否已经写到最后一个地址
    reg load_req_toggle_axi;  // AXI->clk 跨域请求翻转位
    reg [2:0] load_req_sync_clk;

    always @(posedge AXI_clk_i) begin
        if (reset) begin
            coeff_en_prev      <= 1'b0;
            last_coeff_written <= 1'b0;
            load_req_toggle_axi<= 1'b0;
        end else begin
            coeff_en_prev <= coeff_en_i;

            // 标记“最后一个权重字已写入”
            if (coeff_en_i && coeff_addr_i == COEFF_WORDS - 1)
                last_coeff_written <= 1'b1;
            else if (!coeff_en_i)
                last_coeff_written <= 1'b0;

            // 在 coeff_en 下降沿且已写完最后地址时，发出一次加载请求（toggle）
            if (coeff_en_prev && !coeff_en_i && last_coeff_written)
                load_req_toggle_axi <= ~load_req_toggle_axi;
        end
    end

    always @(posedge clk) begin
        if (reset)
            load_req_sync_clk <= 3'b000;
        else
            load_req_sync_clk <= {load_req_sync_clk[1:0], load_req_toggle_axi};
    end

    assign load_start = load_req_sync_clk[2] ^ load_req_sync_clk[1];

// ============================================================
// 3. 权重加载流水线 — 寄存器声明
// ============================================================

    reg [15:0] pipe_cnt;   // 主流水线计数器 (扩大位宽以适应更长的加载周期)
    reg        loading;    // 加载进行中标志

    // 各层权重/偏置驱动总线
    reg [HL1_N*INNER_ACC_WIDTH-1:0] hl1_inWt_top;    
    reg [HL1_N-1:0]                  hl1_inWtValid;
    reg signed [HL1_N*16-1:0]        hl1_inBias;
    reg                              hl1_inBiasValid;

    reg [HL2_N*INNER_ACC_WIDTH-1:0] hl2_inWt_top;    
    reg [HL2_N-1:0]                  hl2_inWtValid;
    reg signed [HL2_N*16-1:0]        hl2_inBias;
    reg                              hl2_inBiasValid;

    reg [HL3_N*INNER_ACC_WIDTH-1:0] hl3_inWt_top;
    reg [HL3_N-1:0]                  hl3_inWtValid;
    reg signed [HL3_N*16-1:0]        hl3_inBias;
    reg                              hl3_inBiasValid;

    reg [OL_N*INNER_ACC_WIDTH-1:0]  ol_inWt_top;
    reg [OL_N-1:0]                   ol_inWtValid;
    reg signed [OL_N*16-1:0]         ol_inBias;
    reg                              ol_inBiasValid;

    integer j;

    // ---------- 顺序读取用中间寄存器 ----------
    // 缓存当前行的所有神经元权重 (逐个读取填入, 满后一次性送出)
    // 最大神经元数 = HL1_N = 16
    localparam MAX_NEURONS = HL1_N;   // 16
    reg [15:0] wt_buf [0:MAX_NEURONS-1];  // 权重缓存寄存器，临时存储当前行所有神经元的权重，逐个读入参数 → 缓存满后 → 一次性打包送入脉动阵列（systolic array）
    reg [15:0] bias_buf [0:MAX_NEURONS-1]; // 偏置缓存寄存器

    // 顺序读取状态机
    // 加载状态: IDLE -> 各层(权重逐行读取 -> 排空 -> 偏置读取) -> DONE
    localparam [3:0]
        LD_IDLE       = 4'd0,
        LD_HL1_WT_RD  = 4'd1,   // HL1 权重: 逐个读取当前行
        LD_HL1_WT_PUSH= 4'd2,   // HL1 权重: 缓存满, 送入脉动阵列
        LD_HL1_DRAIN  = 4'd3,   // HL1 排空
        LD_HL1_BIAS_RD= 4'd4,   // HL1 偏置读取
        LD_HL2_WT_RD  = 4'd5,
        LD_HL2_WT_PUSH= 4'd6,
        LD_HL2_DRAIN  = 4'd7,
        LD_HL2_BIAS_RD= 4'd8,
        LD_HL3_WT_RD  = 4'd9,
        LD_HL3_WT_PUSH= 4'd10,
        LD_HL3_DRAIN  = 4'd11,
        LD_HL3_BIAS_RD= 4'd12,
        LD_OL_WT_RD   = 4'd13,
        LD_OL_WT_PUSH = 4'd14,
        LD_OL_DRAIN   = 4'd15;
    // 额外状态 (用 5-bit 扩展)
    localparam [4:0]
        LD_OL_BIAS_RD = 5'd16,
        LD_DONE       = 5'd17;

    reg [4:0]  ld_state;
    reg [15:0] rd_param_idx;   // 当前读取的全局参数索引 (在 coeff_data 中的 16-bit 参数编号)
    reg [5:0]  col_cnt;        // 当前读状态已接收的数据计数 (response index)
    reg [5:0]  req_cnt;        // 当前读状态已发起的读请求计数 (request index)
    reg [5:0]  row_cnt;        // 当前层已完成的行计数 (0..INPUT_DIM-1)
    reg [15:0] drain_cnt;      // 排空计数器
    reg [15:0] rd_data_reg;    // 从 coeff_data 读出的 16-bit 数据 (rd_param_idx 发起后约 2 拍有效)
    reg [1:0]  rd_wait_cnt;    // 读返回预热计数: rd_param_idx 发出后需等待 2 拍
    reg [31:0] rd_word_reg;
    reg        rd_word_sel_reg;

    // coeff_data 在 clk 域同步读：
    //   cycle N   : 用 rd_param_idx 发起读
    //   cycle N+1 : rd_word_reg/rd_word_sel_reg 锁存该地址对应字
    //   cycle N+2 : rd_data_reg 返回对应 16-bit 参数
    always @(posedge clk) begin
        if (reset) begin
            rd_word_reg     <= 32'd0;
            rd_word_sel_reg <= 1'b0;
            rd_data_reg     <= 16'd0;
        end else begin
            rd_word_reg     <= coeff_data[rd_param_idx >> 1];
            rd_word_sel_reg <= rd_param_idx[0];
            rd_data_reg     <= rd_word_sel_reg ? rd_word_reg[31:16] : rd_word_reg[15:0];
        end
    end

// ============================================================
// 4. 权重加载状态机 — 逐周期顺序读取 coeff_data
//
// 设计思路 (参考 LoadWeightBias 模块):
//   对每层的每一行权重:
//     步骤 A: 用 NEURONS 个时钟周期, 逐地址从 coeff_data 读取,
//             填入中间缓存寄存器 wt_buf[0..NEURONS-1]
//     步骤 B: 缓存满后, 统一拉高 inWtValid 1 拍, 将 wt_buf 送入脉动阵列
//   重复 INPUT_DIM 行, 然后进入排空等待, 最后加载偏置。
//   每个时钟只访问 coeff_data 的一个地址, 避免多端口冲突。
// ============================================================
    
    // 权重加载状态机工作在高速时钟域 clk
    always @(posedge clk) begin
        if (reset) begin
            ld_state        <= LD_IDLE;
            loading         <= 1'b0;
            load_done       <= 1'b0;
            pipe_cnt        <= 16'd0;
            rd_param_idx    <= 16'd0;
            col_cnt         <= 6'd0;
            req_cnt         <= 6'd0;
            rd_wait_cnt     <= 2'd0;
            row_cnt         <= 6'd0;
            drain_cnt       <= 16'd0;
            hl1_inWt_top    <= {(HL1_N*INNER_ACC_WIDTH){1'b0}};
            hl1_inWtValid   <= {HL1_N{1'b0}};
            hl1_inBias      <= {(HL1_N*16){1'b0}};
            hl1_inBiasValid <= 1'b0;
            hl2_inWt_top    <= {(HL2_N*INNER_ACC_WIDTH){1'b0}};
            hl2_inWtValid   <= {HL2_N{1'b0}};
            hl2_inBias      <= {(HL2_N*16){1'b0}};
            hl2_inBiasValid <= 1'b0;
            hl3_inWt_top    <= {(HL3_N*INNER_ACC_WIDTH){1'b0}};
            hl3_inWtValid   <= {HL3_N{1'b0}};
            hl3_inBias      <= {(HL3_N*16){1'b0}};
            hl3_inBiasValid <= 1'b0;
            ol_inWt_top     <= {(OL_N*INNER_ACC_WIDTH){1'b0}};
            ol_inWtValid    <= {OL_N{1'b0}};
            ol_inBias       <= {(OL_N*16){1'b0}};
            ol_inBiasValid  <= 1'b0;

            for (j = 0; j < MAX_NEURONS; j = j + 1) begin
                wt_buf[j]   <= 16'd0;
                bias_buf[j] <= 16'd0;
            end

        end else begin
            // 默认清除所有 valid 脉冲
            hl1_inWtValid   <= {HL1_N{1'b0}};
            hl1_inBiasValid <= 1'b0;
            hl2_inWtValid   <= {HL2_N{1'b0}};
            hl2_inBiasValid <= 1'b0;
            hl3_inWtValid   <= {HL3_N{1'b0}};
            hl3_inBiasValid <= 1'b0;
            ol_inWtValid    <= {OL_N{1'b0}};
            ol_inBiasValid  <= 1'b0;
            load_done       <= 1'b0;

            case (ld_state)
            // -------------------------------------------------------
            LD_IDLE: begin
                if (load_start) begin
                    ld_state     <= LD_HL1_WT_RD;
                    loading      <= 1'b1;
                    rd_param_idx <= HL1_WT_OFF;
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_wait_cnt  <= 2'd0;
                    row_cnt      <= 6'd0;
                    pipe_cnt     <= 16'd0;
                end
            end

            // =================== HL1 ===================
            // HL1 权重读取: 每拍读一个参数, 存入 wt_buf[col_cnt]
            // 参数布局: wt_rom[neuron * INPUT_DIM + row], 但 coeff_data 中
            // 按 neuron 为外层、row 为内层连续排列:
            //   param_idx = HL1_WT_OFF + neuron * HL1_IN + row
            // 对于当前 row_cnt 行, 第 col_cnt 个神经元的参数索引:
            //   HL1_WT_OFF + col_cnt * HL1_IN + row_cnt
            LD_HL1_WT_RD: begin
                // 先连续发起读请求；每拍可发一个地址。
                if (req_cnt < HL1_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= HL1_WT_OFF + (req_cnt + 1) * HL1_IN + row_cnt;
                end

                // rd_data_reg 相对地址有 2 拍延时，预热完成后再接收数据。
                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    wt_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == HL1_N - 1) begin
                        ld_state    <= LD_HL1_WT_PUSH;
                        col_cnt     <= 6'd0;
                        req_cnt     <= 6'd0;
                        rd_wait_cnt <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            // HL1 权重推送: 将 wt_buf 打包送入 hl1_inWt_top, 拉高 valid
            LD_HL1_WT_PUSH: begin
                for (j = 0; j < HL1_N; j = j + 1)
                    hl1_inWt_top[j*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] <=
                        {{(INNER_ACC_WIDTH-16){wt_buf[j][15]}}, wt_buf[j]};
                hl1_inWtValid <= {HL1_N{1'b1}}; // 当前行的 16 个神经元权重同时有效
                pipe_cnt <= pipe_cnt + 1;  // 每推送一行, pipe_cnt 增加 (用于排空计时)
                if (row_cnt == HL1_IN - 1) begin
                    // 所有行加载完毕, 进入排空
                    ld_state  <= LD_HL1_DRAIN;
                    drain_cnt <= 16'd0;
                    row_cnt   <= 6'd0;
                    col_cnt   <= 6'd0;
                    req_cnt   <= 6'd0;
                    rd_wait_cnt <= 2'd0;
                end else begin
                    row_cnt      <= row_cnt + 1;
                    rd_param_idx <= HL1_WT_OFF + 0 * HL1_IN + (row_cnt + 1); // 下一行第 0 个神经元
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_wait_cnt  <= 2'd0;
                    ld_state     <= LD_HL1_WT_RD;
                end
            end

            // HL1 排空
            LD_HL1_DRAIN: begin
                if (drain_cnt == HL1_DRAIN_CYC - 1) begin
                    ld_state     <= LD_HL1_BIAS_RD;
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_param_idx <= HL1_BIAS_OFF;
                    rd_wait_cnt  <= 2'd0;
                end else begin
                    drain_cnt <= drain_cnt + 1;
                end
            end

            // HL1 偏置读取
            LD_HL1_BIAS_RD: begin
                if (req_cnt < HL1_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= HL1_BIAS_OFF + req_cnt + 1;
                end

                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    bias_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == HL1_N - 1) begin
                        // 偏置全部读完, 下一拍送出并转入 HL2
                        for (j = 0; j < HL1_N; j = j + 1)
                            hl1_inBias[j*16 +: 16] <= bias_buf[j];
                        // 最后一个直接用 rd_data_reg (因为 bias_buf[HL1_N-1] 还未写入)
                        hl1_inBias[(HL1_N-1)*16 +: 16] <= rd_data_reg;
                        hl1_inBiasValid <= 1'b1;
                        ld_state     <= LD_HL2_WT_RD;
                        col_cnt      <= 6'd0;
                        req_cnt      <= 6'd0;
                        row_cnt      <= 6'd0;
                        rd_param_idx <= HL2_WT_OFF + 0 * HL2_IN + 0;
                        rd_wait_cnt  <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            // =================== HL2 ===================
            LD_HL2_WT_RD: begin
                if (req_cnt < HL2_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= HL2_WT_OFF + (req_cnt + 1) * HL2_IN + row_cnt;
                end

                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    wt_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == HL2_N - 1) begin
                        ld_state    <= LD_HL2_WT_PUSH;
                        col_cnt     <= 6'd0;
                        req_cnt     <= 6'd0;
                        rd_wait_cnt <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            LD_HL2_WT_PUSH: begin
                for (j = 0; j < HL2_N; j = j + 1)
                    hl2_inWt_top[j*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] <=
                        {{(INNER_ACC_WIDTH-16){wt_buf[j][15]}}, wt_buf[j]};
                hl2_inWtValid <= {HL2_N{1'b1}};
                if (row_cnt == HL2_IN - 1) begin
                    ld_state  <= LD_HL2_DRAIN;
                    drain_cnt <= 16'd0;
                    row_cnt   <= 6'd0;
                    col_cnt   <= 6'd0;
                    req_cnt   <= 6'd0;
                    rd_wait_cnt <= 2'd0;
                end else begin
                    row_cnt      <= row_cnt + 1;
                    rd_param_idx <= HL2_WT_OFF + 0 * HL2_IN + (row_cnt + 1);
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_wait_cnt  <= 2'd0;
                    ld_state     <= LD_HL2_WT_RD;
                end
            end

            LD_HL2_DRAIN: begin
                if (drain_cnt == HL2_DRAIN_CYC - 1) begin
                    ld_state     <= LD_HL2_BIAS_RD;
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_param_idx <= HL2_BIAS_OFF;
                    rd_wait_cnt  <= 2'd0;
                end else begin
                    drain_cnt <= drain_cnt + 1;
                end
            end

            LD_HL2_BIAS_RD: begin
                if (req_cnt < HL2_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= HL2_BIAS_OFF + req_cnt + 1;
                end

                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    bias_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == HL2_N - 1) begin
                        for (j = 0; j < HL2_N; j = j + 1)
                            hl2_inBias[j*16 +: 16] <= bias_buf[j];
                        hl2_inBias[(HL2_N-1)*16 +: 16] <= rd_data_reg;
                        hl2_inBiasValid <= 1'b1;
                        ld_state     <= LD_HL3_WT_RD;
                        col_cnt      <= 6'd0;
                        req_cnt      <= 6'd0;
                        row_cnt      <= 6'd0;
                        rd_param_idx <= HL3_WT_OFF + 0 * HL3_IN + 0;
                        rd_wait_cnt  <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            // =================== HL3 ===================
            LD_HL3_WT_RD: begin
                if (req_cnt < HL3_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= HL3_WT_OFF + (req_cnt + 1) * HL3_IN + row_cnt;
                end

                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    wt_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == HL3_N - 1) begin
                        ld_state    <= LD_HL3_WT_PUSH;
                        col_cnt     <= 6'd0;
                        req_cnt     <= 6'd0;
                        rd_wait_cnt <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            LD_HL3_WT_PUSH: begin
                for (j = 0; j < HL3_N; j = j + 1)
                    hl3_inWt_top[j*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] <=
                        {{(INNER_ACC_WIDTH-16){wt_buf[j][15]}}, wt_buf[j]};
                hl3_inWtValid <= {HL3_N{1'b1}};
                if (row_cnt == HL3_IN - 1) begin
                    ld_state  <= LD_HL3_DRAIN;
                    drain_cnt <= 16'd0;
                    row_cnt   <= 6'd0;
                    col_cnt   <= 6'd0;
                    req_cnt   <= 6'd0;
                    rd_wait_cnt <= 2'd0;
                end else begin
                    row_cnt      <= row_cnt + 1;
                    rd_param_idx <= HL3_WT_OFF + 0 * HL3_IN + (row_cnt + 1);
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_wait_cnt  <= 2'd0;
                    ld_state     <= LD_HL3_WT_RD;
                end
            end

            LD_HL3_DRAIN: begin
                if (drain_cnt == HL3_DRAIN_CYC - 1) begin
                    ld_state     <= LD_HL3_BIAS_RD;
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_param_idx <= HL3_BIAS_OFF;
                    rd_wait_cnt  <= 2'd0;
                end else begin
                    drain_cnt <= drain_cnt + 1;
                end
            end

            LD_HL3_BIAS_RD: begin
                if (req_cnt < HL3_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= HL3_BIAS_OFF + req_cnt + 1;
                end

                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    bias_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == HL3_N - 1) begin
                        for (j = 0; j < HL3_N; j = j + 1)
                            hl3_inBias[j*16 +: 16] <= bias_buf[j];
                        hl3_inBias[(HL3_N-1)*16 +: 16] <= rd_data_reg;
                        hl3_inBiasValid <= 1'b1;
                        ld_state     <= LD_OL_WT_RD;
                        col_cnt      <= 6'd0;
                        req_cnt      <= 6'd0;
                        row_cnt      <= 6'd0;
                        rd_param_idx <= OL_WT_OFF + 0 * OL_IN + 0;
                        rd_wait_cnt  <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            // =================== OL ===================
            LD_OL_WT_RD: begin
                if (req_cnt < OL_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= OL_WT_OFF + (req_cnt + 1) * OL_IN + row_cnt;
                end

                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    wt_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == OL_N - 1) begin
                        ld_state    <= LD_OL_WT_PUSH;
                        col_cnt     <= 6'd0;
                        req_cnt     <= 6'd0;
                        rd_wait_cnt <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            LD_OL_WT_PUSH: begin
                for (j = 0; j < OL_N; j = j + 1)
                    ol_inWt_top[j*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] <=
                        {{(INNER_ACC_WIDTH-16){wt_buf[j][15]}}, wt_buf[j]};
                ol_inWtValid <= {OL_N{1'b1}};
                if (row_cnt == OL_IN - 1) begin
                    ld_state  <= LD_OL_DRAIN;
                    drain_cnt <= 16'd0;
                    row_cnt   <= 6'd0;
                    col_cnt   <= 6'd0;
                    req_cnt   <= 6'd0;
                    rd_wait_cnt <= 2'd0;
                end else begin
                    row_cnt      <= row_cnt + 1;
                    rd_param_idx <= OL_WT_OFF + 0 * OL_IN + (row_cnt + 1);
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_wait_cnt  <= 2'd0;
                    ld_state     <= LD_OL_WT_RD;
                end
            end

            LD_OL_DRAIN: begin
                if (drain_cnt == OL_DRAIN_CYC - 1) begin
                    ld_state     <= LD_OL_BIAS_RD;
                    col_cnt      <= 6'd0;
                    req_cnt      <= 6'd0;
                    rd_param_idx <= OL_BIAS_OFF;
                    rd_wait_cnt  <= 2'd0;
                end else begin
                    drain_cnt <= drain_cnt + 1;
                end
            end

            LD_OL_BIAS_RD: begin
                if (req_cnt < OL_N - 1) begin
                    req_cnt      <= req_cnt + 1'b1;
                    rd_param_idx <= OL_BIAS_OFF + req_cnt + 1;
                end

                if (rd_wait_cnt < 2) begin
                    rd_wait_cnt <= rd_wait_cnt + 1'b1;
                end else begin
                    bias_buf[col_cnt] <= rd_data_reg;
                    if (col_cnt == OL_N - 1) begin
                        for (j = 0; j < OL_N; j = j + 1)
                            ol_inBias[j*16 +: 16] <= bias_buf[j];
                        ol_inBias[(OL_N-1)*16 +: 16] <= rd_data_reg;
                        ol_inBiasValid <= 1'b1;
                        ld_state  <= LD_DONE;
                        req_cnt   <= 6'd0;
                        rd_wait_cnt <= 2'd0;
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                    end
                end
            end

            LD_DONE: begin
                load_done <= 1'b1;
                loading   <= 1'b0;
                ld_state  <= LD_IDLE;
            end

            default: ld_state <= LD_IDLE;
            endcase
        end
    end

// ============================================================
// 4b. data_in 解码 — 从 128-bit 并行输入中提取 N_POLY=4 组 I/Q
//     位域约定 (用户规格):
//       x_re(n-k): {data_in[(24-8k)+:8], data_in[(56-8k)+:8]}, k=0..3
//       x_im(n-k): {data_in[(88-8k)+:8], data_in[(120-8k)+:8]}, k=0..3
// ============================================================

    wire signed [15:0] cur_re [0:3];
    wire signed [15:0] cur_im [0:3];

    assign cur_re[0] = {data_in[24+:8], data_in[ 56+:8]};  // x_re(n)
    assign cur_re[1] = {data_in[16+:8], data_in[ 48+:8]};  // x_re(n-1)
    assign cur_re[2] = {data_in[ 8+:8], data_in[ 40+:8]};  // x_re(n-2)
    assign cur_re[3] = {data_in[ 0+:8], data_in[ 32+:8]};  // x_re(n-3)

    assign cur_im[0] = {data_in[88+:8], data_in[120+:8]};  // x_im(n)
    assign cur_im[1] = {data_in[80+:8], data_in[112+:8]};  // x_im(n-1)
    assign cur_im[2] = {data_in[72+:8], data_in[104+:8]};  // x_im(n-2)
    assign cur_im[3] = {data_in[64+:8], data_in[ 96+:8]};  // x_im(n-3)

    // 打包为 flat 总线传入 preprocess
    // flat[k*16+:16] = x(n-k), k=0..N_POLY-1
    wire [`N_POLY*16-1:0] pp_x_re_flat;
    wire [`N_POLY*16-1:0] pp_x_im_flat;
    assign pp_x_re_flat = {cur_re[3], cur_re[2], cur_re[1], cur_re[0]};
    assign pp_x_im_flat = {cur_im[3], cur_im[2], cur_im[1], cur_im[0]};

// ============================================================
// 5. 推理数据通路内部连线
// ============================================================

    // preprocess 输出
    wire [HL1_IN*16-1:0]         pp_feat_flat;
    wire                          pp_out_valid;
    wire signed [(`N_POLY*16)-1:0] pp_rk_re_flat;
    wire signed [(`N_POLY*16)-1:0] pp_rk_im_flat;

    // Hidden_Layers 输出 → Output_Layer_Dense 输入 (HL3_N×16-bit)
    wire signed [HL3_N*16-1:0] hl_out;

    // Output_Layer_Dense 输出 → phase_denorm 输入 (2*N_poly × 16-bit)
    wire signed [OL_N*16-1:0]  ol_out;
    wire signed [(`N_POLY*16)-1:0] nn_out_i_flat = ol_out[(`N_POLY*16)-1:0];
    wire signed [(`N_POLY*16)-1:0] nn_out_q_flat = ol_out[(2*`N_POLY*16)-1:(`N_POLY*16)];

// ============================================================
// 6. 时序对齐: rk / valid 流水线移位寄存器
//
// preprocess 在延迟 33 cycles 后同拍输出 feat_flat, pp_out_valid,
// pp_rk_re, pp_rk_im。Hidden_Layers + Output_Layer_Dense 总延迟
// NN_LATENCY cycles，故需将 rk 和 valid 各延迟 NN_LATENCY cycles 后送入 phase_denorm，
// 使其与 ol_out 精确对齐。
// ============================================================

    // rk / valid 需延迟恰好 NN_LATENCY 拍才能与 ol_out 对齐。
    // valid_sr 共 NN_LATENCY+1 位, pp_out_valid 进入 bit[0], 经 NN_LATENCY 次移位到达 bit[NN_LATENCY]。
    // rk_re_sr / rk_im_sr 同理: 共 NN_LATENCY+1 级, 使用索引 [NN_LATENCY]。
    reg [NN_LATENCY:0] valid_sr;
    reg signed [(`N_POLY*16)-1:0] rk_re_sr [0:NN_LATENCY];
    reg signed [(`N_POLY*16)-1:0] rk_im_sr [0:NN_LATENCY];

    integer i;   // 移位寄存器循环变量

    always @(posedge clk) begin
        if (reset) begin
            valid_sr <= {(NN_LATENCY+1){1'b0}};
            for (i = 0; i <= NN_LATENCY; i = i + 1) begin
                rk_re_sr[i] <= {(`N_POLY*16){1'b0}};
                rk_im_sr[i] <= {(`N_POLY*16){1'b0}};
            end
        end else begin
            valid_sr <= {valid_sr[NN_LATENCY-1:0], pp_out_valid};
            rk_re_sr[0] <= pp_rk_re_flat;
            rk_im_sr[0] <= pp_rk_im_flat;
            for (i = 1; i <= NN_LATENCY; i = i + 1) begin
                rk_re_sr[i] <= rk_re_sr[i-1];
                rk_im_sr[i] <= rk_im_sr[i-1];
            end
        end
    end

// ============================================================
// 7. 子模块实例化
// ============================================================

    //----------------------------------------------------------
    // 7.1  preprocess — 预处理 (rst_n 低有效, 取反桥接)
    //      延迟 33 cycles, 输出 feat_flat / pp_out_valid / rk_re / rk_im
    //----------------------------------------------------------
    preprocess u_preprocess (
        .clk        ( clk           ),
        .rst_n      ( ~reset        ),   // 低有效复位
        .x_re_flat  ( pp_x_re_flat  ),   // 4路实部并行输入, flat[k*16+:16]=x_re(n-k)
        .x_im_flat  ( pp_x_im_flat  ),   // 4路虚部并行输入, flat[k*16+:16]=x_im(n-k)
        .feat_flat  ( pp_feat_flat  ),
        .out_valid  ( pp_out_valid  ),
        .rk_re_flat ( pp_rk_re_flat ),
        .rk_im_flat ( pp_rk_im_flat )
    );

    //----------------------------------------------------------
    // 7.2  Hidden_Layers — 三隐藏层串联 (reset 高有效)
    //      全局 NN 流水线延迟 = NN_LATENCY cycles
    //----------------------------------------------------------
    Hidden_Layers #(
        .HL1_IN_DIM  (HL1_IN),
        .HL1_NEURONS (HL1_N),
        .HL2_NEURONS (HL2_N),
        .HL3_NEURONS (HL3_N)
    ) u_hidden_layers (
        .clock          ( clk             ),
        .reset          ( reset           ),
        // 数据通路
        .io_inAct       ( pp_feat_flat    ),
        .io_outAct      ( hl_out          ),
        // HL1 权重/偏置加载接口 (45→16)
        .hl1_inWt_top   ( hl1_inWt_top   ),
        .hl1_inWtValid  ( hl1_inWtValid  ),
        .hl1_inBias     ( hl1_inBias     ),
        .hl1_inBiasValid( hl1_inBiasValid),
        // HL2 权重/偏置加载接口 (16→16)
        .hl2_inWt_top   ( hl2_inWt_top   ),
        .hl2_inWtValid  ( hl2_inWtValid  ),
        .hl2_inBias     ( hl2_inBias     ),
        .hl2_inBiasValid( hl2_inBiasValid),
        // HL3 权重/偏置加载接口 (16→16)
        .hl3_inWt_top   ( hl3_inWt_top   ),
        .hl3_inWtValid  ( hl3_inWtValid  ),
        .hl3_inBias     ( hl3_inBias     ),
        .hl3_inBiasValid( hl3_inBiasValid)
    );

    //----------------------------------------------------------
    // 7.3  Output_Layer_Dense — 输出层线性变换 16→8 (reset 高有效)
    //      内部已含 ShiftRegisterSkew (DIRECTION=1) 完成 de-skew,
    //      io_outAct 输出已对齐, 无需额外延迟。
    //      Sigmoid LUT 接口本层不使用: io_outSigAddr 悬空,
    //      io_inSigData 接零。
    //----------------------------------------------------------
    Output_Layer_Dense #(
        .INPUT_DIM (OL_IN),
        .NEURONS   (OL_N)
    ) u_output_layer (
        .clock         ( clk           ),
        .reset         ( reset         ),
        // 数据通路
        .io_inAct      ( hl_out        ),
        .io_outAct     ( ol_out        ),
        // 权重/偏置加载接口 (16→8)
        .io_inWt_top   ( ol_inWt_top  ),
        .io_inWtValid  ( ol_inWtValid ),
        .io_inBias     ( ol_inBias    ),
        .io_inBiasValid( ol_inBiasValid),
        // Sigmoid LUT 接口 (本层 ACT_FUNC=0 Linear, 不使用)
        .io_outSigAddr (                 ),
        .io_inSigData  ( {(OL_N*16){1'b0}} )
    );

    //----------------------------------------------------------
    // 7.4  phase_denorm — 相位反归一化 (rst_n 低有效, 取反桥接)
    //      延迟 2 cycles: out_i/out_q/out_valid 在 nn_valid 之后 2 拍有效
    //      端到端总延迟 = 33 (preprocess) + NN_LATENCY (NN) + 2 (phase_denorm)
    //----------------------------------------------------------
    phase_denorm u_phase_denorm (
        .clk      ( clk            ),
        .rst_n    ( ~reset         ),   // 低有效复位
        .nn_out_i_flat ( nn_out_i_flat               ),
        .nn_out_q_flat ( nn_out_q_flat               ),
        .nn_valid      ( valid_sr[NN_LATENCY]      ),  // 与 ol_out 延迟对齐
        .rk_re_flat    ( rk_re_sr[NN_LATENCY]      ),  // 同理
        .rk_im_flat    ( rk_im_sr[NN_LATENCY]      ),  // 同理
        .out_i_flat    ( y_i                         ),
        .out_q_flat    ( y_q                         ),
        .out_valid     ( out_valid                   )
    );

endmodule
