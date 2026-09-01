`timescale 1ns / 1ps
`include "prvtdnn_defines.vh"

//=============================================================================
// prvtdnn_top_tb - PRVTDNN V6 top-level end-to-end simulation
//
// V6 interface features covered by this TB:
// 1) AXI-style parameter loading through coeff_i/coeff_addr_i/coeff_en_i
// 2) 128-bit parallel input stream (data_in)
// 3) Full pipeline checks against MATLAB golden files:
//    preprocess feat/rk, hidden layers, output layer, phase_denorm
//=============================================================================

module prvtdnn_top_tb;

    // ------------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------------
    localparam CLK_PERIOD       = 10;
    localparam ACT_WIDTH        = 16;
    localparam WEIGHT_WIDTH     = 16;
    localparam BIAS_WIDTH       = 16;
    localparam INNER_ACC_WIDTH  = 48;

    localparam HL1_IN_DIM       = `FC1_IN_DIM;   // 45
    localparam HL1_NEURONS      = `FC1_OUT_DIM;  // 24
    localparam HL2_IN_DIM       = `FC2_IN_DIM;
    localparam HL2_NEURONS      = `FC2_OUT_DIM;  // 16
    localparam HL3_IN_DIM       = `FC3_IN_DIM;
    localparam HL3_NEURONS      = `FC3_OUT_DIM;  // 16
    localparam OL_IN_DIM        = `FC5_IN_DIM;
    localparam OL_NEURONS       = `FC5_OUT_DIM;  // 8
    localparam N_POLY           = `N_POLY;       // 4
    localparam FEAT_DIM         = `FEAT_DIM;     // 45
    localparam L1_BIAS_SHIFT    = 14;

    localparam NUM_CYCLES       = 50;   // data_in.hex lines
    localparam TOLERANCE        = 4;
    localparam PP_INIT_SKIP     = 16;   // Skip initial PP invalid window; do not apply to L1-L4/PD alignment.

    localparam PP_NUM_GOLD      = NUM_CYCLES * FEAT_DIM;      // 2250
    localparam RK_NUM_GOLD      = NUM_CYCLES * N_POLY;        // 200
    localparam L1_NUM_GOLD      = NUM_CYCLES * HL1_NEURONS;   // 1600
    localparam L2_NUM_GOLD      = NUM_CYCLES * HL2_NEURONS;   // 800
    localparam L3_NUM_GOLD      = NUM_CYCLES * HL3_NEURONS;   // 800
    localparam L4_NUM_GOLD      = NUM_CYCLES * OL_NEURONS;    // 400

    // L1 parameter layout offsets in 16-bit parameter index domain.
    localparam HL1_WT_OFF       = 0;
    localparam HL1_BIAS_OFF     = HL1_WT_OFF + HL1_NEURONS * HL1_IN_DIM;
    localparam HL2_WT_OFF       = HL1_BIAS_OFF + HL1_NEURONS;
    localparam HL2_BIAS_OFF     = HL2_WT_OFF + HL2_NEURONS * HL2_IN_DIM;
    localparam HL3_WT_OFF       = HL2_BIAS_OFF + HL2_NEURONS;
    localparam HL3_BIAS_OFF     = HL3_WT_OFF + HL3_NEURONS * HL3_IN_DIM;
    localparam OL_WT_OFF        = HL3_BIAS_OFF + HL3_NEURONS;
    localparam OL_BIAS_OFF      = OL_WT_OFF + OL_NEURONS * OL_IN_DIM;

    localparam PP_LATENCY       = 33;
    // 当前 PE 时序下，按每经过 1 行额外增加 0 拍建模。
    localparam PE_ROW_EXTRA_CYC = 0;
    // 每个 Dense 层输出边界额外 1 拍（仿真观测: L1/L2/L3/L4 累计分别慢 1/2/3/4 拍）。
    localparam DENSE_STAGE_EXTRA_CYC = 1;
    localparam L1_NN_LATENCY    = HL1_IN_DIM + HL1_NEURONS + 1 + HL1_IN_DIM * PE_ROW_EXTRA_CYC + DENSE_STAGE_EXTRA_CYC;
    localparam L2_NN_LATENCY    = L1_NN_LATENCY + HL2_NEURONS + 3 + HL2_IN_DIM * PE_ROW_EXTRA_CYC + DENSE_STAGE_EXTRA_CYC;
    localparam L3_NN_LATENCY    = L2_NN_LATENCY + HL3_NEURONS + 3 + HL3_IN_DIM * PE_ROW_EXTRA_CYC + DENSE_STAGE_EXTRA_CYC;
    localparam L4_NN_LATENCY    = L3_NN_LATENCY + OL_NEURONS  + 3 + OL_IN_DIM  * PE_ROW_EXTRA_CYC + DENSE_STAGE_EXTRA_CYC;

    localparam L1_LATENCY       = PP_LATENCY + L1_NN_LATENCY;
    localparam L2_LATENCY       = PP_LATENCY + L2_NN_LATENCY;
    localparam L3_LATENCY       = PP_LATENCY + L3_NN_LATENCY;
    localparam L4_LATENCY       = PP_LATENCY + L4_NN_LATENCY;

    // ------------------------------------------------------------------------
    // Waveform/check alignment compensation (unit: clk cycles)
    // NOTE: set to 0 as baseline. If a stage still leads/lags, tune per-stage.
    // ------------------------------------------------------------------------
    localparam PP_RT_LEAD_CYC   = 0;
    localparam L1_RT_LEAD_CYC   = 0;
    // L2 file-level alignment (l2_fpga_rt row1 ~ l2_gold_rt row11):
    // compensate by 10 sample clocks for current run.
    localparam L2_RT_LEAD_CYC   = 10;
    // L3 file-level alignment (l3_fpga_rt row2 ~ l3_gold_rt row22):
    // compensate by 20 sample clocks for current run.
    localparam L3_RT_LEAD_CYC   = 20;
    localparam L4_RT_LEAD_CYC   = 0;
    // Keep heavyweight PE internal diagnostics off for faster/cleaner regression.
    localparam ENABLE_INTERNAL_PE_DIAG = 0;

    localparam PP_RT_LATENCY    = (PP_LATENCY - 1) - PP_RT_LEAD_CYC;
    // Empirical alignment from l1_diag_skew_check: first active L1 sample appears at cycle 48.
    // Keep formula-derived latency for reference, but use tuned value for compare/dump window.
    localparam L1_RT_LATENCY    = 48 - L1_RT_LEAD_CYC;
    // Use L1 empirical anchor and preserve inter-stage delta to retune later stages.
    localparam L2_RT_LATENCY    = L1_RT_LATENCY + (L2_LATENCY - L1_LATENCY) - L2_RT_LEAD_CYC;
    localparam L3_RT_LATENCY    = L1_RT_LATENCY + (L3_LATENCY - L1_LATENCY) - L3_RT_LEAD_CYC;
    localparam L4_RT_LATENCY    = L1_RT_LATENCY + (L4_LATENCY - L1_LATENCY) - L4_RT_LEAD_CYC;
    localparam PD_RT_LATENCY    = L4_RT_LATENCY + PD_LATENCY;

    // L1-L4 checker uses cycle_cnt gate; apply the same lead compensation.
    localparam L1_CHK_LATENCY   = L1_RT_LATENCY;
    localparam L2_CHK_LATENCY   = L2_RT_LATENCY;
    localparam L3_CHK_LATENCY   = L3_RT_LATENCY;
    localparam L4_CHK_LATENCY   = L4_RT_LATENCY;

    localparam PD_LATENCY       = 2;
    localparam E2E_LATENCY      = PP_LATENCY + L4_NN_LATENCY + PD_LATENCY;
    localparam TOTAL_CYCLES     = NUM_CYCLES + E2E_LATENCY + 30;

    localparam COEFF_WORDS      = `TOTAL_PARAMS / 2; // 708

    // ------------------------------------------------------------------------
    // DUT I/O
    // ------------------------------------------------------------------------
    reg                clk;
    reg                AXI_clk_i;
    reg                reset;

    reg  [127:0]       data_in;
    reg                vio_wdpd_i;

    reg  [31:0]        coeff_i;
    reg  [15:0]        coeff_addr_i;
    reg                coeff_en_i;

    wire [127:0]       data_out;
    wire               out_valid;
    wire [15:0]        sample0_r_ila;
    wire [15:0]        sample0_i_ila;

    wire signed [(N_POLY*16)-1:0] y_i;
    wire signed [(N_POLY*16)-1:0] y_q;
    assign y_i = data_out[127:64];
    assign y_q = data_out[63:0];

    // ------------------------------------------------------------------------
    // DUT instance
    // ------------------------------------------------------------------------
    prvtdnn_top u_dut (
        .clk            (clk),
        .AXI_clk_i      (AXI_clk_i),
        .reset          (reset),
        .data_in        (data_in),
        .vio_wdpd_i     (vio_wdpd_i),
        .coeff_i        (coeff_i),
        .coeff_addr_i   (coeff_addr_i),
        .coeff_en_i     (coeff_en_i),
        .data_out       (data_out),
        .out_valid      (out_valid),
        .sample0_r_ila  (sample0_r_ila),
        .sample0_i_ila  (sample0_i_ila)
    );

    // ------------------------------------------------------------------------
    // De-skew taps for hidden-layer checking
    // ------------------------------------------------------------------------
    wire signed [HL1_NEURONS*ACT_WIDTH-1:0] l1_deskewed;
    wire signed [HL2_NEURONS*ACT_WIDTH-1:0] l2_deskewed;
    wire signed [HL3_NEURONS*ACT_WIDTH-1:0] l3_deskewed;

    ShiftRegisterSkew #(
        .NUM_CH(HL1_NEURONS),
        .DATA_WIDTH(ACT_WIDTH),
        .DIRECTION(1)
    ) l1_deskew_inst (
        .clock(clk),
        .reset(reset),
        .io_in(u_dut.u_hidden_layers.hl1_to_hl2),
        .io_out(l1_deskewed)
    );

    ShiftRegisterSkew #(
        .NUM_CH(HL2_NEURONS),
        .DATA_WIDTH(ACT_WIDTH),
        .DIRECTION(1)
    ) l2_deskew_inst (
        .clock(clk),
        .reset(reset),
        .io_in(u_dut.u_hidden_layers.hl2_to_hl3),
        .io_out(l2_deskewed)
    );

    ShiftRegisterSkew #(
        .NUM_CH(HL3_NEURONS),
        .DATA_WIDTH(ACT_WIDTH),
        .DIRECTION(1)
    ) l3_deskew_inst (
        .clock(clk),
        .reset(reset),
        .io_in(u_dut.hl_out),
        .io_out(l3_deskewed)
    );

    // Expose HL1 PE weight registers as a flat bus for runtime weight-load checking.
    wire [HL1_IN_DIM*HL1_NEURONS*WEIGHT_WIDTH-1:0] l1_pe_weight_flat;
    wire [HL1_IN_DIM*HL1_NEURONS*ACT_WIDTH-1:0] l1_pe_act_flat;
    wire [HL1_IN_DIM*HL1_NEURONS*INNER_ACC_WIDTH-1:0] l1_pe_mul_flat;
    wire [HL2_IN_DIM*HL2_NEURONS*WEIGHT_WIDTH-1:0] l2_pe_weight_flat;
    wire [HL2_IN_DIM*HL2_NEURONS*ACT_WIDTH-1:0] l2_pe_act_flat;
    wire [HL2_IN_DIM*HL2_NEURONS*INNER_ACC_WIDTH-1:0] l2_pe_mul_flat;
    wire [HL2_IN_DIM*HL2_NEURONS*INNER_ACC_WIDTH-1:0] l2_pe_psum_flat;
    wire [HL2_IN_DIM*HL2_NEURONS*INNER_ACC_WIDTH-1:0] l2_pe_inwtps_flat;
    wire [HL3_IN_DIM*HL3_NEURONS*WEIGHT_WIDTH-1:0] l3_pe_weight_flat;
    wire [OL_IN_DIM*OL_NEURONS*WEIGHT_WIDTH-1:0] l4_pe_weight_flat;
    genvar wr, wc;
    generate
        for (wr = 0; wr < HL1_IN_DIM; wr = wr + 1) begin : l1_w_row
            for (wc = 0; wc < HL1_NEURONS; wc = wc + 1) begin : l1_w_col
                assign l1_pe_weight_flat[(wr*HL1_NEURONS + wc)*WEIGHT_WIDTH +: WEIGHT_WIDTH] =
                    u_dut.u_hidden_layers.hl1_inst.systolic_array_inst.row_gen[wr].col_gen[wc].pe_inst.weightReg;
                assign l1_pe_act_flat[(wr*HL1_NEURONS + wc)*ACT_WIDTH +: ACT_WIDTH] =
                    u_dut.u_hidden_layers.hl1_inst.systolic_array_inst.row_gen[wr].col_gen[wc].pe_inst.actReg;
                assign l1_pe_mul_flat[(wr*HL1_NEURONS + wc)*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] =
                    u_dut.u_hidden_layers.hl1_inst.systolic_array_inst.row_gen[wr].col_gen[wc].pe_inst.mulReg;
            end
        end
    endgenerate

    genvar wr2, wc2;
    generate
        for (wr2 = 0; wr2 < HL2_IN_DIM; wr2 = wr2 + 1) begin : l2_w_row
            for (wc2 = 0; wc2 < HL2_NEURONS; wc2 = wc2 + 1) begin : l2_w_col
                assign l2_pe_weight_flat[(wr2*HL2_NEURONS + wc2)*WEIGHT_WIDTH +: WEIGHT_WIDTH] =
                    u_dut.u_hidden_layers.hl2_inst.systolic_array_inst.row_gen[wr2].col_gen[wc2].pe_inst.weightReg;
                assign l2_pe_act_flat[(wr2*HL2_NEURONS + wc2)*ACT_WIDTH +: ACT_WIDTH] =
                    u_dut.u_hidden_layers.hl2_inst.systolic_array_inst.row_gen[wr2].col_gen[wc2].pe_inst.actReg;
                assign l2_pe_mul_flat[(wr2*HL2_NEURONS + wc2)*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] =
                    u_dut.u_hidden_layers.hl2_inst.systolic_array_inst.row_gen[wr2].col_gen[wc2].pe_inst.mulReg;
                assign l2_pe_psum_flat[(wr2*HL2_NEURONS + wc2)*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] =
                    u_dut.u_hidden_layers.hl2_inst.systolic_array_inst.row_gen[wr2].col_gen[wc2].pe_inst.parSumReg;
                assign l2_pe_inwtps_flat[(wr2*HL2_NEURONS + wc2)*INNER_ACC_WIDTH +: INNER_ACC_WIDTH] =
                    u_dut.u_hidden_layers.hl2_inst.systolic_array_inst.row_gen[wr2].col_gen[wc2].pe_inst.io_inWtPS;
            end
        end
    endgenerate

    genvar wr3, wc3;
    generate
        for (wr3 = 0; wr3 < HL3_IN_DIM; wr3 = wr3 + 1) begin : l3_w_row
            for (wc3 = 0; wc3 < HL3_NEURONS; wc3 = wc3 + 1) begin : l3_w_col
                assign l3_pe_weight_flat[(wr3*HL3_NEURONS + wc3)*WEIGHT_WIDTH +: WEIGHT_WIDTH] =
                    u_dut.u_hidden_layers.hl3_inst.systolic_array_inst.row_gen[wr3].col_gen[wc3].pe_inst.weightReg;
            end
        end
    endgenerate

    genvar wr4, wc4;
    generate
        for (wr4 = 0; wr4 < OL_IN_DIM; wr4 = wr4 + 1) begin : l4_w_row
            for (wc4 = 0; wc4 < OL_NEURONS; wc4 = wc4 + 1) begin : l4_w_col
                assign l4_pe_weight_flat[(wr4*OL_NEURONS + wc4)*WEIGHT_WIDTH +: WEIGHT_WIDTH] =
                    u_dut.u_output_layer.systolic_array_inst.row_gen[wr4].col_gen[wc4].pe_inst.weightReg;
            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Clocks
    // ------------------------------------------------------------------------
    initial clk = 1'b0;
    always #(CLK_PERIOD/10.0) clk = ~clk;

    initial AXI_clk_i = 1'b0;
    always #(CLK_PERIOD/2.0) AXI_clk_i = ~AXI_clk_i;

    // ------------------------------------------------------------------------
    // Memories
    // ------------------------------------------------------------------------
    reg [127:0] data_in_mem [0:NUM_CYCLES-1];
    reg [31:0]  coeff_mem   [0:COEFF_WORDS-1];

    reg [15:0] pp_gold_mem       [0:PP_NUM_GOLD-1];
    reg [15:0] pp_rk_re_gold_mem [0:RK_NUM_GOLD-1];
    reg [15:0] pp_rk_im_gold_mem [0:RK_NUM_GOLD-1];

    reg [15:0] l1_gold_mem [0:L1_NUM_GOLD-1];
    reg [15:0] l2_gold_mem [0:L2_NUM_GOLD-1];
    reg [15:0] l3_gold_mem [0:L3_NUM_GOLD-1];
    reg [15:0] l4_gold_mem [0:L4_NUM_GOLD-1];

    reg [15:0] pd_golden_i_mem [0:RK_NUM_GOLD-1];
    reg [15:0] pd_golden_q_mem [0:RK_NUM_GOLD-1];

    // ------------------------------------------------------------------------
    // Check helpers and counters
    // ------------------------------------------------------------------------
    integer i, j;
    integer cycle_cnt;
    integer pp_check_idx, pp_rk_check_idx;
    integer pp_seen_idx, pp_rk_seen_idx;
    integer l1_check_idx, l2_check_idx, l3_check_idx, l4_check_idx;
    integer pd_check_idx;

    integer pp_pass, pp_fail, pp_total, pp_max_err, pp_first_fail;
    integer rk_pass, rk_fail, rk_total, rk_max_err_re, rk_max_err_im, rk_first_fail;
    integer l1_pass, l1_fail, l1_total, l1_max_err, l1_first_fail;
    integer l2_pass, l2_fail, l2_total, l2_max_err, l2_first_fail;
    integer l3_pass, l3_fail, l3_total, l3_max_err, l3_first_fail;
    integer l4_pass, l4_fail, l4_total, l4_max_err, l4_first_fail;
    integer pd_pass, pd_fail, pd_total, pd_max_err_i, pd_max_err_q, pd_first_fail;

    reg signed [ACT_WIDTH-1:0] fpga_val;
    reg signed [ACT_WIDTH-1:0] gold_val;
    integer cur_err;
    integer err_re, err_im;
    integer err_i, err_q;
    integer wt_row_idx, wt_col_idx;
    integer wt_param_idx;
    reg [31:0] wt_word;
    reg [15:0] wt_expected;
    reg [15:0] wt_got;
    integer l1_wt_pass, l1_wt_fail, l1_wt_first_fail;
    integer l2_wt_pass, l2_wt_fail, l2_wt_first_fail;
    integer l3_wt_pass, l3_wt_fail, l3_wt_first_fail;
    integer l4_wt_pass, l4_wt_fail, l4_wt_first_fail;

    reg signed [INNER_ACC_WIDTH-1:0] l2_sum_prev [0:HL2_NEURONS-1];
    reg signed [BIAS_WIDTH-1:0]      l2_bias_prev [0:HL2_NEURONS-1];
    reg signed [ACT_WIDTH-1:0]       l2_trunc_prev [0:HL2_NEURONS-1];
    reg signed [ACT_WIDTH-1:0]       l2_trunc_ne_prev [0:HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_insum_prev [0:HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_bshift_prev [0:HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_insum_ne_prev [0:HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_bshift_ne_prev [0:HL2_NEURONS-1];
    reg                               l2_stage_prev_valid;
    reg                               l2_stage_ne_prev_valid;
    reg signed [INNER_ACC_WIDTH-1:0] l2_expected_plus;
    reg signed [INNER_ACC_WIDTH-1:0] l2_plus_now;
    reg signed [ACT_WIDTH-1:0]       l2_expected_act;
    reg signed [ACT_WIDTH-1:0]       l2_act_now;
    integer l2_plus_chk_pass, l2_plus_chk_fail, l2_plus_chk_first_fail;
    integer l2_plus_dir_pass, l2_plus_dir_fail, l2_plus_dir_first_fail;
    integer l2_plus_ne_pass, l2_plus_ne_fail, l2_plus_ne_first_fail;
    integer l2_act_chk_pass,  l2_act_chk_fail,  l2_act_chk_first_fail;

    integer l2_mul_chk_pass,  l2_mul_chk_fail,  l2_mul_chk_first_fail;
    reg                       l2_mul_prev_valid;
    integer l2_psum_chk_pass, l2_psum_chk_fail, l2_psum_chk_first_fail;
    reg                       l2_psum_prev_valid;
    integer l2_psum_ne_pass, l2_psum_ne_fail, l2_psum_ne_first_fail;
    reg                       l2_psum_ne_prev_valid;
    reg signed [WEIGHT_WIDTH-1:0] l2_pe_w_prev [0:HL2_IN_DIM*HL2_NEURONS-1];
    reg signed [ACT_WIDTH-1:0]    l2_pe_a_prev [0:HL2_IN_DIM*HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_pe_inwtps_prev [0:HL2_IN_DIM*HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_pe_mul_prev [0:HL2_IN_DIM*HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_pe_inwtps_ne_prev [0:HL2_IN_DIM*HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_pe_mul_ne_prev [0:HL2_IN_DIM*HL2_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l2_expected_mul;
    reg signed [INNER_ACC_WIDTH-1:0] l2_mul_now;
    reg signed [INNER_ACC_WIDTH-1:0] l2_expected_psum;
    reg signed [INNER_ACC_WIDTH-1:0] l2_psum_now;
    integer l2_pe_idx;

    reg signed [ACT_WIDTH-1:0] l2_raw_delay_ref [0:HL2_NEURONS-1][0:HL2_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l2_raw_now;
    reg signed [ACT_WIDTH-1:0] l2_deskew_now;
    integer l2_skew_chk_pass, l2_skew_chk_fail, l2_skew_chk_first_fail;
    integer l2_skew_ch, l2_skew_d;

    reg signed [INNER_ACC_WIDTH-1:0] l1_sum_prev [0:HL1_NEURONS-1];
    reg signed [BIAS_WIDTH-1:0]      l1_bias_prev [0:HL1_NEURONS-1];
    reg signed [ACT_WIDTH-1:0]       l1_trunc_prev [0:HL1_NEURONS-1];
    reg                               l1_stage_prev_valid;
    reg signed [INNER_ACC_WIDTH-1:0] l1_expected_plus;
    reg signed [INNER_ACC_WIDTH-1:0] l1_plus_now;
    reg signed [ACT_WIDTH-1:0]       l1_expected_act;
    reg signed [ACT_WIDTH-1:0]       l1_act_now;
    integer l1_plus_chk_pass, l1_plus_chk_fail, l1_plus_chk_first_fail;
    integer l1_act_chk_pass,  l1_act_chk_fail,  l1_act_chk_first_fail;
    integer l1_mul_chk_pass,  l1_mul_chk_fail,  l1_mul_chk_first_fail;
    reg                       l1_mul_prev_valid;
    reg signed [WEIGHT_WIDTH-1:0] l1_pe_w_prev [0:HL1_IN_DIM*HL1_NEURONS-1];
    reg signed [ACT_WIDTH-1:0]    l1_pe_a_prev [0:HL1_IN_DIM*HL1_NEURONS-1];
    reg signed [INNER_ACC_WIDTH-1:0] l1_expected_mul;
    reg signed [INNER_ACC_WIDTH-1:0] l1_mul_now;
    integer l1_pe_idx;

    reg signed [ACT_WIDTH-1:0] l1_raw_delay_ref [0:HL1_NEURONS-1][0:HL1_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l1_raw_now;
    reg signed [ACT_WIDTH-1:0] l1_deskew_now;
    integer l1_skew_chk_pass, l1_skew_chk_fail, l1_skew_chk_first_fail;
    integer l1_skew_ch, l1_skew_d;

    // ------------------------------------------------------------------------
    // Real-time waveform comparison signals (for GUI waveform view)
    // ------------------------------------------------------------------------
    reg                     pp_rt_valid;
    reg                     l1_rt_valid;
    reg                     l2_rt_valid;
    reg                     l3_rt_valid;
    reg                     l4_rt_valid;
    reg                     pd_rt_valid;

    reg signed [ACT_WIDTH-1:0] pp_fpga_rt [0:FEAT_DIM-1];
    reg signed [ACT_WIDTH-1:0] pp_gold_rt [0:FEAT_DIM-1];

    reg signed [ACT_WIDTH-1:0] rk_re_fpga_rt [0:N_POLY-1];
    reg signed [ACT_WIDTH-1:0] rk_re_gold_rt [0:N_POLY-1];
    reg signed [ACT_WIDTH-1:0] rk_im_fpga_rt [0:N_POLY-1];
    reg signed [ACT_WIDTH-1:0] rk_im_gold_rt [0:N_POLY-1];

    reg signed [ACT_WIDTH-1:0] l1_fpga_rt [0:HL1_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l1_gold_rt [0:HL1_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l2_fpga_rt [0:HL2_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l2_gold_rt [0:HL2_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l3_fpga_rt [0:HL3_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l3_gold_rt [0:HL3_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l4_fpga_rt [0:OL_NEURONS-1];
    reg signed [ACT_WIDTH-1:0] l4_gold_rt [0:OL_NEURONS-1];

    reg signed [ACT_WIDTH-1:0] pd_i_fpga_rt [0:N_POLY-1];
    reg signed [ACT_WIDTH-1:0] pd_i_gold_rt [0:N_POLY-1];
    reg signed [ACT_WIDTH-1:0] pd_q_fpga_rt [0:N_POLY-1];
    reg signed [ACT_WIDTH-1:0] pd_q_gold_rt [0:N_POLY-1];

    integer pp_rt_idx;
    integer l1_rt_idx;
    integer l2_rt_idx;
    integer l3_rt_idx;
    integer l4_rt_idx;
    integer pd_rt_idx;

    // ------------------------------------------------------------------------
    // Absolute path roots (Vivado xsim friendly)
    // ------------------------------------------------------------------------
    localparam ROOT_DIR  = "G:/Undergraduate Thesis/Undergraduate Thesis_fpga/FPGA_PRVTDNN_by Xu_V7_polyphase_on board/sim_All Top/";
    localparam INPUT_DIR = "G:/Undergraduate Thesis/Undergraduate Thesis_fpga/FPGA_PRVTDNN_by Xu_V7_polyphase_on board/sim_All Top/Input/";
    localparam MODEL_DIR = "G:/Undergraduate Thesis/Undergraduate Thesis_fpga/FPGA_PRVTDNN_by Xu_V7_polyphase_on board/sim_All Top/Model/";
    localparam GOLD_DIR  = "G:/Undergraduate Thesis/Undergraduate Thesis_fpga/FPGA_PRVTDNN_by Xu_V7_polyphase_on board/sim_All Top/Gold/";
    localparam OUT_DIR   = "G:/Undergraduate Thesis/Undergraduate Thesis_fpga/FPGA_PRVTDNN_by Xu_V7_polyphase_on board/sim_All Top/FPGA_Out_Data/";

    // Dump file descriptors
    integer fd_pp_fpga, fd_pp_gold;
    integer fd_l1_fpga, fd_l1_gold;
    integer fd_l1_raw;
    integer fd_l2_fpga, fd_l2_gold;
    integer fd_l2_raw;
    integer fd_l3_fpga, fd_l3_gold;
    integer fd_l3_raw;
    integer fd_l4_fpga, fd_l4_gold;
    integer fd_l4_raw;
    integer fd_pd_fpga, fd_pd_gold;
    integer fd_pp_valid_feat;
    integer fd_l1_wt_diag;
    integer fd_l2_wt_diag;
    integer fd_l3_wt_diag;
    integer fd_l4_wt_diag;
    integer fd_l2_skew_diag;
    integer fd_l2_mul_diag;
    integer fd_l2_psum_diag;
    integer fd_l2_psum_ne_diag;
    integer fd_l2_bias_diag;
    integer fd_l2_bias_dir_diag;
    integer fd_l2_bias_ne_diag;
    integer fd_l2_act_diag;
    integer fd_l1_skew_diag;
    integer fd_l1_mul_diag;
    integer fd_l1_bias_diag;
    integer fd_l1_act_diag;

    // ------------------------------------------------------------------------
    // Main flow
    // ------------------------------------------------------------------------
    initial begin
        // init signals
        reset        = 1'b1;
        data_in      = 128'd0;
        vio_wdpd_i   = 1'b1;
        coeff_i      = 32'd0;
        coeff_addr_i = 16'd0;
        coeff_en_i   = 1'b0;

        // init counters
        pp_check_idx    = 0;
        pp_rk_check_idx = 0;
        pp_seen_idx     = 0;
        pp_rk_seen_idx  = 0;
        l1_check_idx    = 0;
        l2_check_idx    = 0;
        l3_check_idx    = 0;
        l4_check_idx    = 0;
        pd_check_idx    = 0;
        cycle_cnt       = 0;

        pp_pass = 0; pp_fail = 0; pp_total = 0; pp_max_err = 0; pp_first_fail = 0;
        rk_pass = 0; rk_fail = 0; rk_total = 0; rk_max_err_re = 0; rk_max_err_im = 0; rk_first_fail = 0;
        l1_pass = 0; l1_fail = 0; l1_total = 0; l1_max_err = 0; l1_first_fail = 0;
        l2_pass = 0; l2_fail = 0; l2_total = 0; l2_max_err = 0; l2_first_fail = 0;
        l3_pass = 0; l3_fail = 0; l3_total = 0; l3_max_err = 0; l3_first_fail = 0;
        l4_pass = 0; l4_fail = 0; l4_total = 0; l4_max_err = 0; l4_first_fail = 0;
        pd_pass = 0; pd_fail = 0; pd_total = 0; pd_max_err_i = 0; pd_max_err_q = 0; pd_first_fail = 0;
        l1_wt_pass = 0; l1_wt_fail = 0; l1_wt_first_fail = 0;
        l2_wt_pass = 0; l2_wt_fail = 0; l2_wt_first_fail = 0;
        l3_wt_pass = 0; l3_wt_fail = 0; l3_wt_first_fail = 0;
        l4_wt_pass = 0; l4_wt_fail = 0; l4_wt_first_fail = 0;

        l2_plus_chk_pass = 0; l2_plus_chk_fail = 0; l2_plus_chk_first_fail = 0;
        l2_plus_dir_pass = 0; l2_plus_dir_fail = 0; l2_plus_dir_first_fail = 0;
        l2_plus_ne_pass = 0; l2_plus_ne_fail = 0; l2_plus_ne_first_fail = 0;
        l2_act_chk_pass  = 0; l2_act_chk_fail  = 0; l2_act_chk_first_fail  = 0;
        l2_mul_chk_pass  = 0; l2_mul_chk_fail  = 0; l2_mul_chk_first_fail  = 0;
        l2_psum_chk_pass = 0; l2_psum_chk_fail = 0; l2_psum_chk_first_fail = 0;
        l2_psum_ne_pass = 0; l2_psum_ne_fail = 0; l2_psum_ne_first_fail = 0;
        l2_skew_chk_pass = 0; l2_skew_chk_fail = 0; l2_skew_chk_first_fail = 0;
        l2_stage_prev_valid = 1'b0;
        l2_stage_ne_prev_valid = 1'b0;
        l2_mul_prev_valid = 1'b0;
        l2_psum_prev_valid = 1'b0;
        l2_psum_ne_prev_valid = 1'b0;

        l1_plus_chk_pass = 0; l1_plus_chk_fail = 0; l1_plus_chk_first_fail = 0;
        l1_act_chk_pass  = 0; l1_act_chk_fail  = 0; l1_act_chk_first_fail  = 0;
        l1_mul_chk_pass  = 0; l1_mul_chk_fail  = 0; l1_mul_chk_first_fail  = 0;
        l1_skew_chk_pass = 0; l1_skew_chk_fail = 0; l1_skew_chk_first_fail = 0;
        l1_stage_prev_valid = 1'b0;
        l1_mul_prev_valid = 1'b0;

        pp_rt_valid = 1'b0;
        l1_rt_valid = 1'b0;
        l2_rt_valid = 1'b0;
        l3_rt_valid = 1'b0;
        l4_rt_valid = 1'b0;
        pd_rt_valid = 1'b0;

        for (j = 0; j < FEAT_DIM; j = j + 1) begin
            pp_fpga_rt[j] = 0;
            pp_gold_rt[j] = 0;
        end
        for (j = 0; j < N_POLY; j = j + 1) begin
            rk_re_fpga_rt[j] = 0;
            rk_re_gold_rt[j] = 0;
            rk_im_fpga_rt[j] = 0;
            rk_im_gold_rt[j] = 0;
            pd_i_fpga_rt[j] = 0;
            pd_i_gold_rt[j] = 0;
            pd_q_fpga_rt[j] = 0;
            pd_q_gold_rt[j] = 0;
        end
        for (j = 0; j < HL1_NEURONS; j = j + 1) begin
            l1_fpga_rt[j] = 0;
            l1_gold_rt[j] = 0;
            l1_sum_prev[j] = 0;
            l1_bias_prev[j] = 0;
            l1_trunc_prev[j] = 0;
            for (i = 0; i < HL1_NEURONS; i = i + 1)
                l1_raw_delay_ref[j][i] = 0;
        end
        for (i = 0; i < HL1_IN_DIM*HL1_NEURONS; i = i + 1) begin
            l1_pe_w_prev[i] = 0;
            l1_pe_a_prev[i] = 0;
        end
        for (j = 0; j < HL2_NEURONS; j = j + 1) begin
            l2_fpga_rt[j] = 0;
            l2_gold_rt[j] = 0;
            l2_sum_prev[j] = 0;
            l2_bias_prev[j] = 0;
            l2_trunc_prev[j] = 0;
            l2_trunc_ne_prev[j] = 0;
            l2_insum_prev[j] = 0;
            l2_bshift_prev[j] = 0;
            l2_insum_ne_prev[j] = 0;
            l2_bshift_ne_prev[j] = 0;
            for (i = 0; i < HL2_NEURONS; i = i + 1)
                l2_raw_delay_ref[j][i] = 0;
        end
        for (i = 0; i < HL2_IN_DIM*HL2_NEURONS; i = i + 1) begin
            l2_pe_w_prev[i] = 0;
            l2_pe_a_prev[i] = 0;
            l2_pe_inwtps_prev[i] = 0;
            l2_pe_mul_prev[i] = 0;
            l2_pe_inwtps_ne_prev[i] = 0;
            l2_pe_mul_ne_prev[i] = 0;
        end
        for (j = 0; j < HL3_NEURONS; j = j + 1) begin
            l3_fpga_rt[j] = 0;
            l3_gold_rt[j] = 0;
        end
        for (j = 0; j < OL_NEURONS; j = j + 1) begin
            l4_fpga_rt[j] = 0;
            l4_gold_rt[j] = 0;
        end

        // load files
        $readmemh({INPUT_DIR, "data_in.hex"}, data_in_mem);
        $readmemh({MODEL_DIR, "parameter_in.mem"}, coeff_mem);

        $readmemh({GOLD_DIR, "preprocess_feat_golden.hex"}, pp_gold_mem);
        $readmemh({GOLD_DIR, "preprocess_rk_re_golden.hex"}, pp_rk_re_gold_mem);
        $readmemh({GOLD_DIR, "preprocess_rk_im_golden.hex"}, pp_rk_im_gold_mem);

        $readmemh({GOLD_DIR, "layer1_golden.hex"}, l1_gold_mem);
        $readmemh({GOLD_DIR, "layer2_golden.hex"}, l2_gold_mem);
        $readmemh({GOLD_DIR, "layer3_golden.hex"}, l3_gold_mem);
        $readmemh({GOLD_DIR, "layer4_golden.hex"}, l4_gold_mem);

        $readmemh({GOLD_DIR, "phase_denorm_golden_i.hex"}, pd_golden_i_mem);
        $readmemh({GOLD_DIR, "phase_denorm_golden_q.hex"}, pd_golden_q_mem);

        #(CLK_PERIOD * 10);
        reset = 1'b0;
        #(CLK_PERIOD * 5);

        $display("==============================================================");
        $display("PRVTDNN V6 top-level simulation start");
        $display("NUM_CYCLES=%0d, TOL=%0d, COEFF_WORDS=%0d", NUM_CYCLES, TOLERANCE, COEFF_WORDS);
        $display("PP=%0d, L1=%0d, L2=%0d, L3=%0d, L4=%0d, E2E=%0d", PP_LATENCY, L1_LATENCY, L2_LATENCY, L3_LATENCY, L4_LATENCY, E2E_LATENCY);
        $display("==============================================================");

        // -------------------------------------------------------------
        // Open dump files (one line = one valid sample)
        // -------------------------------------------------------------
        fd_pp_fpga = $fopen({OUT_DIR, "pp_fpga_rt.txt"}, "w");
        fd_pp_gold = $fopen({OUT_DIR, "pp_gold_rt.txt"}, "w");
        fd_l1_fpga = $fopen({OUT_DIR, "l1_fpga_rt.txt"}, "w");
        fd_l1_raw  = $fopen({OUT_DIR, "l1_fpga_raw_rt.txt"}, "w");
        fd_l1_gold = $fopen({OUT_DIR, "l1_gold_rt.txt"}, "w");
        fd_l2_fpga = $fopen({OUT_DIR, "l2_fpga_rt.txt"}, "w");
        fd_l2_raw  = $fopen({OUT_DIR, "l2_fpga_raw_rt.txt"}, "w");
        fd_l2_gold = $fopen({OUT_DIR, "l2_gold_rt.txt"}, "w");
        fd_l3_fpga = $fopen({OUT_DIR, "l3_fpga_rt.txt"}, "w");
        fd_l3_raw  = $fopen({OUT_DIR, "l3_fpga_raw_rt.txt"}, "w");
        fd_l3_gold = $fopen({OUT_DIR, "l3_gold_rt.txt"}, "w");
        fd_l4_fpga = $fopen({OUT_DIR, "l4_fpga_rt.txt"}, "w");
        fd_l4_raw  = $fopen({OUT_DIR, "l4_fpga_raw_rt.txt"}, "w");
        fd_l4_gold = $fopen({OUT_DIR, "l4_gold_rt.txt"}, "w");
        fd_pd_fpga = $fopen({OUT_DIR, "pd_fpga_rt.txt"}, "w");
        fd_pd_gold = $fopen({OUT_DIR, "pd_gold_rt.txt"}, "w");
        fd_pp_valid_feat = $fopen({OUT_DIR, "pp_diag_valid_feat.txt"}, "w");
        fd_l1_wt_diag = $fopen({OUT_DIR, "l1_diag_wt_check.txt"}, "w");
        fd_l2_wt_diag = $fopen({OUT_DIR, "l2_diag_wt_check.txt"}, "w");
        fd_l3_wt_diag = $fopen({OUT_DIR, "l3_diag_wt_check.txt"}, "w");
        fd_l4_wt_diag = $fopen({OUT_DIR, "l4_diag_wt_check.txt"}, "w");
        fd_l2_skew_diag = $fopen({OUT_DIR, "l2_diag_skew_check.txt"}, "w");
        fd_l2_mul_diag = $fopen({OUT_DIR, "l2_diag_mul_check.txt"}, "w");
        fd_l2_psum_diag = $fopen({OUT_DIR, "l2_diag_psum_check.txt"}, "w");
        fd_l2_psum_ne_diag = $fopen({OUT_DIR, "l2_diag_psum_negedge_check.txt"}, "w");
        fd_l2_bias_diag = $fopen({OUT_DIR, "l2_diag_bias_check.txt"}, "w");
        fd_l2_bias_dir_diag = $fopen({OUT_DIR, "l2_diag_bias_direct_check.txt"}, "w");
        fd_l2_bias_ne_diag = $fopen({OUT_DIR, "l2_diag_bias_negedge_check.txt"}, "w");
        fd_l2_act_diag = $fopen({OUT_DIR, "l2_diag_act_check.txt"}, "w");
        fd_l1_skew_diag = $fopen({OUT_DIR, "l1_diag_skew_check.txt"}, "w");
        fd_l1_mul_diag = $fopen({OUT_DIR, "l1_diag_mul_check.txt"}, "w");
        fd_l1_bias_diag = $fopen({OUT_DIR, "l1_diag_bias_check.txt"}, "w");
        fd_l1_act_diag = $fopen({OUT_DIR, "l1_diag_act_check.txt"}, "w");

        $fwrite(fd_pp_valid_feat, "cycle,pp_check_idx,feat0..feat44_hex\n");
        $fwrite(fd_l1_wt_diag, "row,col,got_hex,exp_hex,match\n");
        $fwrite(fd_l2_wt_diag, "row,col,got_hex,exp_hex,match\n");
        $fwrite(fd_l3_wt_diag, "row,col,got_hex,exp_hex,match\n");
        $fwrite(fd_l4_wt_diag, "row,col,got_hex,exp_hex,match\n");
        $fwrite(fd_l2_skew_diag, "cycle,ch,deskew,exp_deskew,raw,match\n");
        $fwrite(fd_l2_mul_diag, "cycle,row,col,mul,exp_mul,weight_prev,act_prev,match\n");
        $fwrite(fd_l2_psum_diag, "cycle,row,col,psum,exp_psum,inwtps_prev,mul_prev,match\n");
        $fwrite(fd_l2_psum_ne_diag, "cycle,row,col,psum,exp_psum,inwtps_ne_prev,mul_ne_prev,match\n");
        $fwrite(fd_l2_bias_diag, "cycle,ch,plus,exp_plus,sum_prev,bias_prev,match\n");
        $fwrite(fd_l2_bias_dir_diag, "cycle,ch,plus,exp_plus,insum_prev,bias_shift_prev,match\n");
        $fwrite(fd_l2_bias_ne_diag, "cycle,ch,plus,exp_plus,insum_ne_prev,bias_shift_ne_prev,match\n");
        $fwrite(fd_l2_act_diag, "cycle,ch,act,exp_act,trunc_prev,match\n");
        $fwrite(fd_l1_skew_diag, "cycle,ch,deskew,exp_deskew,raw,match\n");
        $fwrite(fd_l1_mul_diag, "cycle,row,col,mul,exp_mul,weight_prev,act_prev,match\n");
        $fwrite(fd_l1_bias_diag, "cycle,ch,plus,exp_plus,sum_prev,bias_prev,match\n");
        $fwrite(fd_l1_act_diag, "cycle,ch,act,exp_act,trunc_prev,match\n");

        if (fd_pp_fpga == 0 || fd_pp_gold == 0 ||
            fd_l1_fpga == 0 || fd_l1_raw == 0 || fd_l1_gold == 0 ||
            fd_l2_fpga == 0 || fd_l2_raw == 0 || fd_l2_gold == 0 ||
            fd_l3_fpga == 0 || fd_l3_raw == 0 || fd_l3_gold == 0 ||
            fd_l4_fpga == 0 || fd_l4_raw == 0 || fd_l4_gold == 0 ||
            fd_pd_fpga == 0 || fd_pd_gold == 0 ||
            fd_pp_valid_feat == 0 ||
            fd_l1_wt_diag == 0 || fd_l2_wt_diag == 0 || fd_l3_wt_diag == 0 || fd_l4_wt_diag == 0 ||
            fd_l2_skew_diag == 0 || fd_l2_mul_diag == 0 || fd_l2_psum_diag == 0 || fd_l2_psum_ne_diag == 0 || fd_l2_bias_diag == 0 || fd_l2_bias_dir_diag == 0 || fd_l2_bias_ne_diag == 0 || fd_l2_act_diag == 0 ||
            fd_l1_skew_diag == 0 ||
            fd_l1_mul_diag == 0 || fd_l1_bias_diag == 0 || fd_l1_act_diag == 0) begin
            $display("[TB] ERROR: failed to open one or more dump files in FPGA_Out_Data");
            $finish;
        end

        // -------------------------------------------------------------
        // Phase 1: load parameters through AXI interface
        // -------------------------------------------------------------
        $display("[%0t] Phase 1: AXI parameter loading start", $time);

        @(negedge AXI_clk_i);
        coeff_en_i <= 1'b1;
        for (i = 0; i < COEFF_WORDS; i = i + 1) begin
            coeff_addr_i <= i[15:0];
            coeff_i      <= coeff_mem[i];
            @(negedge AXI_clk_i);
        end
        coeff_en_i   <= 1'b0;
        coeff_addr_i <= 16'd0;
        coeff_i      <= 32'd0;

        // wait internal load_done pulse (generated in clk domain)
        // Sample with clk to avoid missing narrow pulse when clk is faster than AXI_clk_i.
        begin : wait_load_done_blk
            integer load_wait_cycles;
            load_wait_cycles = 0;
            forever begin
                @(posedge clk);
                if (u_dut.load_done) disable wait_load_done_blk;
                load_wait_cycles = load_wait_cycles + 1;
                if (load_wait_cycles > 20000) begin
                    $display("[TB] ERROR: timeout waiting load_done");
                    $finish;
                end
            end
        end

        $display("[%0t] Phase 1 done: load_done observed", $time);

        // -------------------------------------------------------------
        // L1 weight-load correctness check (PE weightReg snapshot)
        // -------------------------------------------------------------
        for (wt_row_idx = 0; wt_row_idx < HL1_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
            for (wt_col_idx = 0; wt_col_idx < HL1_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                wt_param_idx = HL1_WT_OFF + wt_col_idx * HL1_IN_DIM + wt_row_idx;
                wt_word = coeff_mem[wt_param_idx >> 1];
                if (wt_param_idx[0])
                    wt_expected = wt_word[31:16];
                else
                    wt_expected = wt_word[15:0];

                wt_got = l1_pe_weight_flat[(wt_row_idx*HL1_NEURONS + wt_col_idx)*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                if (wt_got === wt_expected) begin
                    l1_wt_pass = l1_wt_pass + 1;
                    $fwrite(fd_l1_wt_diag, "%0d,%0d,%04h,%04h,1\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                end else begin
                    l1_wt_fail = l1_wt_fail + 1;
                    $fwrite(fd_l1_wt_diag, "%0d,%0d,%04h,%04h,0\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                    if (l1_wt_first_fail < 10) begin
                        $display("[L1_WT][r=%0d c=%0d] FAIL: got=%04h exp=%04h", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                        l1_wt_first_fail = l1_wt_first_fail + 1;
                    end
                end
            end
        end
        $display("[L1_WT] pass=%0d fail=%0d total=%0d", l1_wt_pass, l1_wt_fail, HL1_IN_DIM*HL1_NEURONS);

        for (wt_row_idx = 0; wt_row_idx < HL2_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
            for (wt_col_idx = 0; wt_col_idx < HL2_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                wt_param_idx = HL2_WT_OFF + wt_col_idx * HL2_IN_DIM + wt_row_idx;
                wt_word = coeff_mem[wt_param_idx >> 1];
                if (wt_param_idx[0]) wt_expected = wt_word[31:16];
                else                 wt_expected = wt_word[15:0];

                wt_got = l2_pe_weight_flat[(wt_row_idx*HL2_NEURONS + wt_col_idx)*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                if (wt_got === wt_expected) begin
                    l2_wt_pass = l2_wt_pass + 1;
                    $fwrite(fd_l2_wt_diag, "%0d,%0d,%04h,%04h,1\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                end else begin
                    l2_wt_fail = l2_wt_fail + 1;
                    $fwrite(fd_l2_wt_diag, "%0d,%0d,%04h,%04h,0\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                    if (l2_wt_first_fail < 10) begin
                        $display("[L2_WT][r=%0d c=%0d] FAIL: got=%04h exp=%04h", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                        l2_wt_first_fail = l2_wt_first_fail + 1;
                    end
                end
            end
        end
        $display("[L2_WT] pass=%0d fail=%0d total=%0d", l2_wt_pass, l2_wt_fail, HL2_IN_DIM*HL2_NEURONS);

        for (wt_row_idx = 0; wt_row_idx < HL3_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
            for (wt_col_idx = 0; wt_col_idx < HL3_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                wt_param_idx = HL3_WT_OFF + wt_col_idx * HL3_IN_DIM + wt_row_idx;
                wt_word = coeff_mem[wt_param_idx >> 1];
                if (wt_param_idx[0]) wt_expected = wt_word[31:16];
                else                 wt_expected = wt_word[15:0];

                wt_got = l3_pe_weight_flat[(wt_row_idx*HL3_NEURONS + wt_col_idx)*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                if (wt_got === wt_expected) begin
                    l3_wt_pass = l3_wt_pass + 1;
                    $fwrite(fd_l3_wt_diag, "%0d,%0d,%04h,%04h,1\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                end else begin
                    l3_wt_fail = l3_wt_fail + 1;
                    $fwrite(fd_l3_wt_diag, "%0d,%0d,%04h,%04h,0\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                    if (l3_wt_first_fail < 10) begin
                        $display("[L3_WT][r=%0d c=%0d] FAIL: got=%04h exp=%04h", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                        l3_wt_first_fail = l3_wt_first_fail + 1;
                    end
                end
            end
        end
        $display("[L3_WT] pass=%0d fail=%0d total=%0d", l3_wt_pass, l3_wt_fail, HL3_IN_DIM*HL3_NEURONS);

        for (wt_row_idx = 0; wt_row_idx < OL_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
            for (wt_col_idx = 0; wt_col_idx < OL_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                wt_param_idx = OL_WT_OFF + wt_col_idx * OL_IN_DIM + wt_row_idx;
                wt_word = coeff_mem[wt_param_idx >> 1];
                if (wt_param_idx[0]) wt_expected = wt_word[31:16];
                else                 wt_expected = wt_word[15:0];

                wt_got = l4_pe_weight_flat[(wt_row_idx*OL_NEURONS + wt_col_idx)*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                if (wt_got === wt_expected) begin
                    l4_wt_pass = l4_wt_pass + 1;
                    $fwrite(fd_l4_wt_diag, "%0d,%0d,%04h,%04h,1\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                end else begin
                    l4_wt_fail = l4_wt_fail + 1;
                    $fwrite(fd_l4_wt_diag, "%0d,%0d,%04h,%04h,0\n", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                    if (l4_wt_first_fail < 10) begin
                        $display("[L4_WT][r=%0d c=%0d] FAIL: got=%04h exp=%04h", wt_row_idx, wt_col_idx, wt_got, wt_expected);
                        l4_wt_first_fail = l4_wt_first_fail + 1;
                    end
                end
            end
        end
        $display("[L4_WT] pass=%0d fail=%0d total=%0d", l4_wt_pass, l4_wt_fail, OL_IN_DIM*OL_NEURONS);

        repeat (5) @(posedge clk);

        // -------------------------------------------------------------
        // Phase 2: inference + checking
        // -------------------------------------------------------------
        $display("[%0t] Phase 2: start inference", $time);

        for (cycle_cnt = 0; cycle_cnt < TOTAL_CYCLES; cycle_cnt = cycle_cnt + 1) begin

            // drive input
            @(negedge clk);
            if (cycle_cnt < NUM_CYCLES)
                data_in <= data_in_mem[cycle_cnt];
            else
                data_in <= 128'd0;

            // check outputs
            @(posedge clk);
            #1;

            // ---------------------------------------------------------
            // Real-time waveform mirrors: FPGA outputs and aligned golden
            // ---------------------------------------------------------
            // Keep waveform RT indices based on stage latency only.
            // Optional PP_INIT_SKIP is handled in PP/RK checker path, not here.
            // PP mirrors follow checker indexing (valid-based) rather than fixed latency model.
            pp_rt_idx = pp_check_idx;
            l1_rt_idx = cycle_cnt - L1_RT_LATENCY;
            l2_rt_idx = cycle_cnt - L2_RT_LATENCY;
            l3_rt_idx = cycle_cnt - L3_RT_LATENCY;
            l4_rt_idx = cycle_cnt - L4_RT_LATENCY;
            pd_rt_idx = cycle_cnt - PD_RT_LATENCY;

            pp_rt_valid = u_dut.pp_out_valid && (pp_seen_idx >= PP_INIT_SKIP) && (pp_rt_idx < NUM_CYCLES);
            l1_rt_valid = (l1_rt_idx >= 0) && (l1_rt_idx < NUM_CYCLES);
            l2_rt_valid = (l2_rt_idx >= 0) && (l2_rt_idx < NUM_CYCLES);
            l3_rt_valid = (l3_rt_idx >= 0) && (l3_rt_idx < NUM_CYCLES);
            l4_rt_valid = (l4_rt_idx >= 0) && (l4_rt_idx < NUM_CYCLES);
            pd_rt_valid = (pd_rt_idx >= 0) && (pd_rt_idx < NUM_CYCLES);

            for (j = 0; j < FEAT_DIM; j = j + 1) begin
                pp_fpga_rt[j] = $signed(u_dut.pp_feat_flat[j*ACT_WIDTH +: ACT_WIDTH]);
                if (pp_rt_valid)
                    pp_gold_rt[j] = $signed(pp_gold_mem[pp_rt_idx * FEAT_DIM + j]);
                else
                    pp_gold_rt[j] = 0;
            end

            for (j = 0; j < N_POLY; j = j + 1) begin
                rk_re_fpga_rt[j] = $signed(u_dut.pp_rk_re_flat[j*ACT_WIDTH +: ACT_WIDTH]);
                rk_im_fpga_rt[j] = $signed(u_dut.pp_rk_im_flat[j*ACT_WIDTH +: ACT_WIDTH]);
                if (pp_rt_valid) begin
                    rk_re_gold_rt[j] = $signed(pp_rk_re_gold_mem[pp_rt_idx * N_POLY + j]);
                    rk_im_gold_rt[j] = $signed(pp_rk_im_gold_mem[pp_rt_idx * N_POLY + j]);
                end else begin
                    rk_re_gold_rt[j] = 0;
                    rk_im_gold_rt[j] = 0;
                end
            end

            for (j = 0; j < HL1_NEURONS; j = j + 1) begin
                l1_fpga_rt[j] = $signed(l1_deskewed[j*ACT_WIDTH +: ACT_WIDTH]);
                if (l1_rt_valid)
                    l1_gold_rt[j] = $signed(l1_gold_mem[l1_rt_idx * HL1_NEURONS + j]);
                else
                    l1_gold_rt[j] = 0;
            end

            for (j = 0; j < HL2_NEURONS; j = j + 1) begin
                if (l2_rt_valid) begin
                    l2_fpga_rt[j] = $signed(l2_deskewed[j*ACT_WIDTH +: ACT_WIDTH]);
                    l2_gold_rt[j] = $signed(l2_gold_mem[l2_rt_idx * HL2_NEURONS + j]);
                end else begin
                    l2_fpga_rt[j] = 0;
                    l2_gold_rt[j] = 0;
                end
            end

            for (j = 0; j < HL3_NEURONS; j = j + 1) begin
                l3_fpga_rt[j] = $signed(l3_deskewed[j*ACT_WIDTH +: ACT_WIDTH]);
                if (l3_rt_valid)
                    l3_gold_rt[j] = $signed(l3_gold_mem[l3_rt_idx * HL3_NEURONS + j]);
                else
                    l3_gold_rt[j] = 0;
            end

            for (j = 0; j < OL_NEURONS; j = j + 1) begin
                l4_fpga_rt[j] = $signed(u_dut.ol_out[j*ACT_WIDTH +: ACT_WIDTH]);
                if (l4_rt_valid)
                    l4_gold_rt[j] = $signed(l4_gold_mem[l4_rt_idx * OL_NEURONS + j]);
                else
                    l4_gold_rt[j] = 0;
            end

            for (j = 0; j < N_POLY; j = j + 1) begin
                pd_i_fpga_rt[j] = $signed(y_i[j*ACT_WIDTH +: ACT_WIDTH]);
                pd_q_fpga_rt[j] = $signed(y_q[j*ACT_WIDTH +: ACT_WIDTH]);
                if (pd_rt_valid) begin
                    pd_i_gold_rt[j] = $signed(pd_golden_i_mem[pd_rt_idx * N_POLY + j]);
                    pd_q_gold_rt[j] = $signed(pd_golden_q_mem[pd_rt_idx * N_POLY + j]);
                end else begin
                    pd_i_gold_rt[j] = 0;
                    pd_q_gold_rt[j] = 0;
                end
            end

            // ---------------------------------------------------------
            // Dump real-time vectors for offline comparison
            // ---------------------------------------------------------
            if (pp_rt_valid) begin
                for (j = 0; j < FEAT_DIM; j = j + 1) begin
                    $fwrite(fd_pp_fpga, "%04h", $unsigned(pp_fpga_rt[j]));
                    $fwrite(fd_pp_gold, "%04h", $unsigned(pp_gold_rt[j]));
                    if (j != FEAT_DIM-1) begin
                        $fwrite(fd_pp_fpga, ",");
                        $fwrite(fd_pp_gold, ",");
                    end
                end
                $fwrite(fd_pp_fpga, "\n");
                $fwrite(fd_pp_gold, "\n");
            end

            if (l1_rt_valid) begin
                for (j = 0; j < HL1_NEURONS; j = j + 1) begin
                    $fwrite(fd_l1_fpga, "%04h", $unsigned(l1_fpga_rt[j]));
                    $fwrite(fd_l1_raw,  "%04h", $unsigned(u_dut.u_hidden_layers.hl1_to_hl2[j*ACT_WIDTH +: ACT_WIDTH]));
                    $fwrite(fd_l1_gold, "%04h", $unsigned(l1_gold_rt[j]));
                    if (j != HL1_NEURONS-1) begin
                        $fwrite(fd_l1_fpga, ",");
                        $fwrite(fd_l1_raw,  ",");
                        $fwrite(fd_l1_gold, ",");
                    end
                end
                $fwrite(fd_l1_fpga, "\n");
                $fwrite(fd_l1_raw,  "\n");
                $fwrite(fd_l1_gold, "\n");
            end

            if (l2_rt_valid) begin
                for (j = 0; j < HL2_NEURONS; j = j + 1) begin
                    $fwrite(fd_l2_fpga, "%04h", $unsigned(l2_fpga_rt[j]));
                    $fwrite(fd_l2_raw,  "%04h", $unsigned(u_dut.u_hidden_layers.hl2_to_hl3[j*ACT_WIDTH +: ACT_WIDTH]));
                    $fwrite(fd_l2_gold, "%04h", $unsigned(l2_gold_rt[j]));
                    if (j != HL2_NEURONS-1) begin
                        $fwrite(fd_l2_fpga, ",");
                        $fwrite(fd_l2_raw,  ",");
                        $fwrite(fd_l2_gold, ",");
                    end
                end
                $fwrite(fd_l2_fpga, "\n");
                $fwrite(fd_l2_raw,  "\n");
                $fwrite(fd_l2_gold, "\n");
            end

            if (l3_rt_valid) begin
                for (j = 0; j < HL3_NEURONS; j = j + 1) begin
                    $fwrite(fd_l3_fpga, "%04h", $unsigned(l3_fpga_rt[j]));
                    $fwrite(fd_l3_raw,  "%04h", $unsigned(u_dut.hl_out[j*ACT_WIDTH +: ACT_WIDTH]));
                    $fwrite(fd_l3_gold, "%04h", $unsigned(l3_gold_rt[j]));
                    if (j != HL3_NEURONS-1) begin
                        $fwrite(fd_l3_fpga, ",");
                        $fwrite(fd_l3_raw,  ",");
                        $fwrite(fd_l3_gold, ",");
                    end
                end
                $fwrite(fd_l3_fpga, "\n");
                $fwrite(fd_l3_raw,  "\n");
                $fwrite(fd_l3_gold, "\n");
            end

            if (l4_rt_valid) begin
                for (j = 0; j < OL_NEURONS; j = j + 1) begin
                    $fwrite(fd_l4_fpga, "%04h", $unsigned(l4_fpga_rt[j]));
                    $fwrite(fd_l4_raw,  "%04h", $unsigned(u_dut.u_output_layer.raw_outAct[j*ACT_WIDTH +: ACT_WIDTH]));
                    $fwrite(fd_l4_gold, "%04h", $unsigned(l4_gold_rt[j]));
                    if (j != OL_NEURONS-1) begin
                        $fwrite(fd_l4_fpga, ",");
                        $fwrite(fd_l4_raw,  ",");
                        $fwrite(fd_l4_gold, ",");
                    end
                end
                $fwrite(fd_l4_fpga, "\n");
                $fwrite(fd_l4_raw,  "\n");
                $fwrite(fd_l4_gold, "\n");
            end

            if (pd_rt_valid) begin
                for (j = 0; j < N_POLY; j = j + 1) begin
                    $fwrite(fd_pd_fpga, "%04h", $unsigned(pd_i_fpga_rt[j]));
                    $fwrite(fd_pd_gold, "%04h", $unsigned(pd_i_gold_rt[j]));
                    $fwrite(fd_pd_fpga, ",%04h", $unsigned(pd_q_fpga_rt[j]));
                    $fwrite(fd_pd_gold, ",%04h", $unsigned(pd_q_gold_rt[j]));
                    if (j != N_POLY-1) begin
                        $fwrite(fd_pd_fpga, ",");
                        $fwrite(fd_pd_gold, ",");
                    end
                end
                $fwrite(fd_pd_fpga, "\n");
                $fwrite(fd_pd_gold, "\n");
            end

            // preprocess feature check
            if (u_dut.pp_out_valid) begin
                if (pp_seen_idx >= PP_INIT_SKIP && pp_check_idx < NUM_CYCLES) begin
                    $fwrite(fd_pp_valid_feat, "%0d,%0d", cycle_cnt, pp_check_idx);
                    for (j = 0; j < FEAT_DIM; j = j + 1)
                        $fwrite(fd_pp_valid_feat, ",%04h", $unsigned(u_dut.pp_feat_flat[j*ACT_WIDTH +: ACT_WIDTH]));
                    $fwrite(fd_pp_valid_feat, "\n");

                    for (j = 0; j < FEAT_DIM; j = j + 1) begin
                        fpga_val = $signed(u_dut.pp_feat_flat[j*ACT_WIDTH +: ACT_WIDTH]);
                        gold_val = $signed(pp_gold_mem[pp_check_idx * FEAT_DIM + j]);
                        cur_err = fpga_val - gold_val;
                        if (cur_err < 0) cur_err = -cur_err;

                        pp_total = pp_total + 1;
                        if (cur_err > pp_max_err) pp_max_err = cur_err;
                        if (cur_err <= TOLERANCE) begin
                            pp_pass = pp_pass + 1;
                        end else begin
                            pp_fail = pp_fail + 1;
                            if (pp_first_fail < 10) begin
                                $display("[PP][%0d][%0d] FAIL: fpga=%0d gold=%0d err=%0d", pp_check_idx, j, fpga_val, gold_val, cur_err);
                                pp_first_fail = pp_first_fail + 1;
                            end
                        end
                    end
                    pp_check_idx = pp_check_idx + 1;
                end
                pp_seen_idx = pp_seen_idx + 1;
            end

            // preprocess rk check
            if (u_dut.pp_out_valid) begin
                if (pp_rk_seen_idx >= PP_INIT_SKIP && pp_rk_check_idx < NUM_CYCLES) begin
                    for (j = 0; j < N_POLY; j = j + 1) begin
                        err_re = $signed(u_dut.pp_rk_re_flat[j*ACT_WIDTH +: ACT_WIDTH]) - $signed(pp_rk_re_gold_mem[pp_rk_check_idx * N_POLY + j]);
                        err_im = $signed(u_dut.pp_rk_im_flat[j*ACT_WIDTH +: ACT_WIDTH]) - $signed(pp_rk_im_gold_mem[pp_rk_check_idx * N_POLY + j]);
                        if (err_re < 0) err_re = -err_re;
                        if (err_im < 0) err_im = -err_im;

                        rk_total = rk_total + 1;
                        if (err_re > rk_max_err_re) rk_max_err_re = err_re;
                        if (err_im > rk_max_err_im) rk_max_err_im = err_im;

                        if (err_re <= TOLERANCE && err_im <= TOLERANCE) begin
                            rk_pass = rk_pass + 1;
                        end else begin
                            rk_fail = rk_fail + 1;
                            if (rk_first_fail < 10) begin
                                $display("[RK][%0d][%0d] FAIL: err_re=%0d err_im=%0d", pp_rk_check_idx, j, err_re, err_im);
                                rk_first_fail = rk_first_fail + 1;
                            end
                        end
                    end
                    pp_rk_check_idx = pp_rk_check_idx + 1;
                end
                pp_rk_seen_idx = pp_rk_seen_idx + 1;
            end

            // layer1 check
            if (l1_rt_valid && l1_check_idx < NUM_CYCLES) begin
                for (j = 0; j < HL1_NEURONS; j = j + 1) begin
                    fpga_val = $signed(l1_deskewed[j*ACT_WIDTH +: ACT_WIDTH]);
                    gold_val = $signed(l1_gold_mem[l1_check_idx * HL1_NEURONS + j]);
                    cur_err = fpga_val - gold_val;
                    if (cur_err < 0) cur_err = -cur_err;

                    l1_total = l1_total + 1;
                    if (cur_err > l1_max_err) l1_max_err = cur_err;
                    if (cur_err <= TOLERANCE) begin
                        l1_pass = l1_pass + 1;
                    end else begin
                        l1_fail = l1_fail + 1;
                        if (l1_first_fail < 10) begin
                            $display("[L1][%0d][%0d] FAIL: fpga=%0d gold=%0d err=%0d", l1_check_idx, j, fpga_val, gold_val, cur_err);
                            l1_first_fail = l1_first_fail + 1;
                        end
                    end
                end
                l1_check_idx = l1_check_idx + 1;
            end

            // L1/L2 PE internal diagnostics (mul/add/bias/act/skew) are optional.
            if (ENABLE_INTERNAL_PE_DIAG) begin
            // L1 stage-logic checks: bias add and activation mapping.
            // plusReg/actReg are registered in BiasActivation, so compare with previous-cycle references.
            if (l1_stage_prev_valid) begin
                for (j = 0; j < HL1_NEURONS; j = j + 1) begin
                    l1_expected_plus = l1_sum_prev[j] +
                        ($signed({{(INNER_ACC_WIDTH-BIAS_WIDTH){l1_bias_prev[j][BIAS_WIDTH-1]}}, l1_bias_prev[j]}) <<< L1_BIAS_SHIFT);
                    l1_plus_now = $signed(u_dut.u_hidden_layers.hl1_inst.bias_act_inst.plusReg[j]);
                    if (l1_plus_now === l1_expected_plus) begin
                        l1_plus_chk_pass = l1_plus_chk_pass + 1;
                        $fwrite(fd_l1_bias_diag, "%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                            cycle_cnt, j, l1_plus_now, l1_expected_plus, l1_sum_prev[j], l1_bias_prev[j]);
                    end else begin
                        l1_plus_chk_fail = l1_plus_chk_fail + 1;
                        $fwrite(fd_l1_bias_diag, "%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                            cycle_cnt, j, l1_plus_now, l1_expected_plus, l1_sum_prev[j], l1_bias_prev[j]);
                        if (l1_plus_chk_first_fail < 10) begin
                            $display("[L1_BIAS][%0d] FAIL: plus=%0d exp=%0d", j, l1_plus_now, l1_expected_plus);
                            l1_plus_chk_first_fail = l1_plus_chk_first_fail + 1;
                        end
                    end

                    l1_expected_act = l1_trunc_prev[j][ACT_WIDTH-1] ?
                        (l1_trunc_prev[j] >>> 3) : l1_trunc_prev[j];
                    l1_act_now = $signed(u_dut.u_hidden_layers.hl1_inst.bias_act_inst.actReg[j]);
                    if (l1_act_now === l1_expected_act) begin
                        l1_act_chk_pass = l1_act_chk_pass + 1;
                        $fwrite(fd_l1_act_diag, "%0d,%0d,%0d,%0d,%0d,1\n",
                            cycle_cnt, j, l1_act_now, l1_expected_act, l1_trunc_prev[j]);
                    end else begin
                        l1_act_chk_fail = l1_act_chk_fail + 1;
                        $fwrite(fd_l1_act_diag, "%0d,%0d,%0d,%0d,%0d,0\n",
                            cycle_cnt, j, l1_act_now, l1_expected_act, l1_trunc_prev[j]);
                        if (l1_act_chk_first_fail < 10) begin
                            $display("[L1_ACT][%0d] FAIL: act=%0d exp=%0d", j, l1_act_now, l1_expected_act);
                            l1_act_chk_first_fail = l1_act_chk_first_fail + 1;
                        end
                    end
                end
            end

            // L1 multiply-stage check across all PEs.
            // For systolic columns c>0, mulReg aligns with the left-neighbor act snapshot
            // in this TB sampling scheme (wavefront propagation + register timing).
            if (l1_mul_prev_valid) begin
                for (wt_row_idx = 0; wt_row_idx < HL1_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
                    for (wt_col_idx = 0; wt_col_idx < HL1_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                        l1_pe_idx = wt_row_idx*HL1_NEURONS + wt_col_idx;
                        if (wt_col_idx == 0)
                            l1_expected_mul = $signed(l1_pe_w_prev[l1_pe_idx]) * $signed(l1_pe_a_prev[l1_pe_idx]);
                        else
                            l1_expected_mul = $signed(l1_pe_w_prev[l1_pe_idx]) * $signed(l1_pe_a_prev[l1_pe_idx-1]);
                        l1_mul_now = $signed(l1_pe_mul_flat[l1_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                        if (l1_mul_now === l1_expected_mul) begin
                            l1_mul_chk_pass = l1_mul_chk_pass + 1;
                            $fwrite(fd_l1_mul_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l1_mul_now, l1_expected_mul,
                                l1_pe_w_prev[l1_pe_idx], l1_pe_a_prev[l1_pe_idx]);
                        end else begin
                            l1_mul_chk_fail = l1_mul_chk_fail + 1;
                            $fwrite(fd_l1_mul_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l1_mul_now, l1_expected_mul,
                                l1_pe_w_prev[l1_pe_idx], l1_pe_a_prev[l1_pe_idx]);
                            if (l1_mul_chk_first_fail < 10) begin
                                $display("[L1_MUL][r=%0d c=%0d] FAIL: mul=%0d exp=%0d", wt_row_idx, wt_col_idx, l1_mul_now, l1_expected_mul);
                                l1_mul_chk_first_fail = l1_mul_chk_first_fail + 1;
                            end
                        end
                    end
                end
            end

            // L1 staircase de-skew check with TB reference shift model.
            for (l1_skew_ch = 0; l1_skew_ch < HL1_NEURONS; l1_skew_ch = l1_skew_ch + 1) begin
                l1_raw_now = $signed(u_dut.u_hidden_layers.hl1_to_hl2[l1_skew_ch*ACT_WIDTH +: ACT_WIDTH]);

                for (l1_skew_d = HL1_NEURONS-1; l1_skew_d > 0; l1_skew_d = l1_skew_d - 1)
                    l1_raw_delay_ref[l1_skew_ch][l1_skew_d] = l1_raw_delay_ref[l1_skew_ch][l1_skew_d-1];
                l1_raw_delay_ref[l1_skew_ch][0] = l1_raw_now;

                l1_deskew_now = $signed(l1_deskewed[l1_skew_ch*ACT_WIDTH +: ACT_WIDTH]);
                if (l1_deskew_now === l1_raw_delay_ref[l1_skew_ch][HL1_NEURONS-1-l1_skew_ch]) begin
                    l1_skew_chk_pass = l1_skew_chk_pass + 1;
                    $fwrite(fd_l1_skew_diag, "%0d,%0d,%0d,%0d,%0d,1\n",
                        cycle_cnt, l1_skew_ch, l1_deskew_now,
                        l1_raw_delay_ref[l1_skew_ch][HL1_NEURONS-1-l1_skew_ch], l1_raw_now);
                end else begin
                    l1_skew_chk_fail = l1_skew_chk_fail + 1;
                    $fwrite(fd_l1_skew_diag, "%0d,%0d,%0d,%0d,%0d,0\n",
                        cycle_cnt, l1_skew_ch, l1_deskew_now,
                        l1_raw_delay_ref[l1_skew_ch][HL1_NEURONS-1-l1_skew_ch], l1_raw_now);
                    if (l1_skew_chk_first_fail < 10) begin
                        $display("[L1_SKEW][ch=%0d] FAIL: deskew=%0d exp=%0d", l1_skew_ch, l1_deskew_now, l1_raw_delay_ref[l1_skew_ch][HL1_NEURONS-1-l1_skew_ch]);
                        l1_skew_chk_first_fail = l1_skew_chk_first_fail + 1;
                    end
                end
            end

            // Update previous-cycle references for next-stage checks.
            for (j = 0; j < HL1_NEURONS; j = j + 1) begin
                l1_sum_prev[j] = $signed(u_dut.u_hidden_layers.hl1_inst.array_out_sum[j*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                l1_bias_prev[j] = $signed(u_dut.u_hidden_layers.hl1_inst.bias_act_inst.biasReg[j]);
                l1_trunc_prev[j] = $signed(u_dut.u_hidden_layers.hl1_inst.bias_act_inst.truncWire[j]);
            end
            l1_stage_prev_valid = 1'b1;

            for (wt_row_idx = 0; wt_row_idx < HL1_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
                for (wt_col_idx = 0; wt_col_idx < HL1_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                    l1_pe_idx = wt_row_idx*HL1_NEURONS + wt_col_idx;
                    l1_pe_w_prev[l1_pe_idx] = $signed(l1_pe_weight_flat[l1_pe_idx*WEIGHT_WIDTH +: WEIGHT_WIDTH]);
                    l1_pe_a_prev[l1_pe_idx] = $signed(l1_pe_act_flat[l1_pe_idx*ACT_WIDTH +: ACT_WIDTH]);
                end
            end
            l1_mul_prev_valid = 1'b1;

            if (l2_stage_prev_valid && l2_stage_ne_prev_valid) begin
                for (j = 0; j < HL2_NEURONS; j = j + 1) begin
                    l2_expected_plus = l2_insum_ne_prev[j] + l2_bshift_ne_prev[j];
                    l2_plus_now = $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.plusReg[j]);
                    if (l2_plus_now === l2_expected_plus) begin
                        l2_plus_chk_pass = l2_plus_chk_pass + 1;
                        $fwrite(fd_l2_bias_diag, "%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                            cycle_cnt, j, l2_plus_now, l2_expected_plus, l2_insum_ne_prev[j], l2_bshift_ne_prev[j]);
                    end else begin
                        l2_plus_chk_fail = l2_plus_chk_fail + 1;
                        $fwrite(fd_l2_bias_diag, "%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                            cycle_cnt, j, l2_plus_now, l2_expected_plus, l2_insum_ne_prev[j], l2_bshift_ne_prev[j]);
                        if (l2_plus_chk_first_fail < 10) begin
                            $display("[L2_BIAS][%0d] FAIL: plus=%0d exp=%0d", j, l2_plus_now, l2_expected_plus);
                            l2_plus_chk_first_fail = l2_plus_chk_first_fail + 1;
                        end
                    end

                    l2_expected_plus = l2_insum_prev[j] + l2_bshift_prev[j];
                    if (l2_plus_now === l2_expected_plus) begin
                        l2_plus_dir_pass = l2_plus_dir_pass + 1;
                        $fwrite(fd_l2_bias_dir_diag, "%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                            cycle_cnt, j, l2_plus_now, l2_expected_plus, l2_insum_prev[j], l2_bshift_prev[j]);
                    end else begin
                        l2_plus_dir_fail = l2_plus_dir_fail + 1;
                        $fwrite(fd_l2_bias_dir_diag, "%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                            cycle_cnt, j, l2_plus_now, l2_expected_plus, l2_insum_prev[j], l2_bshift_prev[j]);
                        if (l2_plus_dir_first_fail < 10) begin
                            $display("[L2_BIAS_DIR][%0d] FAIL: plus=%0d exp=%0d", j, l2_plus_now, l2_expected_plus);
                            l2_plus_dir_first_fail = l2_plus_dir_first_fail + 1;
                        end
                    end

                    l2_expected_plus = l2_insum_ne_prev[j] + l2_bshift_ne_prev[j];
                    if (l2_plus_now === l2_expected_plus) begin
                        l2_plus_ne_pass = l2_plus_ne_pass + 1;
                        $fwrite(fd_l2_bias_ne_diag, "%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                            cycle_cnt, j, l2_plus_now, l2_expected_plus, l2_insum_ne_prev[j], l2_bshift_ne_prev[j]);
                    end else begin
                        l2_plus_ne_fail = l2_plus_ne_fail + 1;
                        $fwrite(fd_l2_bias_ne_diag, "%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                            cycle_cnt, j, l2_plus_now, l2_expected_plus, l2_insum_ne_prev[j], l2_bshift_ne_prev[j]);
                        if (l2_plus_ne_first_fail < 10) begin
                            $display("[L2_BIAS_NE][%0d] FAIL: plus=%0d exp=%0d", j, l2_plus_now, l2_expected_plus);
                            l2_plus_ne_first_fail = l2_plus_ne_first_fail + 1;
                        end
                    end

                    l2_expected_act = l2_trunc_ne_prev[j][ACT_WIDTH-1] ?
                        (l2_trunc_ne_prev[j] >>> 3) : l2_trunc_ne_prev[j];
                    l2_act_now = $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.actReg[j]);
                    if (l2_act_now === l2_expected_act) begin
                        l2_act_chk_pass = l2_act_chk_pass + 1;
                        $fwrite(fd_l2_act_diag, "%0d,%0d,%0d,%0d,%0d,1\n",
                            cycle_cnt, j, l2_act_now, l2_expected_act, l2_trunc_ne_prev[j]);
                    end else begin
                        l2_act_chk_fail = l2_act_chk_fail + 1;
                        $fwrite(fd_l2_act_diag, "%0d,%0d,%0d,%0d,%0d,0\n",
                            cycle_cnt, j, l2_act_now, l2_expected_act, l2_trunc_ne_prev[j]);
                        if (l2_act_chk_first_fail < 10) begin
                            $display("[L2_ACT][%0d] FAIL: act=%0d exp=%0d", j, l2_act_now, l2_expected_act);
                            l2_act_chk_first_fail = l2_act_chk_first_fail + 1;
                        end
                    end
                end
            end

            // L2 multiply-stage check across all PEs.
            // For columns c>0, expected input activation aligns to left-neighbor snapshot
            // under current TB sampling point.
            if (l2_mul_prev_valid) begin
                for (wt_row_idx = 0; wt_row_idx < HL2_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
                    for (wt_col_idx = 0; wt_col_idx < HL2_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                        l2_pe_idx = wt_row_idx*HL2_NEURONS + wt_col_idx;
                        if (wt_col_idx == 0)
                            l2_expected_mul = $signed(l2_pe_w_prev[l2_pe_idx]) * $signed(l2_pe_a_prev[l2_pe_idx]);
                        else
                            l2_expected_mul = $signed(l2_pe_w_prev[l2_pe_idx]) * $signed(l2_pe_a_prev[l2_pe_idx-1]);
                        l2_mul_now = $signed(l2_pe_mul_flat[l2_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                        if (l2_mul_now === l2_expected_mul) begin
                            l2_mul_chk_pass = l2_mul_chk_pass + 1;
                            $fwrite(fd_l2_mul_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l2_mul_now, l2_expected_mul,
                                l2_pe_w_prev[l2_pe_idx], l2_pe_a_prev[l2_pe_idx]);
                        end else begin
                            l2_mul_chk_fail = l2_mul_chk_fail + 1;
                            $fwrite(fd_l2_mul_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l2_mul_now, l2_expected_mul,
                                l2_pe_w_prev[l2_pe_idx], l2_pe_a_prev[l2_pe_idx]);
                            if (l2_mul_chk_first_fail < 10) begin
                                $display("[L2_MUL][r=%0d c=%0d] FAIL: mul=%0d exp=%0d", wt_row_idx, wt_col_idx, l2_mul_now, l2_expected_mul);
                                l2_mul_chk_first_fail = l2_mul_chk_first_fail + 1;
                            end
                        end
                    end
                end
            end

            // L2 PE accumulate-stage check: parSumReg should match previous-cycle
            // io_inWtPS + mulReg at each PE.
            if (l2_psum_prev_valid) begin
                for (wt_row_idx = 0; wt_row_idx < HL2_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
                    for (wt_col_idx = 0; wt_col_idx < HL2_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                        l2_pe_idx = wt_row_idx*HL2_NEURONS + wt_col_idx;
                        l2_expected_psum = $signed(l2_pe_inwtps_prev[l2_pe_idx]) + $signed(l2_pe_mul_prev[l2_pe_idx]);
                        l2_psum_now = $signed(l2_pe_psum_flat[l2_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                        if (l2_psum_now === l2_expected_psum) begin
                            l2_psum_chk_pass = l2_psum_chk_pass + 1;
                            $fwrite(fd_l2_psum_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l2_psum_now, l2_expected_psum,
                                l2_pe_inwtps_prev[l2_pe_idx], l2_pe_mul_prev[l2_pe_idx]);
                        end else begin
                            l2_psum_chk_fail = l2_psum_chk_fail + 1;
                            $fwrite(fd_l2_psum_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l2_psum_now, l2_expected_psum,
                                l2_pe_inwtps_prev[l2_pe_idx], l2_pe_mul_prev[l2_pe_idx]);
                            if (l2_psum_chk_first_fail < 10) begin
                                $display("[L2_PSUM][r=%0d c=%0d] FAIL: psum=%0d exp=%0d",
                                    wt_row_idx, wt_col_idx, l2_psum_now, l2_expected_psum);
                                l2_psum_chk_first_fail = l2_psum_chk_first_fail + 1;
                            end
                        end
                    end
                end
            end

            if (l2_psum_ne_prev_valid) begin
                for (wt_row_idx = 0; wt_row_idx < HL2_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
                    for (wt_col_idx = 0; wt_col_idx < HL2_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                        l2_pe_idx = wt_row_idx*HL2_NEURONS + wt_col_idx;
                        l2_expected_psum = $signed(l2_pe_inwtps_ne_prev[l2_pe_idx]) + $signed(l2_pe_mul_ne_prev[l2_pe_idx]);
                        l2_psum_now = $signed(l2_pe_psum_flat[l2_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                        if (l2_psum_now === l2_expected_psum) begin
                            l2_psum_ne_pass = l2_psum_ne_pass + 1;
                            $fwrite(fd_l2_psum_ne_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,1\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l2_psum_now, l2_expected_psum,
                                l2_pe_inwtps_ne_prev[l2_pe_idx], l2_pe_mul_ne_prev[l2_pe_idx]);
                        end else begin
                            l2_psum_ne_fail = l2_psum_ne_fail + 1;
                            $fwrite(fd_l2_psum_ne_diag, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0\n",
                                cycle_cnt, wt_row_idx, wt_col_idx, l2_psum_now, l2_expected_psum,
                                l2_pe_inwtps_ne_prev[l2_pe_idx], l2_pe_mul_ne_prev[l2_pe_idx]);
                            if (l2_psum_ne_first_fail < 10) begin
                                $display("[L2_PSUM_NE][r=%0d c=%0d] FAIL: psum=%0d exp=%0d",
                                    wt_row_idx, wt_col_idx, l2_psum_now, l2_expected_psum);
                                l2_psum_ne_first_fail = l2_psum_ne_first_fail + 1;
                            end
                        end
                    end
                end
            end

            for (l2_skew_ch = 0; l2_skew_ch < HL2_NEURONS; l2_skew_ch = l2_skew_ch + 1) begin
                l2_raw_now = $signed(u_dut.u_hidden_layers.hl2_to_hl3[l2_skew_ch*ACT_WIDTH +: ACT_WIDTH]);
                for (l2_skew_d = HL2_NEURONS-1; l2_skew_d > 0; l2_skew_d = l2_skew_d - 1)
                    l2_raw_delay_ref[l2_skew_ch][l2_skew_d] = l2_raw_delay_ref[l2_skew_ch][l2_skew_d-1];
                l2_raw_delay_ref[l2_skew_ch][0] = l2_raw_now;

                l2_deskew_now = $signed(l2_deskewed[l2_skew_ch*ACT_WIDTH +: ACT_WIDTH]);
                if (l2_deskew_now === l2_raw_delay_ref[l2_skew_ch][HL2_NEURONS-1-l2_skew_ch]) begin
                    l2_skew_chk_pass = l2_skew_chk_pass + 1;
                    $fwrite(fd_l2_skew_diag, "%0d,%0d,%0d,%0d,%0d,1\n",
                        cycle_cnt, l2_skew_ch, l2_deskew_now,
                        l2_raw_delay_ref[l2_skew_ch][HL2_NEURONS-1-l2_skew_ch], l2_raw_now);
                end else begin
                    l2_skew_chk_fail = l2_skew_chk_fail + 1;
                    $fwrite(fd_l2_skew_diag, "%0d,%0d,%0d,%0d,%0d,0\n",
                        cycle_cnt, l2_skew_ch, l2_deskew_now,
                        l2_raw_delay_ref[l2_skew_ch][HL2_NEURONS-1-l2_skew_ch], l2_raw_now);
                    if (l2_skew_chk_first_fail < 10) begin
                        $display("[L2_SKEW][ch=%0d] FAIL: deskew=%0d exp=%0d", l2_skew_ch, l2_deskew_now, l2_raw_delay_ref[l2_skew_ch][HL2_NEURONS-1-l2_skew_ch]);
                        l2_skew_chk_first_fail = l2_skew_chk_first_fail + 1;
                    end
                end
            end

            for (j = 0; j < HL2_NEURONS; j = j + 1) begin
                l2_sum_prev[j] = $signed(u_dut.u_hidden_layers.hl2_inst.array_out_sum[j*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                l2_bias_prev[j] = $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.biasReg[j]);
                l2_trunc_prev[j] = $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.truncWire[j]);
                l2_insum_prev[j] = $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.inSumWire[j]);
                l2_bshift_prev[j] = $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.biasShiftWire[j]);
            end
            l2_stage_prev_valid = 1'b1;

            for (wt_row_idx = 0; wt_row_idx < HL2_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
                for (wt_col_idx = 0; wt_col_idx < HL2_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                    l2_pe_idx = wt_row_idx*HL2_NEURONS + wt_col_idx;
                    l2_pe_w_prev[l2_pe_idx] = $signed(l2_pe_weight_flat[l2_pe_idx*WEIGHT_WIDTH +: WEIGHT_WIDTH]);
                    l2_pe_a_prev[l2_pe_idx] = $signed(l2_pe_act_flat[l2_pe_idx*ACT_WIDTH +: ACT_WIDTH]);
                    l2_pe_inwtps_prev[l2_pe_idx] = $signed(l2_pe_inwtps_flat[l2_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                    l2_pe_mul_prev[l2_pe_idx] = $signed(l2_pe_mul_flat[l2_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                end
            end
            l2_mul_prev_valid = 1'b1;
            l2_psum_prev_valid = 1'b1;
            end

            // layer2 check
            if (l2_rt_valid && l2_rt_idx < NUM_CYCLES) begin
                for (j = 0; j < HL2_NEURONS; j = j + 1) begin
                    fpga_val = $signed(l2_deskewed[j*ACT_WIDTH +: ACT_WIDTH]);
                    gold_val = $signed(l2_gold_mem[l2_rt_idx * HL2_NEURONS + j]);
                    cur_err = fpga_val - gold_val;
                    if (cur_err < 0) cur_err = -cur_err;

                    l2_total = l2_total + 1;
                    if (cur_err > l2_max_err) l2_max_err = cur_err;
                    if (cur_err <= TOLERANCE) begin
                        l2_pass = l2_pass + 1;
                    end else begin
                        l2_fail = l2_fail + 1;
                        if (l2_first_fail < 10) begin
                            $display("[L2][%0d][%0d] FAIL: fpga=%0d gold=%0d err=%0d", l2_rt_idx, j, fpga_val, gold_val, cur_err);
                            l2_first_fail = l2_first_fail + 1;
                        end
                    end
                end
            end

            // layer3 check
            if (l3_rt_valid && l3_rt_idx < NUM_CYCLES) begin
                for (j = 0; j < HL3_NEURONS; j = j + 1) begin
                    fpga_val = $signed(l3_deskewed[j*ACT_WIDTH +: ACT_WIDTH]);
                    gold_val = $signed(l3_gold_mem[l3_rt_idx * HL3_NEURONS + j]);
                    cur_err = fpga_val - gold_val;
                    if (cur_err < 0) cur_err = -cur_err;

                    l3_total = l3_total + 1;
                    if (cur_err > l3_max_err) l3_max_err = cur_err;
                    if (cur_err <= TOLERANCE) begin
                        l3_pass = l3_pass + 1;
                    end else begin
                        l3_fail = l3_fail + 1;
                        if (l3_first_fail < 10) begin
                            $display("[L3][%0d][%0d] FAIL: fpga=%0d gold=%0d err=%0d", l3_rt_idx, j, fpga_val, gold_val, cur_err);
                            l3_first_fail = l3_first_fail + 1;
                        end
                    end
                end
                l3_check_idx = l3_check_idx + 1;
            end

            // layer4 check (u_dut.ol_out already deskewed internally)
            if (l4_rt_valid && l4_check_idx < NUM_CYCLES) begin
                for (j = 0; j < OL_NEURONS; j = j + 1) begin
                    fpga_val = $signed(u_dut.ol_out[j*ACT_WIDTH +: ACT_WIDTH]);
                    gold_val = $signed(l4_gold_mem[l4_check_idx * OL_NEURONS + j]);
                    cur_err = fpga_val - gold_val;
                    if (cur_err < 0) cur_err = -cur_err;

                    l4_total = l4_total + 1;
                    if (cur_err > l4_max_err) l4_max_err = cur_err;
                    if (cur_err <= TOLERANCE) begin
                        l4_pass = l4_pass + 1;
                    end else begin
                        l4_fail = l4_fail + 1;
                        if (l4_first_fail < 10) begin
                            $display("[L4][%0d][%0d] FAIL: fpga=%0d gold=%0d err=%0d", l4_check_idx, j, fpga_val, gold_val, cur_err);
                            l4_first_fail = l4_first_fail + 1;
                        end
                    end
                end
                l4_check_idx = l4_check_idx + 1;
            end

            // final phase_denorm check
            if (pd_rt_valid && out_valid && pd_check_idx < NUM_CYCLES) begin
                for (j = 0; j < N_POLY; j = j + 1) begin
                    err_i = $signed(y_i[j*ACT_WIDTH +: ACT_WIDTH]) - $signed(pd_golden_i_mem[pd_check_idx * N_POLY + j]);
                    err_q = $signed(y_q[j*ACT_WIDTH +: ACT_WIDTH]) - $signed(pd_golden_q_mem[pd_check_idx * N_POLY + j]);
                    if (err_i < 0) err_i = -err_i;
                    if (err_q < 0) err_q = -err_q;

                    pd_total = pd_total + 1;
                    if (err_i > pd_max_err_i) pd_max_err_i = err_i;
                    if (err_q > pd_max_err_q) pd_max_err_q = err_q;

                    if (err_i <= TOLERANCE && err_q <= TOLERANCE) begin
                        pd_pass = pd_pass + 1;
                    end else begin
                        pd_fail = pd_fail + 1;
                        if (pd_first_fail < 10) begin
                            $display("[PD][%0d][%0d] FAIL: err_i=%0d err_q=%0d", pd_check_idx, j, err_i, err_q);
                            pd_first_fail = pd_first_fail + 1;
                        end
                    end
                end
                pd_check_idx = pd_check_idx + 1;
            end

        end

        // --------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------
        $display("");
        $display("==============================================================");
        $display("Simulation summary (V6)");
        $display("==============================================================");
        $display("PP      : checked=%0d/%0d pass=%0d/%0d fail=%0d max_err=%0d", pp_check_idx, NUM_CYCLES, pp_pass, pp_total, pp_fail, pp_max_err);
        $display("PP_RK   : checked=%0d/%0d pass=%0d/%0d fail=%0d max_re=%0d max_im=%0d", pp_rk_check_idx, NUM_CYCLES, rk_pass, rk_total, rk_fail, rk_max_err_re, rk_max_err_im);
        $display("L1      : checked=%0d/%0d pass=%0d/%0d fail=%0d max_err=%0d", l1_check_idx, NUM_CYCLES, l1_pass, l1_total, l1_fail, l1_max_err);
        $display("L1_WT   : pass=%0d fail=%0d total=%0d", l1_wt_pass, l1_wt_fail, HL1_IN_DIM*HL1_NEURONS);
        $display("L2_WT   : pass=%0d fail=%0d total=%0d", l2_wt_pass, l2_wt_fail, HL2_IN_DIM*HL2_NEURONS);
        $display("L3_WT   : pass=%0d fail=%0d total=%0d", l3_wt_pass, l3_wt_fail, HL3_IN_DIM*HL3_NEURONS);
        $display("L4_WT   : pass=%0d fail=%0d total=%0d", l4_wt_pass, l4_wt_fail, OL_IN_DIM*OL_NEURONS);
        if (ENABLE_INTERNAL_PE_DIAG) begin
            $display("L2_SKEW : pass=%0d fail=%0d", l2_skew_chk_pass, l2_skew_chk_fail);
            $display("L2_MUL  : pass=%0d fail=%0d", l2_mul_chk_pass, l2_mul_chk_fail);
            $display("L2_PSUM : pass=%0d fail=%0d", l2_psum_chk_pass, l2_psum_chk_fail);
            $display("L2_PSUM_NE: pass=%0d fail=%0d", l2_psum_ne_pass, l2_psum_ne_fail);
            $display("L2_BIAS : pass=%0d fail=%0d", l2_plus_chk_pass, l2_plus_chk_fail);
            $display("L2_BIAS_DIR: pass=%0d fail=%0d", l2_plus_dir_pass, l2_plus_dir_fail);
            $display("L2_BIAS_NE : pass=%0d fail=%0d", l2_plus_ne_pass, l2_plus_ne_fail);
            $display("L2_ACT  : pass=%0d fail=%0d", l2_act_chk_pass, l2_act_chk_fail);
            $display("L1_SKEW : pass=%0d fail=%0d", l1_skew_chk_pass, l1_skew_chk_fail);
            $display("L1_MUL  : pass=%0d fail=%0d", l1_mul_chk_pass, l1_mul_chk_fail);
            $display("L1_BIAS : pass=%0d fail=%0d", l1_plus_chk_pass, l1_plus_chk_fail);
            $display("L1_ACT  : pass=%0d fail=%0d", l1_act_chk_pass, l1_act_chk_fail);
        end
        $display("L2      : checked=%0d/%0d pass=%0d/%0d fail=%0d max_err=%0d", l2_check_idx, NUM_CYCLES, l2_pass, l2_total, l2_fail, l2_max_err);
        $display("L3      : checked=%0d/%0d pass=%0d/%0d fail=%0d max_err=%0d", l3_check_idx, NUM_CYCLES, l3_pass, l3_total, l3_fail, l3_max_err);
        $display("L4      : checked=%0d/%0d pass=%0d/%0d fail=%0d max_err=%0d", l4_check_idx, NUM_CYCLES, l4_pass, l4_total, l4_fail, l4_max_err);
        $display("PD      : checked=%0d/%0d pass=%0d/%0d fail=%0d max_i=%0d max_q=%0d", pd_check_idx, NUM_CYCLES, pd_pass, pd_total, pd_fail, pd_max_err_i, pd_max_err_q);

        if ((pp_fail + rk_fail + l1_fail + l2_fail + l3_fail + l4_fail + pd_fail) == 0)
            $display(">>>>>>>>> ALL PASS <<<<<<<<<");
        else
            $display(">>>>>>>>> FAILURES FOUND, CHECK LOGS <<<<<<<<<");

        $display("==============================================================");

        // close dump files
        $fclose(fd_pp_fpga); $fclose(fd_pp_gold);
        $fclose(fd_l1_fpga); $fclose(fd_l1_raw); $fclose(fd_l1_gold);
        $fclose(fd_l2_fpga); $fclose(fd_l2_raw); $fclose(fd_l2_gold);
        $fclose(fd_l3_fpga); $fclose(fd_l3_raw); $fclose(fd_l3_gold);
        $fclose(fd_l4_fpga); $fclose(fd_l4_raw); $fclose(fd_l4_gold);
        $fclose(fd_pd_fpga); $fclose(fd_pd_gold);
        $fclose(fd_pp_valid_feat);
        $fclose(fd_l1_wt_diag); $fclose(fd_l2_wt_diag); $fclose(fd_l3_wt_diag); $fclose(fd_l4_wt_diag);
        $fclose(fd_l2_skew_diag); $fclose(fd_l2_mul_diag); $fclose(fd_l2_psum_diag); $fclose(fd_l2_psum_ne_diag); $fclose(fd_l2_bias_diag); $fclose(fd_l2_bias_dir_diag); $fclose(fd_l2_bias_ne_diag); $fclose(fd_l2_act_diag);
        $fclose(fd_l1_skew_diag);
        $fclose(fd_l1_mul_diag); $fclose(fd_l1_bias_diag); $fclose(fd_l1_act_diag);

        $finish;
    end

    always @(negedge clk) begin
        if (ENABLE_INTERNAL_PE_DIAG) begin
            if (reset) begin
                l2_stage_ne_prev_valid <= 1'b0;
                for (j = 0; j < HL2_NEURONS; j = j + 1) begin
                    l2_insum_ne_prev[j] <= 0;
                    l2_bshift_ne_prev[j] <= 0;
                end
            end else begin
                for (j = 0; j < HL2_NEURONS; j = j + 1) begin
                    l2_insum_ne_prev[j] <= $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.inSumWire[j]);
                    l2_bshift_ne_prev[j] <= $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.biasShiftWire[j]);
                    l2_trunc_ne_prev[j] <= $signed(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.truncWire[j]);
                end
                for (wt_row_idx = 0; wt_row_idx < HL2_IN_DIM; wt_row_idx = wt_row_idx + 1) begin
                    for (wt_col_idx = 0; wt_col_idx < HL2_NEURONS; wt_col_idx = wt_col_idx + 1) begin
                        l2_pe_idx = wt_row_idx*HL2_NEURONS + wt_col_idx;
                        l2_pe_inwtps_ne_prev[l2_pe_idx] <= $signed(l2_pe_inwtps_flat[l2_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                        l2_pe_mul_ne_prev[l2_pe_idx] <= $signed(l2_pe_mul_flat[l2_pe_idx*INNER_ACC_WIDTH +: INNER_ACC_WIDTH]);
                    end
                end
                l2_stage_ne_prev_valid <= 1'b1;
                l2_psum_ne_prev_valid <= 1'b1;
            end
        end else begin
            l2_stage_ne_prev_valid <= 1'b0;
            l2_psum_ne_prev_valid <= 1'b0;
        end
    end

endmodule
