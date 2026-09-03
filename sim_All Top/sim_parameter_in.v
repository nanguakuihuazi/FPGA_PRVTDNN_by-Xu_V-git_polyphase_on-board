`timescale 1ns / 1ps
`include "prvtdnn_defines.vh"

// Self-checking parameter-load test.  It verifies the AXI image, the
// unpadded source ordering, and the final PE/bias registers of all four layers.
module sim_parameter_in;
    localparam CLK_PERIOD = 10;
    localparam COEFF_WORDS = `TOTAL_PARAMS / 2;
    localparam LOAD_TIMEOUT_CYCLES = 20000;
    localparam L1_IN = `FC1_IN_DIM, L1_N = `FC1_OUT_DIM;
    localparam L2_IN = `FC2_IN_DIM, L2_N = `FC2_OUT_DIM;
    localparam L3_IN = `FC3_IN_DIM, L3_N = `FC3_OUT_DIM;
    localparam L4_IN = `FC5_IN_DIM, L4_N = `FC5_OUT_DIM;
    localparam L1_W_COUNT = `FC1_W_SIZE, L1_B_COUNT = `FC1_B_SIZE;
    localparam L2_W_COUNT = `FC2_W_SIZE, L2_B_COUNT = `FC2_B_SIZE;
    localparam L3_W_COUNT = `FC3_W_SIZE, L3_B_COUNT = `FC3_B_SIZE;
    localparam L4_W_COUNT = `FC5_W_SIZE, L4_B_COUNT = `FC5_B_SIZE;

    reg clk, AXI_clk_i, reset, vio_wdpd_i, coeff_en_i;
    reg [127:0] data_in;
    reg [31:0] coeff_i;
    reg [15:0] coeff_addr_i;
    wire [127:0] data_out;
    wire out_valid;
    wire [15:0] sample0_r_ila, sample0_i_ila;

    reg [31:0] coeff_mem [0:COEFF_WORDS-1];
    reg [15:0] all_weights_mem [0:`TOTAL_PARAMS-1];
    reg [15:0] l1_weight_mem [0:L1_W_COUNT-1];
    reg [15:0] l1_bias_mem   [0:L1_B_COUNT-1];
    reg [15:0] l2_weight_mem [0:L2_W_COUNT-1];
    reg [15:0] l2_bias_mem   [0:L2_B_COUNT-1];
    reg [15:0] l3_weight_mem [0:L3_W_COUNT-1];
    reg [15:0] l3_bias_mem   [0:L3_B_COUNT-1];
    reg [15:0] l4_weight_mem [0:L4_W_COUNT-1];
    reg [15:0] l4_bias_mem   [0:L4_B_COUNT-1];

    wire [L1_W_COUNT-1:0] l1_weight_error;
    wire [L1_B_COUNT-1:0] l1_bias_error;
    wire [L2_W_COUNT-1:0] l2_weight_error;
    wire [L2_B_COUNT-1:0] l2_bias_error;
    wire [L3_W_COUNT-1:0] l3_weight_error;
    wire [L3_B_COUNT-1:0] l3_bias_error;
    wire [L4_W_COUNT-1:0] l4_weight_error;
    wire [L4_B_COUNT-1:0] l4_bias_error;

    integer i, error_count, load_wait_cycles;

    // Unpadded files are arranged in exactly the DUT parameter-load order.
    function [15:0] expected_param;
        input integer param_index;
        begin
            if (param_index < `FC1_B_BASE)
                expected_param = l1_weight_mem[param_index - `FC1_W_BASE];
            else if (param_index < `FC2_W_BASE)
                expected_param = l1_bias_mem[param_index - `FC1_B_BASE];
            else if (param_index < `FC2_B_BASE)
                expected_param = l2_weight_mem[param_index - `FC2_W_BASE];
            else if (param_index < `FC3_W_BASE)
                expected_param = l2_bias_mem[param_index - `FC2_B_BASE];
            else if (param_index < `FC3_B_BASE)
                expected_param = l3_weight_mem[param_index - `FC3_W_BASE];
            else if (param_index < `FC5_W_BASE)
                expected_param = l3_bias_mem[param_index - `FC3_B_BASE];
            else if (param_index < `FC5_B_BASE)
                expected_param = l4_weight_mem[param_index - `FC5_W_BASE];
            else
                expected_param = l4_bias_mem[param_index - `FC5_B_BASE];
        end
    endfunction

    prvtdnn_top u_dut (
        .clk(clk), .AXI_clk_i(AXI_clk_i), .reset(reset), .data_in(data_in),
        .vio_wdpd_i(vio_wdpd_i), .coeff_i(coeff_i),
        .coeff_addr_i(coeff_addr_i), .coeff_en_i(coeff_en_i),
        .data_out(data_out), .out_valid(out_valid),
        .sample0_r_ila(sample0_r_ila), .sample0_i_ila(sample0_i_ila)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD / 10.0) clk = ~clk;
    initial AXI_clk_i = 1'b0;
    always #(CLK_PERIOD / 2.0) AXI_clk_i = ~AXI_clk_i;

    // Genvar indexes make every hierarchical PE reference an elaboration-time
    // constant, as required for generated row_gen/col_gen instance arrays.
    genvar r1, c1, r2, c2, r3, c3, r4, c4, b1, b2, b3, b4;
    generate
        for (r1 = 0; r1 < L1_IN; r1 = r1 + 1) begin : gen_l1_weight_row
            for (c1 = 0; c1 < L1_N; c1 = c1 + 1) begin : gen_l1_weight_col
                parameter_endpoint_checker #(.LAYER(1), .IS_BIAS(0), .NEURON(c1), .ROW(r1)) check (
                    .clk(clk), .load_done(u_dut.load_done),
                    .actual(u_dut.u_hidden_layers.hl1_inst.systolic_array_inst.row_gen[r1].col_gen[c1].pe_inst.weightReg),
                    .expected(l1_weight_mem[c1*L1_IN+r1]), .error_flag(l1_weight_error[c1*L1_IN+r1]));
            end
        end
        for (b1 = 0; b1 < L1_N; b1 = b1 + 1) begin : gen_l1_bias
            parameter_endpoint_checker #(.LAYER(1), .IS_BIAS(1), .NEURON(b1), .ROW(0)) check (
                .clk(clk), .load_done(u_dut.load_done), .actual(u_dut.u_hidden_layers.hl1_inst.bias_act_inst.biasReg[b1]),
                .expected(l1_bias_mem[b1]), .error_flag(l1_bias_error[b1]));
        end
        for (r2 = 0; r2 < L2_IN; r2 = r2 + 1) begin : gen_l2_weight_row
            for (c2 = 0; c2 < L2_N; c2 = c2 + 1) begin : gen_l2_weight_col
                parameter_endpoint_checker #(.LAYER(2), .IS_BIAS(0), .NEURON(c2), .ROW(r2)) check (
                    .clk(clk), .load_done(u_dut.load_done),
                    .actual(u_dut.u_hidden_layers.hl2_inst.systolic_array_inst.row_gen[r2].col_gen[c2].pe_inst.weightReg),
                    .expected(l2_weight_mem[c2*L2_IN+r2]), .error_flag(l2_weight_error[c2*L2_IN+r2]));
            end
        end
        for (b2 = 0; b2 < L2_N; b2 = b2 + 1) begin : gen_l2_bias
            parameter_endpoint_checker #(.LAYER(2), .IS_BIAS(1), .NEURON(b2), .ROW(0)) check (
                .clk(clk), .load_done(u_dut.load_done), .actual(u_dut.u_hidden_layers.hl2_inst.bias_act_inst.biasReg[b2]),
                .expected(l2_bias_mem[b2]), .error_flag(l2_bias_error[b2]));
        end
        for (r3 = 0; r3 < L3_IN; r3 = r3 + 1) begin : gen_l3_weight_row
            for (c3 = 0; c3 < L3_N; c3 = c3 + 1) begin : gen_l3_weight_col
                parameter_endpoint_checker #(.LAYER(3), .IS_BIAS(0), .NEURON(c3), .ROW(r3)) check (
                    .clk(clk), .load_done(u_dut.load_done),
                    .actual(u_dut.u_hidden_layers.hl3_inst.systolic_array_inst.row_gen[r3].col_gen[c3].pe_inst.weightReg),
                    .expected(l3_weight_mem[c3*L3_IN+r3]), .error_flag(l3_weight_error[c3*L3_IN+r3]));
            end
        end
        for (b3 = 0; b3 < L3_N; b3 = b3 + 1) begin : gen_l3_bias
            parameter_endpoint_checker #(.LAYER(3), .IS_BIAS(1), .NEURON(b3), .ROW(0)) check (
                .clk(clk), .load_done(u_dut.load_done), .actual(u_dut.u_hidden_layers.hl3_inst.bias_act_inst.biasReg[b3]),
                .expected(l3_bias_mem[b3]), .error_flag(l3_bias_error[b3]));
        end
        for (r4 = 0; r4 < L4_IN; r4 = r4 + 1) begin : gen_l4_weight_row
            for (c4 = 0; c4 < L4_N; c4 = c4 + 1) begin : gen_l4_weight_col
                parameter_endpoint_checker #(.LAYER(4), .IS_BIAS(0), .NEURON(c4), .ROW(r4)) check (
                    .clk(clk), .load_done(u_dut.load_done),
                    .actual(u_dut.u_output_layer.systolic_array_inst.row_gen[r4].col_gen[c4].pe_inst.weightReg),
                    .expected(l4_weight_mem[c4*L4_IN+r4]), .error_flag(l4_weight_error[c4*L4_IN+r4]));
            end
        end
        for (b4 = 0; b4 < L4_N; b4 = b4 + 1) begin : gen_l4_bias
            parameter_endpoint_checker #(.LAYER(4), .IS_BIAS(1), .NEURON(b4), .ROW(0)) check (
                .clk(clk), .load_done(u_dut.load_done), .actual(u_dut.u_output_layer.bias_act_inst.biasReg[b4]),
                .expected(l4_bias_mem[b4]), .error_flag(l4_bias_error[b4]));
        end
    endgenerate

    initial begin
        reset = 1'b1; data_in = 128'd0; vio_wdpd_i = 1'b1;
        coeff_i = 32'd0; coeff_addr_i = 16'd0; coeff_en_i = 1'b0;
        error_count = 0;

        $readmemh("parameter_in.mem", coeff_mem);
        $readmemh("all_weights_poly_multiout.mem", all_weights_mem);
        $readmemh("layer1_weights_poly_multiout.mem", l1_weight_mem);
        $readmemh("layer1_bias_poly_multiout.mem", l1_bias_mem);
        $readmemh("layer2_weights_poly_multiout.mem", l2_weight_mem);
        $readmemh("layer2_bias_poly_multiout.mem", l2_bias_mem);
        $readmemh("layer3_weights_poly_multiout.mem", l3_weight_mem);
        $readmemh("layer3_bias_poly_multiout.mem", l3_bias_mem);
        $readmemh("layer4_weights_poly_multiout.mem", l4_weight_mem);
        $readmemh("layer4_bias_poly_multiout.mem", l4_bias_mem);

        // Check the aggregate source and 16-bit-to-32-bit AXI packing before
        // transfer.  parameter_in.mem word i is {parameter 2*i+1, parameter 2*i}.
        for (i = 0; i < `TOTAL_PARAMS; i = i + 1) begin
            if (all_weights_mem[i] !== expected_param(i)) begin
                error_count = error_count + 1;
                $display("[ERROR] unpadded aggregate index=%0d expected=%h got=%h",
                         i, expected_param(i), all_weights_mem[i]);
            end
        end
        for (i = 0; i < COEFF_WORDS; i = i + 1) begin
            if (coeff_mem[i] !== {expected_param(2*i+1), expected_param(2*i)}) begin
                error_count = error_count + 1;
                $display("[ERROR] packed word index=%0d expected=%h got=%h",
                         i, {expected_param(2*i+1), expected_param(2*i)}, coeff_mem[i]);
            end
        end

        #(CLK_PERIOD * 10); reset = 1'b0; #(CLK_PERIOD * 5);
        $display("==============================================================");
        $display("Layered parameter verification start: %0d AXI words, %0d parameters",
                 COEFF_WORDS, `TOTAL_PARAMS);
        $display("==============================================================");

        for (i = 0; i < COEFF_WORDS; i = i + 1) begin
            @(negedge AXI_clk_i);
            coeff_en_i <= 1'b1; coeff_addr_i <= i[15:0]; coeff_i <= coeff_mem[i];
            @(posedge AXI_clk_i); #1;
            if (u_dut.coeff_data[i] !== coeff_mem[i]) begin
                error_count = error_count + 1;
                $display("[ERROR] AXI write index=%0d expected=%h got=%h",
                         i, coeff_mem[i], u_dut.coeff_data[i]);
            end
        end

        @(negedge AXI_clk_i);
        coeff_en_i <= 1'b0; coeff_addr_i <= 16'd0; coeff_i <= 32'd0;
        @(posedge AXI_clk_i); #1;
        for (i = 0; i < COEFF_WORDS; i = i + 1) begin
            if (u_dut.coeff_data[i] !== coeff_mem[i]) begin
                error_count = error_count + 1;
                $display("[ERROR] final AXI image index=%0d expected=%h got=%h",
                         i, coeff_mem[i], u_dut.coeff_data[i]);
            end
        end

        load_wait_cycles = 0;
        begin : wait_for_load_done
            forever begin
                @(posedge clk);
                if (u_dut.load_done) disable wait_for_load_done;
                load_wait_cycles = load_wait_cycles + 1;
                if (load_wait_cycles > LOAD_TIMEOUT_CYCLES) begin
                    error_count = error_count + 1;
                    $display("[ERROR] timeout waiting for u_dut.load_done after %0d clk cycles",
                             LOAD_TIMEOUT_CYCLES);
                    disable wait_for_load_done;
                end
            end
        end

        #2; // Endpoint checkers sample one delta after load_done.
        for (i = 0; i < L1_W_COUNT; i = i + 1) if (l1_weight_error[i]) error_count = error_count + 1;
        for (i = 0; i < L1_B_COUNT; i = i + 1) if (l1_bias_error[i]) error_count = error_count + 1;
        for (i = 0; i < L2_W_COUNT; i = i + 1) if (l2_weight_error[i]) error_count = error_count + 1;
        for (i = 0; i < L2_B_COUNT; i = i + 1) if (l2_bias_error[i]) error_count = error_count + 1;
        for (i = 0; i < L3_W_COUNT; i = i + 1) if (l3_weight_error[i]) error_count = error_count + 1;
        for (i = 0; i < L3_B_COUNT; i = i + 1) if (l3_bias_error[i]) error_count = error_count + 1;
        for (i = 0; i < L4_W_COUNT; i = i + 1) if (l4_weight_error[i]) error_count = error_count + 1;
        for (i = 0; i < L4_B_COUNT; i = i + 1) if (l4_bias_error[i]) error_count = error_count + 1;

        repeat (4) @(posedge AXI_clk_i);
        if (coeff_en_i !== 1'b0) begin
            error_count = error_count + 1;
            $display("[ERROR] coeff_en_i must remain low after the transfer");
        end
        if (error_count == 0)
            $display("[PASS] Layered parameter verification passed: 1360 weights and 56 biases match all four layers.");
        else
            $display("[FAIL] Layered parameter verification failed: %0d error(s).", error_count);
        $finish;
    end
endmodule

module parameter_endpoint_checker #(
    parameter LAYER = 0,
    parameter IS_BIAS = 0,
    parameter NEURON = 0,
    parameter ROW = 0
) (
    input clk,
    input load_done,
    input signed [15:0] actual,
    input [15:0] expected,
    output reg error_flag
);
    initial error_flag = 1'b0;
    always @(posedge load_done) begin
        // BiasActivation captures io_inBiasValid first, then writes biasReg
        // on the following clk edge.  This also safely samples all PE values.
        @(posedge clk); #1;
        if (actual !== expected) begin
            error_flag = 1'b1;
            if (IS_BIAS)
                $display("[ERROR] L%0d bias neuron=%0d expected=%h got=%h",
                         LAYER, NEURON, expected, actual);
            else
                $display("[ERROR] L%0d weight neuron=%0d row=%0d expected=%h got=%h",
                         LAYER, NEURON, ROW, expected, actual);
        end
    end
endmodule
