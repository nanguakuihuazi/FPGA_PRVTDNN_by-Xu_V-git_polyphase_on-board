//=============================================================================
// PRVTDNN ??????
// ?? MATLAB PhasedArray_ForwardModeling_PRVTDNN_Polyphase_Fixed_V2.m ???????
// ??: PN-RVTDNN Polyphase ? MLP (45?32?28?24?8), ????, Leaky ReLU
// ????: Q1.14 (1??? + 1??? + 14???), ?? [-2, 2)
//=============================================================================

`ifndef PRVTDNN_DEFINES_VH
`define PRVTDNN_DEFINES_VH

//-------------------------------------------------------------
// ?????? (?? PhasedArray_ForwardModeling_PRVTDNN_Polyphase_Fixed_V2.m)
//-------------------------------------------------------------
`define M               5           // ????
`define N_POLY          4           // ?????
`define M_POLY          (`M + `N_POLY - 1)
`define TAP_NUM         (`M_POLY + 1)
`define P               3           // ???? (R, R?, R?)

// ????: Re(X_pn)?9 + Im(X_pn)?9 + R?9 + R??9 + R??9 = 45
`define FEAT_DIM        (5 * `TAP_NUM)
`define RK_DIM          `N_POLY

// FC ??? (hidden_sizes = [16, 16, 16], output = 2*N_POLY = 8)
`define FC1_IN_DIM      `FEAT_DIM   // 45
`define FC1_OUT_DIM     16
`define FC2_IN_DIM      16
`define FC2_OUT_DIM     16
`define FC3_IN_DIM      16
`define FC3_OUT_DIM     16
`define FC5_IN_DIM      16
`define FC5_OUT_DIM     (2 * `N_POLY)

// ????? (ceil(log2(IN_DIM)), ?? pad ? 2^TREE_DEPTH)
`define FC1_TREE_DEPTH  6           // ceil(log2(45)) = 6, pad to 64
`define FC2_TREE_DEPTH  4           // ceil(log2(16)) = 4, pad to 16
`define FC3_TREE_DEPTH  4           // ceil(log2(16)) = 4, pad to 16
`define FC5_TREE_DEPTH  4           // ceil(log2(16)) = 4, pad to 16

// Leaky ReLU: alpha = 0.125 = 2^(-3) ? ???? 3 ?
`define LEAKY_SHIFT     3

//-------------------------------------------------------------
// ?????? (? VDTDNN ????)
//-------------------------------------------------------------
`define DATA_WIDTH      16          // ??/??/??????
`define FRAC_BITS       14          // Q1.14 ????
`define INT_BITS        1           // Q1.14 ???? (?????)
`define ACC_WIDTH       40          // ????? (???????)
`define PROD_WIDTH      32          // ???? (16?16)
`define PROD_FRAC       28          // ????? = 2 ? FRAC_BITS

//-------------------------------------------------------------
// ???? (? VDTDNN fc_layer.v ??)
//-------------------------------------------------------------
`define TRUNC_POS       (`PROD_FRAC)            // 28
`define SIGN_BIT_POS    (`TRUNC_POS + `INT_BITS) // 29

//-------------------------------------------------------------
// BRAM / ?????? (1416 ? Q1.14 ?)
//-------------------------------------------------------------
// ?????? (?????):
//   W1: 16?45 = 720     b1: 16
//   W2: 16?16 = 256     b2: 16
//   W3: 16?16 = 256     b3: 16
//   W5: 8?16  = 128     b5: 8
//   Total: 1416

`define FC1_W_BASE      0
`define FC1_W_SIZE      (`FC1_OUT_DIM * `FC1_IN_DIM)    // 720
`define FC1_B_BASE      (`FC1_W_BASE + `FC1_W_SIZE)     // 720
`define FC1_B_SIZE      (`FC1_OUT_DIM)                  // 16

`define FC2_W_BASE      (`FC1_B_BASE + `FC1_B_SIZE)     // 736
`define FC2_W_SIZE      (`FC2_OUT_DIM * `FC2_IN_DIM)    // 256
`define FC2_B_BASE      (`FC2_W_BASE + `FC2_W_SIZE)     // 992
`define FC2_B_SIZE      (`FC2_OUT_DIM)                  // 16

`define FC3_W_BASE      (`FC2_B_BASE + `FC2_B_SIZE)     // 1008
`define FC3_W_SIZE      (`FC3_OUT_DIM * `FC3_IN_DIM)    // 256
`define FC3_B_BASE      (`FC3_W_BASE + `FC3_W_SIZE)     // 1264
`define FC3_B_SIZE      (`FC3_OUT_DIM)                  // 16

`define FC5_W_BASE      (`FC3_B_BASE + `FC3_B_SIZE)     // 1280
`define FC5_W_SIZE      (`FC5_OUT_DIM * `FC5_IN_DIM)    // 128
`define FC5_B_BASE      (`FC5_W_BASE + `FC5_W_SIZE)     // 1408
`define FC5_B_SIZE      (`FC5_OUT_DIM)                  // 8

`define TOTAL_PARAMS    (`FC5_B_BASE + `FC5_B_SIZE)     // 1416
`define BRAM_ADDR_WIDTH 11          // ceil(log2(1416)) = 11
`define BRAM_DATA_WIDTH `DATA_WIDTH // 16

//-------------------------------------------------------------
// CORDIC ?? (? VDTDNN ????)
//-------------------------------------------------------------
`define CORDIC_ITER     14          // CORDIC ????
`define CORDIC_LATENCY  (`CORDIC_ITER + 1)  // 15 cycles
`define CORDIC_WIDTH    18          // ???? (?2????)
`define CORDIC_INV_K    16'd9949    // 1/K ? 0.6073 in Q1.14

//-------------------------------------------------------------
// ????????
//-------------------------------------------------------------
// Stage A: 15 cyc (9? CORDIC VEC X[n..n-M_poly])
// Stage B: 15 cyc (9? ?????????? PN ?? + 4? rk)
// Stage C: 2 cyc  (R?, R? ?? + ??)
// Output:  1 cyc  (?????)
`define PP_LATENCY      (2 * `CORDIC_LATENCY + 3) // 2?15+3 = 33 cycles

//-------------------------------------------------------------
// FC ?????
// ???? = 1 (??) + TREE_DEPTH (???) + 1 (??+??+??)
//-------------------------------------------------------------
`define FC1_LATENCY     (1 + `FC1_TREE_DEPTH + 1)  // 8
`define FC2_LATENCY     (1 + `FC2_TREE_DEPTH + 1)  // 6
`define FC3_LATENCY     (1 + `FC3_TREE_DEPTH + 1)  // 6
`define FC5_LATENCY     (1 + `FC5_TREE_DEPTH + 1)  // 6

// ?????
`define TOTAL_LATENCY   (`PP_LATENCY + `FC1_LATENCY + `FC2_LATENCY + \
                         `FC3_LATENCY + `FC5_LATENCY)  // 59

`endif // PRVTDNN_DEFINES_VH
