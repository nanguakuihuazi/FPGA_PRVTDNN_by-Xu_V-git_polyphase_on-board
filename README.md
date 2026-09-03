# FPGA_PRVTDNN_by-Xu_V-git_polyphase_on-board

用于对本科毕设的vivado代码进行版本管理

**一、各部分设计的相应解释**

1.参数加载模块

参数加载的核心逻辑可以用下面这张图表示，网络的权重偏置参数由外部输入信号`input   [31:0]    coeff_i`输入，首先输入给内部的寄存器`    (* ram_style = "block" *) reg [31:0] coeff_data [0:COEFF_WORDS-1];`

```Verilog
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
```

待`coeff_i`输入结束，参数先全部存放在`coeff_data`中，再由`coeff_data`分发至各层的权重偏置注入接口`hl1_inWt_top`和`hl1_inBias`（参数分发的开启由`load_start`信号决定，参数写入`coeff_data`完成后才会拉高`load_start`），该参数分发的过程由状态机控制，`LD_HL1_WT_RD`（读取某一行的权重参数）、`LD_HL1_WT_PUSH`（将某一行的权重推送至`hl1_inWt_top`）、`LD_HL1_BIAS_RD`（读取并推送偏置参数），这里指出`LD_HL1_WT_RD`和`LD_HL1_WT_PUSH`也就是权重参数的读取和推送会执行多次（权重矩阵有多行，一次只能推送一行，而偏置矩阵只有一行所以不涉及多次读取和推送），涉及状态机的反复跳转。

![1788425066670](image/README/1788425066670.png)

至于为什么设置"外界参数先存入`coeff_data`，再分发至各层的权重偏置注入接口"。

时钟域隔离：外部通过 `AXI_clk_i` 写入，而网络加载和计算处于 `clk` 域。`coeff_data` 作为中间参数存储，并依靠`load_start`信号，使两个过程解耦。

状态机：是为了把“多端口读取需求”降为“单端口顺序读取”。如果直接让所有层、所有 PE 同时从 `coeff_data` 取权重/偏置，综合器看到的会是大量并发读端口。例如第一层一次要面向 `16 × 45` 个权重位置，而单个 BRAM 通常只有有限端口（典型为双端口），无法天然支持这种大量独立读取。

这里当时李哲师兄采用的是FIFO结构来解决跨时钟域的问题，采用FIFO和本方案的区别或者优势是，FIFO支持跨时钟域的同时读写，而我的这个方案需要等`coeff_data`全部写完成，也就是要先写入完成才能进行数据的读取和分发。

本版本暂时不尝试FIFO，待后续有时间再考虑FIFO的结构。
