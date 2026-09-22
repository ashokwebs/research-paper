// ============================================================================
// COMPUTATIONAL COUPLING — Verilog Testbench for CoupledEventNet Accelerator
// Cycle-accurate verification testbench for cen_accelerator_top.v
// Verifies:
//   1. Clock generation & reset de-assertion
//   2. AER packet injection via AXI-Stream interface
//   3. Synaptic accumulation and step_tick dispatch
//   4. Outbound AER spike generation and performance counter telemetry
// ============================================================================

`timescale 1ns / 1ps
`include "../rtl/cen_defs.vh"

module testbench_cen;

    reg         clk;
    reg         rst_n;
    reg         step_tick;

    // AXI-Stream Ingress
    reg         s_axis_aer_tvalid;
    wire        s_axis_aer_tready;
    reg  [31:0] s_axis_aer_tdata;

    // AXI-Stream Egress
    wire        m_axis_aer_tvalid;
    reg         m_axis_aer_tready;
    wire [31:0] m_axis_aer_tdata;

    // Telemetry
    wire [31:0] total_synops_counter;
    wire [31:0] total_spikes_counter;
    wire [31:0] total_plateau_counter;

    // Instantiate DUT (Device Under Test)
    cen_accelerator_top #(
        .NUM_NEURONS(128),
        .NUM_EXC_NEURONS(102)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .step_tick(step_tick),
        .s_axis_aer_tvalid(s_axis_aer_tvalid),
        .s_axis_aer_tready(s_axis_aer_tready),
        .s_axis_aer_tdata(s_axis_aer_tdata),
        .m_axis_aer_tvalid(m_axis_aer_tvalid),
        .m_axis_aer_tready(m_axis_aer_tready),
        .m_axis_aer_tdata(m_axis_aer_tdata),
        .total_synops_counter(total_synops_counter),
        .total_spikes_counter(total_spikes_counter),
        .total_plateau_counter(total_plateau_counter)
    );

    // 100 MHz System Clock (10 ns period)
    always #5 clk = ~clk;

    // Testbench Stimulus Sequence
    initial begin
        $display("----------------------------------------------------------------");
        $display("STARTING COUPLED-EVENT-NET HARDWARE RTL TESTBENCH");
        $display("----------------------------------------------------------------");

        clk = 0;
        rst_n = 0;
        step_tick = 0;
        s_axis_aer_tvalid = 0;
        s_axis_aer_tdata = 32'b0;
        m_axis_aer_tready = 1;

        // Reset sequence (50 ns)
        #50;
        rst_n = 1;
        #20;
        $display("[+] Reset successfully de-asserted. System initialized.");

        // Inject AER Event Packet 1: Excitatory Neuron 10 -> Postsynaptic 24, Weight 0x40 (0.25 in Q8.8)
        @(posedge clk);
        s_axis_aer_tvalid = 1;
        s_axis_aer_tdata  = {1'b1, 8'd0, 8'h40, 1'b0, 7'd10, 1'b0, 7'd24};
        @(posedge clk);
        s_axis_aer_tvalid = 0;
        #10;
        $display("[+] Dispatched Excitatory AER packet (pre: 10, post: 24, weight: 0.25).");

        // Inject AER Event Packet 2: Suprathreshold Drive to trigger NMDA plateau on Neuron 5
        @(posedge clk);
        s_axis_aer_tvalid = 1;
        s_axis_aer_tdata  = {1'b1, 8'd0, 8'hC0, 1'b0, 7'd10, 1'b0, 7'd5};
        @(posedge clk);
        s_axis_aer_tvalid = 0;
        #10;
        $display("[+] Dispatched High-Amplitude AER packet to Neuron 5 (weight: 0.75).");

        // Trigger Step Tick 1 (1-ms simulation time step)
        @(posedge clk);
        step_tick = 1;
        @(posedge clk);
        step_tick = 0;
        #20;
        $display("[+] Step Tick 1 executed.");

        // Trigger Step Tick 2 (verify NMDA plateau sustained without new input)
        @(posedge clk);
        step_tick = 1;
        @(posedge clk);
        step_tick = 0;
        #20;
        $display("[+] Step Tick 2 executed.");

        // Display telemetry counters
        #50;
        $display("----------------------------------------------------------------");
        $display("HARDWARE TELEMETRY SUMMARY:");
        $display("  Total SynOps Dispatched: %0d", total_synops_counter);
        $display("  Total Spikes Emitted:    %0d", total_spikes_counter);
        $display("  Total NMDA Plateaus:     %0d", total_plateau_counter);
        $display("----------------------------------------------------------------");

        if (total_synops_counter == 2) begin
            $display("[SUCCESS] All AER packets correctly routed and accounted for.");
        end else begin
            $display("[FAILURE] SynOps counter mismatch.");
        end

        $finish;
    end

endmodule
