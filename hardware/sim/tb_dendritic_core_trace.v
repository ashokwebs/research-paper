// ============================================================================
// COMPUTATIONAL COUPLING -- Trace testbench for dendritic_lif_core
//
// Drives the two-compartment core with a stimulus sequence read from a file and
// writes one CSV row per timestep. The resulting trace is compared against a
// double-precision reference implementation of the same dynamics by
// experiments/hardware/run_rtl_verification.py, which is what turns the
// fixed-point error into a measured quantity rather than an asserted one.
//
// Stimulus file: one line per step, two whitespace-separated signed decimals in
// Q8.8 raw units (dendritic drive, perisomatic inhibitory drive).
// ============================================================================

`timescale 1ns / 1ps
`include "cen_defs.vh"

module tb_dendritic_core_trace;

    parameter integer NSTEPS = 2000;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg step_tick = 1'b0;

    reg signed [15:0] i_syn_dend = 16'sh0000;
    reg signed [15:0] i_syn_soma_inh = 16'sh0000;

    wire signed [15:0] v_dend_out;
    wire signed [15:0] v_soma_out;
    wire               nmda_active_out;
    wire               spike_out;

    integer stim_dend [0:NSTEPS-1];
    integer stim_inh  [0:NSTEPS-1];
    integer fd_out;
    integer fd_in;
    integer i;
    integer code;
    integer a, b;

    dendritic_lif_core dut (
        .clk(clk),
        .rst_n(rst_n),
        .step_tick(step_tick),
        .i_syn_dend(i_syn_dend),
        .i_syn_soma_inh(i_syn_soma_inh),
        .v_thresh_soma(`FP_V_THRESH_SOMA),
        .theta_dend(`FP_THETA_DEND),
        .v_plateau(`FP_V_PLATEAU),
        .g_nmda(`FP_G_NMDA),
        .g_axial(`FP_G_AXIAL),
        .v_dend_out(v_dend_out),
        .v_soma_out(v_soma_out),
        .nmda_active_out(nmda_active_out),
        .spike_out(spike_out)
    );

    always #5 clk = ~clk;   // 100 MHz

    initial begin
        fd_in = $fopen("stimulus.txt", "r");
        if (fd_in == 0) begin
            $display("ERROR: cannot open stimulus.txt");
            $finish;
        end
        for (i = 0; i < NSTEPS; i = i + 1) begin
            code = $fscanf(fd_in, "%d %d\n", a, b);
            stim_dend[i] = a;
            stim_inh[i]  = b;
        end
        $fclose(fd_in);

        fd_out = $fopen("rtl_trace.csv", "w");
        $fwrite(fd_out, "step,v_dend,v_soma,nmda,spike\n");

        // Reset
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        for (i = 0; i < NSTEPS; i = i + 1) begin
            i_syn_dend     = stim_dend[i][15:0];
            i_syn_soma_inh = stim_inh[i][15:0];

            // One step_tick pulse per simulated timestep.
            @(negedge clk);
            step_tick = 1'b1;
            @(posedge clk);
            @(negedge clk);
            step_tick = 1'b0;
            @(posedge clk);

            $fwrite(fd_out, "%0d,%0d,%0d,%0d,%0d\n",
                    i, $signed(v_dend_out), $signed(v_soma_out),
                    nmda_active_out, spike_out);
        end

        $fclose(fd_out);
        $display("[tb] wrote %0d steps to rtl_trace.csv", NSTEPS);
        $finish;
    end

endmodule
