// ============================================================================
// COMPUTATIONAL COUPLING — Two-Compartment Active Dendritic LIF Hardware Core
// Synthesizable Verilog RTL for high-efficiency neuromorphic inference
// Arithmetic: 16-bit Signed Fixed-Point Q8.8
// Features:
//   - High-impedance dendritic compartment with NMDA regenerative plateau
//   - Forward axial conductance coupling (V_d -> V_s)
//   - Perisomatic GABA shunting inhibition clamp
//   - Somatic action potential generation and refractory state machine
// ============================================================================

`include "cen_defs.vh"

module dendritic_lif_core (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     step_tick,          // 1 ms timestep pulse

    // Incoming Synaptic Inputs (Q8.8)
    input  wire signed [15:0]       i_syn_dend,         // Excitatory dendritic drive
    input  wire signed [15:0]       i_syn_soma_inh,     // Inhibitory perisomatic drive

    // Configuration / Parameters (Q8.8)
    input  wire signed [15:0]       v_thresh_soma,
    input  wire signed [15:0]       theta_dend,
    input  wire signed [15:0]       v_plateau,
    input  wire signed [15:0]       g_nmda,
    input  wire signed [15:0]       g_axial,

    // State Outputs
    output reg  signed [15:0]       v_dend_out,         // Dendritic membrane potential
    output reg  signed [15:0]       v_soma_out,         // Somatic membrane potential
    output reg                      nmda_active_out,    // NMDA plateau status flag
    output reg                      spike_out           // Somatic action potential event
);

    // Q8.8 Fixed-point multiplier function
    function signed [15:0] q88_mult(input signed [15:0] a, input signed [15:0] b);
        reg signed [31:0] prod;
        begin
            prod = a * b;
            q88_mult = prod[23:8]; // Truncate/scale to Q8.8
        end
    endfunction

    // Internal Registers
    reg signed [15:0] v_dend;
    reg signed [15:0] v_soma;
    reg [2:0]         refr_counter;
    reg               nmda_active;

    // Combinatorial signals
    wire signed [15:0] i_axial;
    wire signed [15:0] v_diff;
    wire signed [15:0] nmda_drive;
    wire signed [15:0] plateau_diff;
    wire signed [15:0] alpha_d_eff;
    wire signed [15:0] v_dend_decayed;
    wire signed [15:0] v_soma_decayed;
    wire               shunt_active;

    assign v_diff = v_dend - v_soma;
    assign i_axial = q88_mult(g_axial, v_diff);
    assign plateau_diff = v_plateau - v_dend;
    assign nmda_drive = nmda_active ? q88_mult(g_nmda, plateau_diff) : `FP_ZERO;
    assign alpha_d_eff = nmda_active ? `FP_ALPHA_PLATEAU : `FP_ALPHA_DEND;

    assign v_dend_decayed = q88_mult(alpha_d_eff, v_dend);
    assign v_soma_decayed = q88_mult(`FP_ALPHA_SOMA, v_soma);

    // Perisomatic shunting clamp: active when inhibitory drive exceeds threshold
    assign shunt_active = (i_syn_soma_inh > 16'sh0010);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_dend          <= `FP_ZERO;
            v_soma          <= `FP_ZERO;
            refr_counter    <= 3'b0;
            nmda_active     <= 1'b0;
            spike_out       <= 1'b0;
            v_dend_out      <= `FP_ZERO;
            v_soma_out      <= `FP_ZERO;
            nmda_active_out <= 1'b0;
        end else if (step_tick) begin
            // ----------------------------------------------------------------
            // 1. Dendritic Compartment State Machine & NMDA Dynamics
            // ----------------------------------------------------------------
            if (v_dend >= theta_dend) begin
                nmda_active <= 1'b1;
            end else if (v_dend < (theta_dend - 16'sh0020)) begin
                // Hysteresis release below threshold
                nmda_active <= 1'b0;
            end

            // Update dendritic voltage: decay + incoming synapse + regenerative NMDA
            v_dend <= v_dend_decayed + i_syn_dend + nmda_drive;

            // ----------------------------------------------------------------
            // 2. Somatic Compartment State Machine & Perisomatic Shunting
            // ----------------------------------------------------------------
            if (refr_counter > 0) begin
                refr_counter <= refr_counter - 1'b1;
                v_soma       <= `FP_ZERO; // Clamped at reset
                spike_out    <= 1'b0;
            end else begin
                if (shunt_active) begin
                    // Perisomatic chloride shunting: aggressively pulls soma to resting potential
                    v_soma    <= q88_mult(v_soma_decayed, (16'sh0100 - `FP_SHUNT_EFF));
                    spike_out <= 1'b0;
                end else begin
                    // Normal axial integration
                    if ((v_soma_decayed + i_axial) >= v_thresh_soma) begin
                        spike_out    <= 1'b1;
                        v_soma       <= `FP_ZERO; // Somatic reset
                        refr_counter <= 3'd2;     // 2-step refractory period
                    end else begin
                        spike_out <= 1'b0;
                        v_soma    <= v_soma_decayed + i_axial;
                    end
                end
            end

            // Register diagnostic outputs
            v_dend_out      <= v_dend;
            v_soma_out      <= v_soma;
            nmda_active_out <= nmda_active;
        end else begin
            // Clear single-cycle spike pulse
            spike_out <= 1'b0;
        end
    end

endmodule
