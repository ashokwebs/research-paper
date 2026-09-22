// ============================================================================
// COMPUTATIONAL COUPLING — Top-Level CoupledEventNet Neuromorphic Accelerator
// Synthesizable Verilog RTL Top-Level Core
// Integrates:
//   1. AER Input Stream Decoder & Sparse Motif Crossbar Router
//   2. Dual-Compartment Synaptic Accumulator Buffer
//   3. 128-Channel Vectorized Two-Compartment Active Dendritic LIF Cores
//   4. Outbound AER Spike Encoder & Hardware SynOps / Event Telemetry Counters
// ============================================================================

`include "cen_defs.vh"

module cen_accelerator_top #(
    parameter NUM_NEURONS     = `NUM_NEURONS,
    parameter NUM_EXC_NEURONS = `NUM_EXC_NEURONS
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Step Timing Tick (e.g. 1 kHz / 1 ms simulation tick from PLL)
    input  wire                     step_tick,

    // Inbound AXI-Stream / AER Ingress Interface
    input  wire                     s_axis_aer_tvalid,
    output wire                     s_axis_aer_tready,
    input  wire [31:0]              s_axis_aer_tdata,

    // Outbound AXI-Stream / AER Egress Interface
    output reg                      m_axis_aer_tvalid,
    input  wire                     m_axis_aer_tready,
    output reg  [31:0]              m_axis_aer_tdata,

    // Hardware Telemetry & Performance Counters
    output reg  [31:0]              total_synops_counter,
    output reg  [31:0]              total_spikes_counter,
    output reg  [31:0]              total_plateau_counter
);

    // Internal Wires
    wire                    syn_dispatch_valid;
    wire [6:0]              post_neuron_idx;
    wire signed [15:0]      syn_weight_q88;
    wire                    target_is_dendrite;

    // Flat packed buses (see synaptic_accumulator): channel n is bits [16*n +: 16].
    wire [16*NUM_NEURONS-1:0] i_dend;
    wire [16*NUM_NEURONS-1:0] i_soma_inh;
    wire [NUM_NEURONS-1:0]  spikes_vector;
    wire [NUM_NEURONS-1:0]  plateau_vector;

    // ------------------------------------------------------------------------
    // 1. AER Ingress Crossbar Router
    // ------------------------------------------------------------------------
    aer_crossbar_router #(
        .NUM_NEURONS(NUM_NEURONS),
        .NUM_EXC_NEURONS(NUM_EXC_NEURONS)
    ) u_router (
        .clk(clk),
        .rst_n(rst_n),
        .aer_rx_valid(s_axis_aer_tvalid),
        .aer_rx_ready(s_axis_aer_tready),
        .aer_rx_packet(s_axis_aer_tdata),
        .syn_dispatch_valid(syn_dispatch_valid),
        .post_neuron_idx(post_neuron_idx),
        .syn_weight_q88(syn_weight_q88),
        .target_is_dendrite(target_is_dendrite),
        .neuron_array_ready(1'b1)
    );

    // ------------------------------------------------------------------------
    // 2. Synaptic Accumulation Engine
    // ------------------------------------------------------------------------
    synaptic_accumulator #(
        .NUM_NEURONS(NUM_NEURONS)
    ) u_accumulator (
        .clk(clk),
        .rst_n(rst_n),
        .step_tick(step_tick),
        .syn_dispatch_valid(syn_dispatch_valid),
        .post_neuron_idx(post_neuron_idx),
        .syn_weight_q88(syn_weight_q88),
        .target_is_dendrite(target_is_dendrite),
        .i_dend_out(i_dend),
        .i_soma_inh_out(i_soma_inh)
    );

    // ------------------------------------------------------------------------
    // 3. Array of Two-Compartment Active Dendritic LIF Neuron Cores
    // ------------------------------------------------------------------------
    genvar n;
    generate
        for (n = 0; n < NUM_NEURONS; n = n + 1) begin : gen_neuron_cores
            dendritic_lif_core u_core (
                .clk(clk),
                .rst_n(rst_n),
                .step_tick(step_tick),
                .i_syn_dend(i_dend[16*n +: 16]),
                .i_syn_soma_inh(i_soma_inh[16*n +: 16]),
                .v_thresh_soma(`FP_V_THRESH_SOMA),
                .theta_dend(`FP_THETA_DEND),
                .v_plateau(`FP_V_PLATEAU),
                .g_nmda(`FP_G_NMDA),
                .g_axial(`FP_G_AXIAL),
                .v_dend_out(),
                .v_soma_out(),
                .nmda_active_out(plateau_vector[n]),
                .spike_out(spikes_vector[n])
            );
        end
    endgenerate

    // ------------------------------------------------------------------------
    // 4. Outbound AER Spike Arbiter & Hardware Telemetry Counters
    // ------------------------------------------------------------------------
    reg [6:0] round_robin_idx;
    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axis_aer_tvalid      <= 1'b0;
            m_axis_aer_tdata       <= 32'b0;
            round_robin_idx        <= 7'd0;
            total_synops_counter   <= 32'd0;
            total_spikes_counter   <= 32'd0;
            total_plateau_counter  <= 32'd0;
        end else begin
            // Increment telemetry counters
            if (syn_dispatch_valid) begin
                total_synops_counter <= total_synops_counter + 1'b1;
            end

            // Search for active spike to encode into AER packet
            if (spikes_vector != {NUM_NEURONS{1'b0}} && (!m_axis_aer_tvalid || m_axis_aer_tready)) begin
                m_axis_aer_tvalid <= 1'b1;
                // Encode packet: [31]=1 (valid), [14:8]=neuron_idx
                m_axis_aer_tdata  <= {1'b1, 16'b0, 1'b0, round_robin_idx, 7'b0};
                total_spikes_counter <= total_spikes_counter + 1'b1;
            end else if (m_axis_aer_tready) begin
                m_axis_aer_tvalid <= 1'b0;
            end

            // Tally active dendritic plateaus at step tick
            if (step_tick) begin
                for (k = 0; k < NUM_NEURONS; k = k + 1) begin
                    if (plateau_vector[k]) begin
                        total_plateau_counter <= total_plateau_counter + 1'b1;
                    end
                end
            end
        end
    end

endmodule
