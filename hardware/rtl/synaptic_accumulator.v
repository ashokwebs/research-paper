// ============================================================================
// COMPUTATIONAL COUPLING — Synaptic Accumulator Buffer
// Dual-buffer event accumulator segregating excitatory dendritic drive from
// perisomatic shunting inhibitory drive across 128 physical neuron channels.
// ============================================================================

`include "cen_defs.vh"

module synaptic_accumulator #(
    parameter NUM_NEURONS = `NUM_NEURONS
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     step_tick,

    // Ingress Synaptic Dispatch
    input  wire                     syn_dispatch_valid,
    input  wire [6:0]               post_neuron_idx,
    input  wire signed [15:0]       syn_weight_q88,
    input  wire                     target_is_dendrite,

    // Egress to Neuron Array at step_tick
    output reg  signed [15:0]       i_dend_out [0:NUM_NEURONS-1],
    output reg  signed [15:0]       i_soma_inh_out [0:NUM_NEURONS-1]
);

    // Active accumulation registers
    reg signed [15:0] accum_dend [0:NUM_NEURONS-1];
    reg signed [15:0] accum_soma_inh [0:NUM_NEURONS-1];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                accum_dend[i]     <= `FP_ZERO;
                accum_soma_inh[i] <= `FP_ZERO;
                i_dend_out[i]     <= `FP_ZERO;
                i_soma_inh_out[i] <= `FP_ZERO;
            end
        end else begin
            if (step_tick) begin
                // Latch accumulated values to output and reset accumulators
                for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                    i_dend_out[i]     <= accum_dend[i];
                    i_soma_inh_out[i] <= accum_soma_inh[i];
                    accum_dend[i]     <= `FP_ZERO;
                    accum_soma_inh[i] <= `FP_ZERO;
                end
            end else if (syn_dispatch_valid) begin
                // Accumulate incoming synaptic weight
                if (target_is_dendrite) begin
                    accum_dend[post_neuron_idx] <= accum_dend[post_neuron_idx] + syn_weight_q88;
                end else begin
                    accum_soma_inh[post_neuron_idx] <= accum_soma_inh[post_neuron_idx] + syn_weight_q88;
                end
            end
        end
    end

endmodule
