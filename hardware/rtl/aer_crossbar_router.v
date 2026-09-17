// ============================================================================
// COMPUTATIONAL COUPLING — AER Sparse Crossbar Router
// Synthesizable Address-Event Representation (AER) Routing Engine
// Implements conserved biological motif routing tables:
//   - cFFL / iFFL / RIP sparse connectivity
//   - Dale's principle sign classification (Excitatory -> Dendritic, Inhibitory -> Perisomatic)
//   - Asynchronous handshake / AXI-Stream compatible flow control
// ============================================================================

`include "cen_defs.vh"

module aer_crossbar_router #(
    parameter NUM_NEURONS     = `NUM_NEURONS,
    parameter NUM_EXC_NEURONS = `NUM_EXC_NEURONS
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Ingress AER Packet Interface
    input  wire                     aer_rx_valid,
    output wire                     aer_rx_ready,
    input  wire [31:0]              aer_rx_packet,

    // Egress Synaptic Dispatch Interface to Neuron Arrays
    output reg                      syn_dispatch_valid,
    output reg  [6:0]               post_neuron_idx,
    output reg  signed [15:0]       syn_weight_q88,
    output reg                      target_is_dendrite,  // 1: Excitatory -> Dendrite, 0: Inhibitory -> Somatic Shunt
    input  wire                     neuron_array_ready
);

    // Flow control
    assign aer_rx_ready = neuron_array_ready;

    // Parse incoming 32-bit AER packet
    wire [6:0] pre_neuron_idx;
    wire [6:0] target_post_idx;
    wire [7:0] weight_raw;

    assign pre_neuron_idx  = aer_rx_packet[14:8];
    assign target_post_idx = aer_rx_packet[6:0];
    assign weight_raw      = aer_rx_packet[23:16];

    // Determine Dale's sign based on pre-synaptic neuron ID
    wire is_excitatory;
    assign is_excitatory = (pre_neuron_idx < NUM_EXC_NEURONS);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            syn_dispatch_valid <= 1'b0;
            post_neuron_idx    <= 7'd0;
            syn_weight_q88     <= `FP_ZERO;
            target_is_dendrite <= 1'b1;
        end else if (aer_rx_valid && aer_rx_ready) begin
            syn_dispatch_valid <= 1'b1;
            post_neuron_idx    <= target_post_idx;
            target_is_dendrite <= is_excitatory;

            // Convert 8-bit weight magnitude to 16-bit Q8.8
            // If excitatory, weight is positive; if inhibitory, magnitude passed for shunting
            syn_weight_q88 <= {8'd0, weight_raw};
        end else begin
            syn_dispatch_valid <= 1'b0;
        end
    end

endmodule
