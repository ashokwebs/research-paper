// ============================================================================
// COMPUTATIONAL COUPLING — CoupledEventNet (CEN) Hardware Package Definitions
// Target Technology: Synthesizable FPGA (AMD Xilinx UltraScale+) & ASIC (TSMC 28nm)
// Arithmetic: Fixed-Point Signed Q8.8 (16-bit)
// ============================================================================

`ifndef CEN_DEFS_VH
`define CEN_DEFS_VH

// Data & Address Bit Widths
`define DATA_WIDTH         16          // 16-bit signed Q8.8 fixed-point
`define ADDR_WIDTH         7           // 7-bit neuron index (supports up to 128 neurons)
`define NUM_NEURONS        128         // Core neuron count
`define NUM_EXC_NEURONS    102         // 80% Excitatory
`define NUM_INH_NEURONS    26          // 20% Inhibitory

// Fixed-Point Q8.8 Scaling (1.0 = 256, 0.5 = 128)
`define FP_ONE             16'sh0100   // +1.0 in Q8.8
`define FP_ZERO            16'sh0000   //  0.0 in Q8.8
`define FP_HALF            16'sh0080   // +0.5 in Q8.8
`define FP_V_THRESH_SOMA   16'sh0100   // 1.00 somatic firing threshold
`define FP_THETA_DEND      16'sh0073   // 0.45 dendritic NMDA activation threshold
`define FP_V_PLATEAU       16'sh00D9   // 0.85 NMDA plateau target voltage
`define FP_G_NMDA          16'sh005A   // 0.35 regenerative NMDA plateau conductance
`define FP_G_AXIAL         16'sh0040   // 0.25 forward axial coupling conductance
`define FP_ALPHA_SOMA      16'sh00F3   // exp(-1/20) = 0.9512 in Q8.8 (~243/256)
`define FP_ALPHA_DEND      16'sh00F9   // exp(-1/40) = 0.9753 in Q8.8 (~250/256)
`define FP_ALPHA_PLATEAU   16'sh00FE   // exp(-1/150) = 0.9933 in Q8.8 (~254/256)
`define FP_SHUNT_EFF       16'sh00F3   // 0.95 shunting inhibition factor

// AER (Address-Event Representation) Packet Format (32-bit)
// [31]: Valid event bit
// [30:24]: 7-bit Timestamp / Time-bin tag
// [23:16]: Reserved / Polarity
// [15:8]: Pre-synaptic Neuron ID / Input Channel
// [7:0]: Post-synaptic Target Neuron ID
`define AER_PACKET_WIDTH   32

`endif // CEN_DEFS_VH
