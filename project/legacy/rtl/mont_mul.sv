// CP9: 254-bit Montgomery multiplier for the BN254 base field, the multiply at the heart of the
// proof-verification unit (architecture board, L2). Computes y = a * b * 2^-256 mod p for a, b < p.
//
// Word-serial CIOS (coarsely integrated operand scanning) with word width W:
//   repeat K = 256 / W times:  u = T + a_i * b;  m = (u mod 2^W) * NP mod 2^W;  T = (u + m * p) / 2^W
//   then one cycle of conditional subtraction (T < 2p because p < 2^256 / 4).
// One iteration per clock, so a multiplication takes K + 1 cycles. W is the "row width" of CP8's A_w:
// W = 256 is one iteration (largest, shortest), W = 128 two, W = 64 four.
// The products use plain `*` so Design Compiler picks DesignWare multiplier architectures.
module mont_mul #(
  parameter int W = 128
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,   // pulse with a, b valid; ignored while busy
  input  logic [255:0] a,
  input  logic [255:0] b,
  output logic         busy,
  output logic         done,    // one-cycle pulse with y valid
  output logic [255:0] y
);
  localparam int K = 256 / W;
  // BN254 base-field modulus p and NP = -p^-1 mod 2^W
  localparam logic [255:0] P = 256'h30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47;
  localparam logic [255:0] NP_FULL = 256'hf57a22b791888c6bd8afcbd01833da809ede7d651eca6ac987d20782e4866389;
  localparam logic [W-1:0] NP = NP_FULL[W-1:0];

  logic [255:0] a_q, b_q;       // a shifts right by W each iteration; b is held
  logic [256:0] t_q;            // T < 2p < 2^255, one spare bit
  logic [$clog2(K+1):0] cnt_q;
  logic         fin_q;          // final subtraction cycle

  // one CIOS iteration (combinational row)
  logic [W+256:0] u, s;
  logic [W-1:0]   m;
  logic [256:0]   t_next;
  always_comb begin
    u = {{W{1'b0}}, t_q} + a_q[W-1:0] * b_q;
    m = u[W-1:0] * NP;          // truncated to W bits
    s = u + m * P;              // low W bits are zero by construction
    t_next = s[W+256:W];
  end

  logic [256:0] t_sub;
  assign t_sub = t_q - {1'b0, P};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_q <= '0; b_q <= '0; t_q <= '0; cnt_q <= '0; fin_q <= 1'b0;
      busy <= 1'b0; done <= 1'b0; y <= '0;
    end else begin
      done <= 1'b0;
      if (start && !busy) begin
        a_q <= a; b_q <= b; t_q <= '0; cnt_q <= K[$clog2(K+1):0]; fin_q <= 1'b0; busy <= 1'b1;
      end else if (busy && !fin_q) begin
        t_q <= t_next;
        a_q <= a_q >> W;
        cnt_q <= cnt_q - 1'b1;
        if (cnt_q == 1) fin_q <= 1'b1;
      end else if (fin_q) begin
        y <= t_sub[256] ? t_q[255:0] : t_sub[255:0];  // borrow means T < p
        fin_q <= 1'b0; busy <= 1'b0; done <= 1'b1;
      end
    end
  end
endmodule
