// CP10: interleaved, pipelined version of mont_mul (same math: y = a * b * 2^-256 mod p, word-serial CIOS).
// S independent multiplications take turns on one row datapath, one turn per clock (slot = cycle mod S).
// The row result goes through S register stages and comes back exactly on the same slot's next turn, so
// Design Compiler's retiming can spread those S registers through the multiplier trees.
// On a slot's turn: an empty slot takes a new multiplication and issues its first row at once (T = 0);
// a busy slot issues its next row; a slot whose K rows are done outputs y and, in the same turn, can take
// the next multiplication. With a steady input stream each multiplication costs K = 256/W turns.
module mont_mul_pipe #(
  parameter int W = 64,
  parameter int S = 3,
  parameter int TAGW = 16
) (
  input  logic            clk,
  input  logic            rst_n,
  input  logic            in_valid,
  output logic            in_ready,   // this cycle's slot can take a multiplication
  input  logic [255:0]    a,
  input  logic [255:0]    b,
  input  logic [TAGW-1:0] in_tag,
  output logic            out_valid,
  output logic [255:0]    y,
  output logic [TAGW-1:0] out_tag
);
  localparam int K = 256 / W;
  localparam int SW = (S > 1) ? $clog2(S) : 1;
  localparam int CW = $clog2(K + 1);
  localparam logic [255:0] P = 256'h30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47;
  localparam logic [255:0] NP_FULL = 256'hf57a22b791888c6bd8afcbd01833da809ede7d651eca6ac987d20782e4866389;
  localparam logic [W-1:0] NP = NP_FULL[W-1:0];

  // per-slot state
  logic [S-1:0]      busy;
  logic [255:0]      a_q [S];
  logic [255:0]      b_q [S];
  logic [CW-1:0]     cnt [S];      // rows still to issue
  logic [TAGW-1:0]   tag [S];
  logic [SW-1:0]     slot;

  // row pipeline: S stages, result returns on the same slot's next turn
  logic [256:0] pipe [S];
  logic [256:0] t_back;
  assign t_back = pipe[S-1];

  logic finish, load, issue;
  assign finish = busy[slot] && (cnt[slot] == 0);
  assign in_ready = !busy[slot] || finish;
  assign load = in_valid && in_ready;
  assign issue = load || (busy[slot] && (cnt[slot] != 0));

  // one CIOS row on the selected operands
  logic [256:0]   t_in;
  logic [W-1:0]   a_word;
  logic [255:0]   b_in;
  logic [W+256:0] u, s;
  logic [W-1:0]   m;
  logic [256:0]   t_next;
  always_comb begin
    t_in   = load ? '0 : t_back;
    a_word = load ? a[W-1:0] : a_q[slot][W-1:0];
    b_in   = load ? b : b_q[slot];
    u = {{W{1'b0}}, t_in} + a_word * b_in;
    m = u[W-1:0] * NP;
    s = u + m * P;
    t_next = s[W+256:W];
  end

  logic [256:0] t_sub;
  assign t_sub = t_back - {1'b0, P};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      slot <= '0; busy <= '0; out_valid <= 1'b0; y <= '0; out_tag <= '0;
      for (int i = 0; i < S; i++) begin
        a_q[i] <= '0; b_q[i] <= '0; cnt[i] <= '0; tag[i] <= '0; pipe[i] <= '0;
      end
    end else begin
      slot <= (slot == S - 1) ? '0 : slot + 1'b1;
      pipe[0] <= t_next;
      for (int i = 1; i < S; i++) pipe[i] <= pipe[i-1];

      out_valid <= finish;
      if (finish) begin
        y <= t_sub[256] ? t_back[255:0] : t_sub[255:0];
        out_tag <= tag[slot];
        busy[slot] <= 1'b0;
      end
      if (load) begin
        a_q[slot] <= a >> W;
        b_q[slot] <= b;
        cnt[slot] <= CW'(K - 1);
        tag[slot] <= in_tag;
        busy[slot] <= 1'b1;
      end else if (issue) begin
        a_q[slot] <= a_q[slot] >> W;
        cnt[slot] <= cnt[slot] - 1'b1;
      end
    end
  end
endmodule
