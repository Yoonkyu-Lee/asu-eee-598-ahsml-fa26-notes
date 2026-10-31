// CP10 testbench for mont_mul_pipe: streams vectors.hex in as fast as in_ready allows, checks every result
// by its tag (the vector index), and prints "RESULT W=.. S=.. vectors=.. errors=.. cycles_per_mul=..".
// With +define+SAIF it dumps the DUT's switching to dut.vcd (run_cp10.csh converts it to SAIF).
`timescale 1ns/1ps
module tb_mont_mul_pipe;
  parameter int W = 64;
  parameter int S = 3;
  parameter int NMAX = 4096;

  logic clk = 0, rst_n = 0, in_valid = 0, in_ready, out_valid;
  logic [255:0] a = '0, b = '0, y;
  logic [15:0] in_tag = '0, out_tag;
  logic [255:0] va [NMAX], vb [NMAX], vy [NMAX];
  bit seen [NMAX];
  int n, errors, fd, sent, got, cycles;
  real period = 2.0;  // ns, +PERIOD=<ns>
  initial void'($value$plusargs("PERIOD=%f", period));

  mont_mul_pipe #(.W(W), .S(S)) dut (.*);
  always #(period / 2.0) clk = ~clk;

  // checker
  always @(posedge clk) if (rst_n && out_valid) begin
    if (y !== vy[out_tag] || seen[out_tag]) begin
      errors++;
      if (errors <= 5) $display("MISMATCH tag %0d: got=%h want=%h", out_tag, y, vy[out_tag]);
    end
    seen[out_tag] = 1;
    got++;
  end

  initial begin
    fd = $fopen("vectors.hex", "r");
    if (fd == 0) $fatal(1, "vectors.hex not found");
    n = 0;
    while (n < NMAX && $fscanf(fd, "%h %h %h\n", va[n], vb[n], vy[n]) == 3) n++;
    $fclose(fd);
    errors = 0; sent = 0; got = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
`ifdef SAIF
    $dumpfile("dut.vcd");
    $dumpvars(0, tb_mont_mul_pipe.dut);
`endif
    cycles = 0;
    while (got < n) begin
      @(negedge clk);
      if (sent < n) begin
        in_valid = 1; a = va[sent]; b = vb[sent]; in_tag = sent[15:0];
      end else in_valid = 0;
      @(posedge clk);
      cycles++;
      if (in_valid && in_ready) sent++;
      if (cycles > 100 * n + 1000) $fatal(1, "timeout: sent=%0d got=%0d", sent, got);
    end
`ifdef SAIF
    $dumpflush;
`endif
    $display("RESULT W=%0d S=%0d vectors=%0d errors=%0d cycles_per_mul=%0.2f", W, S, n, errors, real'(cycles) / n);
    $finish;
  end
endmodule
