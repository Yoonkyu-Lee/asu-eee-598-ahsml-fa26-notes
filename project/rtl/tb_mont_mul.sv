// CP9 testbench: feeds vectors.hex (from gen_vectors.py) through mont_mul back to back and compares.
// Prints one summary line: "RESULT W=.. vectors=.. errors=.. cycles_per_mul=..".
// With +define+SAIF it dumps the DUT's switching during the timed run to dut.vcd; run_cp9.csh turns that
// into a SAIF file (vcd2saif) for Design Compiler's power report.
`timescale 1ns/1ps
module tb_mont_mul;
  parameter int W = 128;
  parameter int NMAX = 4096;

  logic clk = 0, rst_n = 0, start = 0;
  logic [255:0] a, b, y;
  logic busy, done;
  logic [255:0] va [NMAX], vb [NMAX], vy [NMAX];
  int n, errors, fd, cycles;
  real period = 2.0;  // ns, +PERIOD=<ns>; match the synthesis clock so the SAIF toggle rates fit it
  initial void'($value$plusargs("PERIOD=%f", period));

  mont_mul #(.W(W)) dut (.*);
  always #(period / 2.0) clk = ~clk;
  always @(posedge clk) if (rst_n) cycles++;

  initial begin
    fd = $fopen("vectors.hex", "r");
    if (fd == 0) $fatal(1, "vectors.hex not found");
    n = 0;
    while (n < NMAX && $fscanf(fd, "%h %h %h\n", va[n], vb[n], vy[n]) == 3) n++;
    $fclose(fd);
    errors = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
`ifdef SAIF
    $dumpfile("dut.vcd");
    $dumpvars(0, tb_mont_mul.dut);
`endif
    cycles = 0;
    for (int i = 0; i < n; i++) begin
      @(negedge clk); a = va[i]; b = vb[i]; start = 1;
      @(negedge clk); start = 0;
      wait (done);
      if (y !== vy[i]) begin
        errors++;
        if (errors <= 5) $display("MISMATCH %0d: a=%h b=%h got=%h want=%h", i, va[i], vb[i], y, vy[i]);
      end
      @(posedge clk);
    end
`ifdef SAIF
    $dumpflush;
`endif
    $display("RESULT W=%0d vectors=%0d errors=%0d cycles_per_mul=%0.2f", W, n, errors, real'(cycles) / n);
    $finish;
  end
endmodule
