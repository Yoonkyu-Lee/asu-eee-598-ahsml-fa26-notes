#!/bin/tcsh -f
# CP10 on Apporto:  tcsh run_cp10.csh <W> <S> <period_ns>
# Same flow as run_cp9.csh for the interleaved multiplier (mont_mul_pipe), with DC register retiming.
if ($#argv < 3) then
  echo "usage: run_cp10.csh <W> <S> <period_ns>"; exit 1
endif
if (! $?CP9_ENV) setenv CP9_ENV ~/labs/lab0/work/env.cshrc
source $CP9_ENV
set W = $1
set S = $2
set PERIOD = $3
cd `dirname $0`
mkdir -p run_pipe_W${W}_S${S}_P${PERIOD}
cd run_pipe_W${W}_S${S}_P${PERIOD}
python3 ../gen_vectors.py 1000 vectors.hex > /dev/null
vcs -full64 -sverilog -timescale=1ns/1ps +define+SAIF -pvalue+tb_mont_mul_pipe.W=$W -pvalue+tb_mont_mul_pipe.S=$S ../mont_mul_pipe.sv ../tb_mont_mul_pipe.sv -o simv -l vcs.log > /dev/null
./simv +PERIOD=$PERIOD -l sim.log | grep -E "RESULT|MISMATCH|timeout"
vcd2saif -input dut.vcd -output mont_mul.saif > vcd2saif.log
rm -f dut.vcd
setenv W $W
setenv S $S
setenv PERIOD $PERIOD
dc_shell -f ../syn_pipe.tcl -output_log_file dc.log > /dev/null
cat summary.txt
grep -E "Total cell area" area.rpt
grep -E "Total Dynamic Power|Cell Leakage Power" power.rpt
