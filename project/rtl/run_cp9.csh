#!/bin/tcsh -f
# CP9 on Apporto (ECEE CAD Desktop - RHEL8):  ./run_cp9.csh <W> <period_ns>
# Simulates mont_mul with VCS on reference vectors (records switching activity), then synthesizes it with
# Design Compiler + FreePDK45. Tool setup comes from $CP9_ENV (default: the lab env script that puts
# VCS, Verdi and DC on PATH and sets PDK_DIR); it is not stored in this public repo.
if ($#argv < 2) then
  echo "usage: run_cp9.csh <W> <period_ns>"; exit 1
endif
if (! $?CP9_ENV) setenv CP9_ENV ~/labs/lab0/work/env.cshrc
source $CP9_ENV
set W = $1
set PERIOD = $2
set here = `dirname $0`
cd $here
mkdir -p run_W${W}_P${PERIOD}
cd run_W${W}_P${PERIOD}
python3 ../gen_vectors.py 1000 vectors.hex > /dev/null
vcs -full64 -sverilog +define+SAIF -pvalue+tb_mont_mul.W=$W ../mont_mul.sv ../tb_mont_mul.sv -o simv -l vcs.log > /dev/null
./simv +PERIOD=$PERIOD -l sim.log | grep -E "RESULT|MISMATCH"
setenv W $W
setenv PERIOD $PERIOD
dc_shell -f ../syn.tcl -output_log_file dc.log > /dev/null
cat summary.txt
grep -E "Total Dynamic Power|Cell Leakage Power" power.rpt
