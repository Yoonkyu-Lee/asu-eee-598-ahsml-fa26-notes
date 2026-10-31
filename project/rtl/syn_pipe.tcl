# CP10: Design Compiler synthesis of mont_mul_pipe with register retiming.
# Run from rtl/run_pipe_W<W>_S<S>_P<period>/ via run_cp10.csh (env W, S, PERIOD, PDK_DIR).
set W $env(W)
set S $env(S)
set PERIOD $env(PERIOD)
set search_path [concat $search_path $env(PDK_DIR)/osu_soc/lib/files]
set target_library gscl45nm.db
set synthetic_library dw_foundation.sldb
set link_library [concat * $target_library $synthetic_library]
set_host_options -max_cores 4

analyze -format sverilog ../mont_mul_pipe.sv
elaborate mont_mul_pipe -parameters "W=$W,S=$S"
current_design [get_designs mont_mul_pipe*]
link
uniquify

create_clock -name clk -period $PERIOD [get_ports clk]
set_input_delay 0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0 -clock clk [all_outputs]
set_max_fanout 16 [current_design]
set_optimize_registers true -designs [current_design]

compile_ultra -retime -no_autoungroup

if {[file exists mont_mul.saif]} {
  read_saif -input mont_mul.saif -instance_name tb_mont_mul_pipe/dut -auto_map_names
}
redirect area.rpt   { report_area -hierarchy }
redirect timing.rpt { report_timing -max_paths 1 }
redirect power.rpt  { report_power }
redirect saif.rpt   { report_saif }
set path [get_timing_paths -max_paths 1]
set fh [open summary.txt w]
puts $fh "SUMMARY W=$W S=$S period_ns=$PERIOD slack_ns=[get_attribute $path slack] arrival_ns=[get_attribute $path arrival]"
close $fh
exit
