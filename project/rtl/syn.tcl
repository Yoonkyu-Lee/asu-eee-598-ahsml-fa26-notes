# CP9: Design Compiler synthesis of mont_mul with FreePDK45 (OSU gscl45nm) + DesignWare.
# Run from rtl/run_W<W>/ via run_cp9.csh, which sets env W, PERIOD (ns) and PDK_DIR.
# Writes area.rpt, timing.rpt, power.rpt and one line to summary.txt.
set W $env(W)
set PERIOD $env(PERIOD)
set search_path [concat $search_path $env(PDK_DIR)/osu_soc/lib/files]
set target_library gscl45nm.db
set synthetic_library dw_foundation.sldb
set link_library [concat * $target_library $synthetic_library]
set_host_options -max_cores 8

analyze -format sverilog ../mont_mul.sv
elaborate mont_mul -parameters "W=$W"
current_design [get_designs mont_mul*]
link
uniquify

create_clock -name clk -period $PERIOD [get_ports clk]
set_input_delay 0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0 -clock clk [all_outputs]
set_max_fanout 16 [current_design]

compile_ultra -no_autoungroup

set saif "mont_mul.saif"
if {[file exists $saif]} {
  read_saif -input $saif -instance_name tb_mont_mul/dut -auto_map_names
}

redirect area.rpt   { report_area -hierarchy }
redirect timing.rpt { report_timing -max_paths 1 }
redirect power.rpt  { report_power }
redirect saif.rpt   { report_saif }

set area [get_attribute [current_design] area]
set path [get_timing_paths -max_paths 1]
set slack [get_attribute $path slack]
set arrival [get_attribute $path arrival]
set fh [open summary.txt w]
puts $fh "SUMMARY W=$W period_ns=$PERIOD area_um2=$area slack_ns=$slack arrival_ns=$arrival"
close $fh
exit
