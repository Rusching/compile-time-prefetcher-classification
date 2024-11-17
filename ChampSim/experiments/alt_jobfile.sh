#!/bin/bash
#
total_jobs=35
job_counter=0
update_progress() {
  local progress=$(( job_counter * 100 / total_jobs ))
  local bar=$(printf '=%.0s' {1..$((progress / 2))})
  printf '\rProgress: [%-50s] %d%%' "$bar" "$progress"
}
echo -e '\nStarting jobs...'
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/Pythia/config/nopref.ini  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=spp_dev2 --config=/Users/maxgoetzmann/dev/Pythia/config/spp_dev2.ini  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_spp.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=bingo --config=/Users/maxgoetzmann/dev/Pythia/config/bingo.ini  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_bingo.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=mlop --config=/Users/maxgoetzmann/dev/Pythia/config/mlop.ini  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_mlop.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/Users/maxgoetzmann/dev/Pythia/config/pythia.ini  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_pythia.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/Pythia/config/nopref.ini --dram_io_freq=150  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref_MTPS150.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=spp_dev2 --config=/Users/maxgoetzmann/dev/Pythia/config/spp_dev2.ini --dram_io_freq=150  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_spp_MTPS150.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=bingo --config=/Users/maxgoetzmann/dev/Pythia/config/bingo.ini --dram_io_freq=150  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_bingo_MTPS150.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=mlop --config=/Users/maxgoetzmann/dev/Pythia/config/mlop.ini --dram_io_freq=150  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_mlop_MTPS150.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/Users/maxgoetzmann/dev/Pythia/config/pythia.ini --dram_io_freq=150  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_pythia_MTPS150.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/Pythia/config/nopref.ini --dram_io_freq=300  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref_MTPS300.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=spp_dev2 --config=/Users/maxgoetzmann/dev/Pythia/config/spp_dev2.ini --dram_io_freq=300  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_spp_MTPS300.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=bingo --config=/Users/maxgoetzmann/dev/Pythia/config/bingo.ini --dram_io_freq=300  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_bingo_MTPS300.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=mlop --config=/Users/maxgoetzmann/dev/Pythia/config/mlop.ini --dram_io_freq=300  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_mlop_MTPS300.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/Users/maxgoetzmann/dev/Pythia/config/pythia.ini --dram_io_freq=300  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_pythia_MTPS300.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/Pythia/config/nopref.ini --dram_io_freq=600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref_MTPS600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=spp_dev2 --config=/Users/maxgoetzmann/dev/Pythia/config/spp_dev2.ini --dram_io_freq=600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_spp_MTPS600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=bingo --config=/Users/maxgoetzmann/dev/Pythia/config/bingo.ini --dram_io_freq=600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_bingo_MTPS600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=mlop --config=/Users/maxgoetzmann/dev/Pythia/config/mlop.ini --dram_io_freq=600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_mlop_MTPS600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/Users/maxgoetzmann/dev/Pythia/config/pythia.ini --dram_io_freq=600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_pythia_MTPS600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/Pythia/config/nopref.ini --dram_io_freq=1200  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref_MTPS1200.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=spp_dev2 --config=/Users/maxgoetzmann/dev/Pythia/config/spp_dev2.ini --dram_io_freq=1200  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_spp_MTPS1200.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=bingo --config=/Users/maxgoetzmann/dev/Pythia/config/bingo.ini --dram_io_freq=1200  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_bingo_MTPS1200.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=mlop --config=/Users/maxgoetzmann/dev/Pythia/config/mlop.ini --dram_io_freq=1200  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_mlop_MTPS1200.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/Users/maxgoetzmann/dev/Pythia/config/pythia.ini --dram_io_freq=1200  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_pythia_MTPS1200.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/Pythia/config/nopref.ini --dram_io_freq=4800  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref_MTPS4800.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=spp_dev2 --config=/Users/maxgoetzmann/dev/Pythia/config/spp_dev2.ini --dram_io_freq=4800  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_spp_MTPS4800.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=bingo --config=/Users/maxgoetzmann/dev/Pythia/config/bingo.ini --dram_io_freq=4800  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_bingo_MTPS4800.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=mlop --config=/Users/maxgoetzmann/dev/Pythia/config/mlop.ini --dram_io_freq=4800  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_mlop_MTPS4800.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/Users/maxgoetzmann/dev/Pythia/config/pythia.ini --dram_io_freq=4800  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_pythia_MTPS4800.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/Pythia/config/nopref.ini --dram_io_freq=9600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref_MTPS9600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=spp_dev2 --config=/Users/maxgoetzmann/dev/Pythia/config/spp_dev2.ini --dram_io_freq=9600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_spp_MTPS9600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=bingo --config=/Users/maxgoetzmann/dev/Pythia/config/bingo.ini --dram_io_freq=9600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_bingo_MTPS9600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=mlop --config=/Users/maxgoetzmann/dev/Pythia/config/mlop.ini --dram_io_freq=9600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_mlop_MTPS9600.out 2>&1
((job_counter++))
update_progress
/Users/maxgoetzmann/dev/Pythia/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --l2c_prefetcher_types=scooby --config=/Users/maxgoetzmann/dev/Pythia/config/pythia.ini --dram_io_freq=9600  -traces /Users/maxgoetzmann/dev/Pythia/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_pythia_MTPS9600.out 2>&1
((job_counter++))
update_progress

echo -e '\nAll jobs complete.'
