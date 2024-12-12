#!/bin/bash
#
total_jobs=1
job_counter=0
start_time=$(date +%s)
update_progress() {
  local progress=$(( job_counter * 100 / total_jobs ))
  local bar=$(printf '=%.0s' {1..$((progress / 2))})
  local current_time=$(date +%s)
  local elapsed_time=$((current_time - start_time))
  local avg_time_per_job=$((elapsed_time / (job_counter + 1)))
  local remaining_jobs=$((total_jobs - job_counter - 1))
  local eta=$((remaining_jobs * avg_time_per_job))
  local eta_min=$((eta / 60))
  local eta_sec=$((eta % 60))
  printf '\rProgress: [%-50s] %d%% | ETA: %02d:%02d' "$bar" "$progress" "$eta_min" "$eta_sec"
}
echo -e '\nStarting jobs...'
update_progress
/Users/maxgoetzmann/dev/compile-time-prefetcher-classification/ChampSim/bin/perceptron-multi-multi-no-ship-1core --warmup_instructions=100000000 --simulation_instructions=500000000 --config=/Users/maxgoetzmann/dev/compile-time-prefetcher-classification/ChampSim/config/nopref.ini  -traces /Users/maxgoetzmann/dev/compile-time-prefetcher-classification/ChampSim/traces/403.gcc-16B.champsimtrace.xz > 403.gcc-16B_nopref.out 2>&1
((job_counter++))
update_progress

echo -e '\nAll jobs complete.'
