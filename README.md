# Compile Time Pretecher Classification  

This project integrates a prefetcher classification system into the Pythia ChampSim implementation, designed for UChicago's CMSC 33100 Advanced Operating Systems.

---

## Installation  

1. Prerequesites

Have `perl` and the `opencv` C++ library installed on your machine. The following commands can be used to verify this:

    bash
# perl
perl -v

# opencv
ls /usr/lib | grep opencv
```

To streamline the setup process, two new scripts have been added to the ChampSim directory. Follow these steps to get started:  

2. Navigate to the ChampSim directory:  

   ```bash  
   cd ChampSim  
   ```  

3. Run the setup script:  

   ```bash  
   ./setup_champsim.sh  
   ```  

   **Note**: If you want to test the simulator quickly without downloading large trace files (dozens of gigabytes), use the following command instead:  

   ```bash  
   ./setup_champsim.sh --test  
   ```  

---

## Running Experiments  

To test experiments execution, use the following command:  

```bash
./run_experiment.sh demo_short.tlist
```

### General Experiment Command  

```bash
./run_experiment.sh [--1C_BASE | --1C_DRAM | --4C | --all] <tlist_file>
```

### Options  

- **Default Behavior:**  
  If no flags are provided, the script defaults to the `--1C_BASE` case.  

- **Custom Trace Lists:**  
  - Use `<tlist_file>` to specify the trace list to run.  
  - For quick tests, use `demo_short.tlist` (if the `--test` flag was included during setup).  
  - For full trace coverage, use `demo_long.tlist` or provide your own custom trace list.  

**Tip:**  
Anecdotally, even minimal test cases (e.g., `demo_short.tlist`) take at least 30 minutes or more to complete due to simulator runtime.

---

## Analyzing Results  

After running experiments, the simulator generates output files named `rollup_*.csv`.  

### Steps to Analyze:  

1. Export the CSV files to your preferred data processing tool (e.g., Python Pandas, Microsoft Excel, Apple Numbers, etc.).  
2. Use these insights to evaluate prefetcher performance and compare experiment results.  

---
