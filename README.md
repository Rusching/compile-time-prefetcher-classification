# Compile Time Pretecher Classification  

This project integrates a prefetcher classification system into the Pythia ChampSim implementation, designed for UChicago's CMSC 33100 Advanced Operating Systems.

---

## Installation  

To streamline the setup process, two new scripts have been added to the ChampSim directory. Follow these steps to get started:  

1. Navigate to the ChampSim directory:  

   ```bash  
   cd ChampSim  
   ```  

2. Run the setup script:  

   ```bash  
   ./setup_champsim.sh  
   ```  

   **Note**: If you want to test the simulator quickly without downloading large trace files (dozens of gigabytes), use the following command instead:  

   ```bash  
   ./setup_champsim.sh --test  
   ```  

---

## Running Experiments  

To execute experiments, use the command:  

```bash  
./run_experiment.sh [--1C_BASE | --1C_DRAM | --4C | --all] <tlist_file>  
```  

### Options:  

- **Default:** If no flags are provided, the script runs the 1C_BASE case.  
- **Custom Traces:** Use `<tlist_file>` to specify the trace list.  
  - For quick tests, supply `demo_short.tlist` (if `--test` was used during setup).  
  - For full trace coverage, use `demo_long.tlist` or a custom trace list.  

**Tip**: Even when running minimal test cases (e.g., `demo_short.tlist`), the simulator may take at least 30 minutes to complete.  

---

## Analyzing Results  

After running experiments, the simulator generates output files named `rollup_*.csv`.  

### Steps to Analyze:  

1. Export the CSV files to your preferred data processing tool (e.g., Python Pandas, Microsoft Excel, Apple Numbers, etc.).  
2. Use these insights to evaluate prefetcher performance and compare experiment results.  

---
