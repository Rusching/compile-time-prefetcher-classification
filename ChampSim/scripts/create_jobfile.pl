#!/usr/bin/perl

use warnings;
use Getopt::Long;
use Trace;
use Exp;

# defaults
my $tlist_file;
my $exp_file;
my $exe;
my $local = "0";
my $ncores = 1;
my $slurm_partition = "slurm_part";
my $exclude_list;
my $include_list;
my $extra;

GetOptions(
    'tlist=s' => \$tlist_file,
    'exp=s' => \$exp_file,
    'exe=s' => \$exe,
    'ncores=s' => \$ncores,
    'local=s' => \$local,
    'exclude=s' => \$exclude_list,
    'partition=s' => \$slurm_partition,
    'exclude=s' => \$exclude_list,
    'include=s' => \$include_list,
    'extra=s' => \$extra,
) or die "Usage: $0 --exe <executable> --exp <exp file> --tlist <trace list>\n";

die "\$PYTHIA_HOME env variable is not defined.\nHave you sourced setvars.sh?\n" unless defined $ENV{'PYTHIA_HOME'};
die "Supply exe\n" unless defined $exe;
die "Supply tlist\n" unless defined $tlist_file;
die "Supply exp\n" unless defined $exp_file;

my $exclude_nodes_list = "";
$exclude_nodes_list = "kratos[$exclude_list]" if defined $exclude_list;
my $include_nodes_list = "";
$include_nodes_list = "kratos[$include_list]" if defined $include_list;

my @trace_info = Trace::parse($tlist_file);
my @exp_info = Exp::parse($exp_file);

if ($ncores == 0) {
    print "have to supply -ncores\n";
    exit 1;
}

# Calculate total number of jobs
my $total_jobs = scalar(@trace_info) * scalar(@exp_info);

# preamble for sbatch script
if ($local eq "0") {
    print "#!/bin/bash -l\n";
    print "#\n";
} else {
    print "#!/bin/bash\n";
    print "#\n";
}

# Bash progress bar and ETA setup
print "total_jobs=$total_jobs\n";
print "job_counter=0\n";
print "start_time=\$(date +%s)\n";  # Record the start time
print "update_progress() {\n";
print "  local progress=\$(( job_counter * 100 / total_jobs ))\n";
print "  local bar=\$(printf '=%.0s' {1..\$((progress / 2))})\n";
print "  local current_time=\$(date +%s)\n";
print "  local elapsed_time=\$((current_time - start_time))\n";
print "  local avg_time_per_job=\$((elapsed_time / (job_counter + 1)))\n";
print "  local remaining_jobs=\$((total_jobs - job_counter - 1))\n";
print "  local eta=\$((remaining_jobs * avg_time_per_job))\n";
print "  local eta_min=\$((eta / 60))\n";
print "  local eta_sec=\$((eta % 60))\n";
print "  printf '\\rProgress: [%-50s] %d%% | ETA: %02d:%02d' \"\$bar\" \"\$progress\" \"\$eta_min\" \"\$eta_sec\"\n";
print "}\n";
print "echo -e '\\nStarting jobs...'\n";
print "update_progress\n";

foreach my $trace (@trace_info) {
    foreach my $exp (@exp_info) {
        my $exp_name = $exp->{"NAME"};
        my $exp_knobs = $exp->{"KNOBS"};
        my $trace_name = $trace->{"NAME"};
        my $trace_input = $trace->{"TRACE"};
        my $trace_knobs = $trace->{"KNOBS"};

        my $cmdline;
        if ($local) {
            $cmdline = "$exe $exp_knobs $trace_knobs -traces $trace_input > ${trace_name}_${exp_name}.out 2>&1";
        } else {
            my $slurm_cmd = "sbatch -p $slurm_partition --mincpus=1";
            $slurm_cmd .= " --nodelist=${include_nodes_list}" if defined $include_list;
            $slurm_cmd .= " --exclude=${exclude_nodes_list}" if defined $exclude_list;
            $slurm_cmd .= " $extra" if defined $extra;
            $slurm_cmd .= " -c $ncores -J ${trace_name}_${exp_name} -o ${trace_name}_${exp_name}.out -e ${trace_name}_${exp_name}.err";
            $cmdline = "$slurm_cmd $ENV{'PYTHIA_HOME'}/wrapper.sh $exe \"$exp_knobs $trace_knobs -traces $trace_input\"";
        }
        
        # Additional hook replace
        $cmdline =~ s/\$\(PYTHIA_HOME\)/$ENV{'PYTHIA_HOME'}/g;
        $cmdline =~ s/\$\(EXP\)/$exp_name/g;
        $cmdline =~ s/\$\(TRACE\)/$trace_name/g;
        $cmdline =~ s/\$\(NCORES\)/$ncores/g;

        print "$cmdline\n";
        print "((job_counter++))\n";
        print "update_progress\n";
    }
}
print "\necho -e '\\nAll jobs complete.'\n";
