#!/usr/bin/perl

package Trace;
use warnings;

sub parse {
    my ($filename) = @_;
    my $fh;
    open($fh, '<', $filename) or die "cannot open file $filename\n";
    chomp(my @lines = <$fh>);
    close($fh);

    my @trace_info;
    my $rec;
    undef $rec;
    foreach my $elem (@lines) {
        if ($elem ne "") {
            my $idx = index($elem, "=");
            my $key = substr($elem, 0, $idx);
            my $value = substr($elem, $idx + 1);

            if ($key eq "NAME") {
                if (defined $rec) {
                    push @trace_info, $rec;
                    undef $rec;
                }
            }
            $rec->{$key} = $value;
        }
    }
    push @trace_info, $rec if defined $rec;

    return @trace_info;
}


1;