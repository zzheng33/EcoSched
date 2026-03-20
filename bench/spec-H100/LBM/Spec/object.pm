$benchnum  = '605';
$benchname = 'lbm_s';
$exename   = 'lbm';
$benchlang = 'C';
@base_exe  = ($exename);

$reltol = 0.006;
$abstol = 0.0000004;
$bench_cflags = "-Ispecmpitime";

@sources = qw( 
lbm.c
main.c
specrand/specrand.c
specmpitime/specmpitime.c
               ) ;

$need_math = 'yes';

sub invoke {
    my ($me) = @_;
    my @rc;
    my $exe = $me->exe_file;
    push (@rc, { 'command' => $exe,
                 'args'    => [ ],
                 'output'  => "$exename.out",
                 'error'   => "$exename.err",
          });
    return @rc;
}


sub pre_build {
        my ($me, $dir, $setup, $target) = @_;
        my $bname = "$benchnum\.$benchname";
        my $pmodel = $me->pmodel;
        $me->pre_build_set_pmodel($dir, $setup, $target, $bname);
        return 0;
}
1;
