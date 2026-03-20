/*******************************************************************
* Common timer file for use with SPEChpc Weak Scaling              *
*
*
*******************************************************************/
#include <stdio.h>
#include <mpi.h>
#include <specmpitime.h>

// SPEC_TIME_TOTAL 0
// SPEC_TIME_INIT 1
// SPEC_TIME_MPI 2
// SPEC_TIME_COMP 3

typedef struct {
     double time;
     double culm_time;
     bool running;
} spec_timers;

spec_timers spectimes[4];
int spectime_rank=-1;

void spectime_start(int timer) {

    if (spectime_rank < 0) {
	MPI_Comm_rank(MPI_COMM_WORLD, &spectime_rank);
	for (int i=0; i < 4; ++i) {
	   spectimes[i].running = false;
	   spectimes[i].time = 0.0;
	   spectimes[i].culm_time = 0.0;
	}
    }
    if (spectime_rank==0) {
	if (!spectimes[timer].running) {
	  spectimes[timer].running = true;
	  spectimes[timer].time = MPI_Wtime();
        } else {
	  // ignore if the timer is already running
	  return;
        }
    }	
}

void spectime_stop(int timer) {
    double endtime;
    if (spectime_rank==0) {
	if (spectimes[timer].running) {
	    endtime = MPI_Wtime();
	    spectimes[timer].culm_time += (endtime - spectimes[timer].time);
	    spectimes[timer].running =false;
        } else {
	  // ignore if the timer is not running
	  return;
        }
    }
}

void spectime_final(bool pass, double units) {
    FILE *fp;
    double rate, tick, resid;
    if (spectime_rank==0) {
	for (int i=0; i < 4; ++i) {
	   if (spectimes[i].running) {
	        spectime_stop(i);
           }
	}
	if (spectimes[SPEC_TIME_COMP].culm_time <= 0) {
	   fprintf(stderr,"ERROR: SPEC Computation timer not used.\n");
	   return;
	}
	if (units > 0.0) {
        	rate = units / spectimes[SPEC_TIME_COMP].culm_time;
        } else {
	        rate = 0.0;
	}
	fp = fopen("spectimes.txt","w");
	if (pass) {
	   fprintf(fp,"Verification: PASSED\n");
	} else { 	
	   fprintf(fp,"Verification: FAILED\n");
        }
        resid = spectimes[SPEC_TIME_TOTAL].culm_time - spectimes[SPEC_TIME_INIT].culm_time - spectimes[SPEC_TIME_COMP].culm_time;	
	fprintf(fp,"%17s %20.9f\n","FOM:",rate); 	
	fprintf(fp,"%17s %20.9f\n","Init time:",spectimes[SPEC_TIME_INIT].culm_time); 	
	fprintf(fp,"%17s %20.9f\n","Core time:",spectimes[SPEC_TIME_COMP].culm_time); 	
	fprintf(fp,"%17s %20.9f\n","Resid time:",resid); 	
	fprintf(fp,"%17s %20.9f\n","Total time:",spectimes[SPEC_TIME_TOTAL].culm_time); 	
        fclose(fp);
    }
}


