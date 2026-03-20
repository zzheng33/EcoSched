/*******************************************************************
* Common timer file for use with SPEChpc Weak Scaling              *
*
*
*******************************************************************/
#include <stdint.h>
#include <stdbool.h>

#define SPEC_TIME_TOTAL 0
#define SPEC_TIME_INIT 1
#define SPEC_TIME_MPI 2
#define SPEC_TIME_COMP 3

void spectime_start(int timer);
void spectime_stop(int timer);
void spectime_final(bool pass, double units);

#if defined(SPEC) && !defined(SPEC_NO_TIMER)
#define SPEC_TIME_START spectime_start(0);
#define SPEC_TIME_STOP  spectime_stop(0);
#define SPEC_TIME_START_INIT spectime_start(1);
#define SPEC_TIME_STOP_INIT spectime_stop(1);
#define SPEC_TIME_START_MPI spectime_start(2);
#define SPEC_TIME_STOP_MPI spectime_stop(2);
#define SPEC_TIME_START_COMP spectime_start(3);
#define SPEC_TIME_STOP_COMP spectime_stop(3);
#define SPEC_TIME_FINAL(P,U) spectime_final(P,U);
#else
#define SPEC_TIME_START 
#define SPEC_TIME_STOP  
#define SPEC_TIME_START_INIT 
#define SPEC_TIME_STOP_INIT 
#define SPEC_TIME_START_MPI 
#define SPEC_TIME_STOP_MPI 
#define SPEC_TIME_START_COMP 
#define SPEC_TIME_STOP_COMP 
#define SPEC_TIME_FINAL(P,U)
#endif 
