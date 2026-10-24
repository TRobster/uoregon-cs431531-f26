#include <stdlib.h> 
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include <inttypes.h>
#include <time.h>

/*
Author: Trevor Robbins
Class: CS 431/531
Project: Calculating PI
Date: 10/23/26
*/

#define PI 3.1415926535


void usage(int argc, char** argv);
double calcPi_Serial(int num_steps);
double calcPi_P1(int num_steps);
double calcPi_P2(int num_steps);


int main(int argc, char** argv)
{
    // get input values
    uint32_t num_steps = 100000;
    if(argc > 1) {
        num_steps = atoi(argv[1]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32"\n", num_steps);
    }
    fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);


    // set up timer
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // calculate in serial 
    start_t = ReadTSC();
    double Pi0 = calcPi_Serial(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi serially with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi0);
    
    // calculate in parallel with integration
    start_t = ReadTSC();
    double Pi1 = calcPi_P1(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi1);


    // calculate in parallel with Monte Carlo
    start_t = ReadTSC();
    double Pi2 = calcPi_P2(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // with %"PRIu32" guesses is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi2);

    
    return 0;
}


void usage(int argc, char** argv)
{
    fprintf(stdout, "usage: %s <# steps>\n", argv[0]);
}

double calcPi_Serial(int num_steps)
{
    double pi = 0.0;

    double  h = 2.0 / ((double) (num_steps));
    for (int i = 0; i < num_steps; i++)
    {
        double x = -1 + (i * h);
        pi += sqrt(1 - x * x) * h;
    }
    // half circle (area pi/2), so double it
    pi = pi * 2;
    return pi;  
}

double calcPi_P1(int num_steps)
{
    double pi = 0.0;
    double h = 2.0 / ((double) (num_steps));
    #pragma omp parallel for schedule(static) reduction(+:pi)
    for (int i = 0; i < num_steps; i++)
    {
        double x = -1 + (i * h);
        pi += sqrt(1 - x * x);
    }
    // scale by rectangle width h once instead of every iteration;
    return pi * h * 2;
}

// xorshift64* step: cheap per-thread PRNG with no shared state (rand() would serialize threads)
static inline uint64_t next_rand(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double calcPi_P2(int num_steps)
{
    uint64_t hits = 0;
    const double scale = 1.0 / 9007199254740992.0;   // maps 53 random bits to [0,1)

    // Found out about this, each thread gets their own private value
    // initalized to 0, then reduces at the end of the parallel execution (pretty neat)
    #pragma omp parallel reduction(+:hits)
    {
        // distinct nonzero seed per thread (hasing golden ratio)
        uint64_t state = 0x9E3779B97F4A7C15ULL * (omp_get_thread_num() + 1);

        #pragma omp for schedule(static)
        for (int i = 0; i < num_steps; i++)
        {
            double x = (next_rand(&state) >> 11) * scale;
            double y = (next_rand(&state) >> 11) * scale;
            // unit circle
            if (x * x + y * y <= 1.0)
                hits++;
        }
    }

    // random points in the unit square; the fraction inside the quarter circle is pi/4
    return 4.0 * (double) hits / (double) num_steps;
}
