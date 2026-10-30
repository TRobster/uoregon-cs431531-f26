#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"

/*
Author: Trevor Robbins
Class: CS431
Professor: Choi
Date: 10/30/26
*/


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


    // set up data 
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) 
    {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

// Hills-Steele Algorithm Implementation
void prefix_sum_p1(int* src, int* prefix, int n)
{
    int *in = src;
    int *out = prefix;
    for (int offset = 1; offset < n; offset *= 2)
    {
        #pragma omp parallel for schedule (static)
        for (int i = 0; i < n; i++)
            {
                if (i >= offset)
                {
                    out[i] = in[i] + in[i - offset];
                }
                else 
                {
                    out[i] = in[i];
                }
            }
        // ping pong buffer 
        int *tmp = in; 
        in = out;
        out = tmp;
    }

}



// Blelloch Scan
void prefix_sum_p2(int* src, int* prefix, int n)
{

    memcpy(prefix, src, n * sizeof(int));
    // up-sweep: build partial sums as a tree
    for (int d = 1; d < n; d *=2)
    {
        #pragma omp parallel for schedule (static)
        for (int i = 0; i < n; i += 2*d)
        {
            prefix[i + 2*d - 1] += prefix[i + d - 1];
        }
    }

    // clear the root so the down-sweep produces an exclusive scan
    prefix[n - 1] = 0;

    // down-sweep: push the partial sums back down the tree
    for (int d = n/2; d >= 1; d /= 2)
    {
        #pragma omp parallel for schedule (static)
        for (int i = 0; i < n; i += 2*d)
        {
            // swap values 
            int t = prefix[i + d - 1];
            prefix[i + d - 1] = prefix[i + 2*d - 1];
            prefix[i + 2*d - 1] += t;
        }
    }

    // exclusive -> inclusive: add each element's own value (src is untouched)
    #pragma omp parallel for schedule (static)
    for (int i = 0; i < n; i++)
    {
        prefix[i] += src[i];
    }
}


