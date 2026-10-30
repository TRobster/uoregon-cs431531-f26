#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"


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
        int *tmp = in; 
        in = out;
        out = tmp;
    }

}

/*
void prefix_sum_p2(int* src, int* prefix, int n)
{
    int T = omp_get_max_threads();
    int* totals = malloc(T * sizeof(int)); // Each thread gets a slot 

#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int nt  = omp_get_num_threads();
    int chunk = (n + nt - 1) / nt;
    int start = tid * chunk;
    int end   = (start + chunk < n) ? start + chunk : n; // take min

    // phase 1: serial scan of chunk
    if (start < n) {
        prefix[start] = src[start];
        for (int i = start + 1; i < end; i++)
        {
            
            prefix[i] = prefix[i-1] + src[i];
        };
        totals[tid] = prefix[end-1];
    } else totals[tid] = 0;

    #pragma omp barrier // synchronization

    // phase 2 one thread scans the T totals into offsets
    #pragma omp single
    {
        int run = 0;
        for (int t = 0; t < nt; t++) 
        { 
            int x = totals[t]; 
            totals[t] = run; 
            run += x; 
        }
    }  // single has an implicit barrier at the end

    // phase 3: add my offset to my chunk
    int off = totals[tid];
    for (int i = start; i < end; i++)
        prefix[i] += off;
}
free(totals);

}
*/

void prefix_sum_p22(int* src, int* prefix, int n)
{
    memcpy(prefix,src, n * sizeof(int));

    for (int d = 1; d < n; d *=2)
    {
        #pragma omp parallel for schedule (static)
        for (int i = 0; i < n; i += 2*d)
        {
            prefix[i + 2*d - 1] += prefix[i + d - 1];
        }
    }

}


