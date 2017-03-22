#include <stdlib.h>
#include <stdio.h>
#include <openacc.h>
#include <time.h>
#include <math.h>

void Sum (float *restrict hC, float *hA, float *hB, int size)
{
 #pragma acc kernels loop present (hC,hA,hB)
 for ( int i=0; i < size; i++ ) 
 {float sum = 0.f;
  float ab = hA[i] * hB[i];
  for ( int j=0; j < 100; j++ ) sum += sinf(ab+j);
  hC [i] = sum;  
 }

}

int main(int argc, char* argv[])
{
 clock_t start, stop;
 float timerValueCPU;

 float *hA,*hB,*hC,sum,ab;
 int size=512*50000; 
 int i,j;
 
 unsigned int mem_size=sizeof(float)*size;
  
 hA = (float*) malloc (mem_size); 
 hB = (float*) malloc (mem_size);
 hC = (float*) malloc (mem_size);
 
 for(i=0;i<size;i++) {hA[i]=sinf(i); hB[i]=cosf(2.f*i-5.f); hC[i]=0.0f;}
 
 // CPU ------------------------------------------------------------------- 
 printf("\n Start ...");
 start = clock ();

#pragma acc data copyin(hA[0:size],hB[0:size]) copyout (hC[0:size])
{ 
 Sum (hC, hA, hB, size);
}

 stop = clock();
 timerValueCPU = (float)(stop - start)/CLOCKS_PER_SEC;
 printf("\n CPU calculation time: %f ms\n",1000.0f*timerValueCPU);

 printf("\n hC[100] = %f \n",hC[100]);
 
 free(hA);
 free(hB);
 free(hC);
 
 return 0;
}




