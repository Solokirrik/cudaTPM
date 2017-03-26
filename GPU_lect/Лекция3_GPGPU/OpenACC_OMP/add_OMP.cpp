#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

 #pragma omp parallel for shared (hA,hB,hC) private (i,j,sum)
 for(i=0;i<size;i++) 
 {sum=0.f; ab=hA[i]*hB[i];
  for(j=0;j<100;j++) sum = sum + sinf(ab+j);
  hC [i] = sum;  
 }
 
 stop = clock();
 timerValueCPU = (float)(stop - start)/CLOCKS_PER_SEC;
 printf("\n CPU calculation time: %f ms\n",1000.0f*timerValueCPU);
 
 free(hA);
 free(hB);
 free(hC);
 
 return 0;
}