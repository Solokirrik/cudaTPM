#include <stdlib.h>
#include <stdio.h>
#include <openacc.h>
#include <time.h>
#include <math.h>



void Matrix_Mul (float *restrict cc, float *a, float *bT, int N)
{
 #pragma acc parallel loop present (cc,a,bT)
 for (int n = 0; n < N; n++ )
 {for (int m = 0; m < N; m++ )
  {float sum=0.f; 
   for(int k = 0; k < N; k++ ) sum+=a[k+n*N]*bT[k+m*N];
   cc[m+n*N] = sum;  
  }
 }
 
}



int main()
{int N=1024*2;
 int NN=N*N;
 int m,n,k;
 double start, stop, timerValueCPU;
 
 int numBytes = NN*sizeof( float ); 
 float *a,*b,*cc,*bT ; 
 
 a =  (float*) malloc (numBytes);
 b =  (float*) malloc (numBytes);
 bT = (float*) malloc (numBytes);
 cc = (float*) malloc (numBytes);

 for(n=0;n<N;n++)
 {for(m=0;m<N;m++)
  {a[m+n*N]=2.0f*m+n;
   b[m+n*N]=m-n;
   bT[m+n*N]=n-m;
  }
 }
 
 float sum;
 
 start = clock ();

#pragma acc data copyin (bT[0:NN],a[0:NN]) copyout (cc[0:NN])
{
  Matrix_Mul (cc,a,bT,N); 
}

 
 stop = clock();
 timerValueCPU = 1000.*(stop - start)/CLOCKS_PER_SEC; 
 printf("\n CPU calculation time %f msec\n",timerValueCPU);
 
 free(a);
 free(b);
 free(bT);
 free(cc);
 
 return 0;
}