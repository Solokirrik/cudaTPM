#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

FILE *out,*out1;
int main()
{int N=1024*2;
 int m,n,k;
 double start, stop, timerValueCPU;
 
 int numBytes = N*N*sizeof( float ); 
 float *a,*b,*c,*cc,*bT,*aT,sum ; 
 
 a = (float*) malloc (numBytes);
 b = (float*) malloc (numBytes);
 bT = (float*) malloc (numBytes);
 aT = (float*) malloc (numBytes);
 c = (float*) malloc (numBytes);
 cc = (float*) malloc (numBytes);

 for(n=0;n<N;n++)
 {for(m=0;m<N;m++)
  {a[m+n*N]=2.0f*m+n;
   b[m+n*N]=m-n;
   aT[m+n*N]=m+n*2.0f;
   bT[m+n*N]=n-m;
  }
 }
 
 start = omp_get_wtime (); 

 #pragma omp parallel for shared (a,b,cc) private (n,m,k)
 for(n=0;n<N;n++)
 {for(m=0;m<N;m++)
  {sum=0.f;
   for(k=0;k<N;k++) sum+=a[k+n*N]*bT[k+m*N]; 
   cc[m+n*N]=sum; 
  }
 }

 stop = omp_get_wtime ();
 printf ("\n OMP Calculation time = %f ms\n", 1000.*(stop-start));

 
 free(a);
 free(b);
 free(bT);
 free(aT);
 free(c);
 free(cc);
 
 return 0;
}