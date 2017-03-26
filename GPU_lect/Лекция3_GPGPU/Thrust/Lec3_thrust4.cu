#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

struct saxpy {
	int a;

	saxpy(int a): a(a) {}

	__host__ __device__ float operator() (int x, int y) {
		return a*x + y;
	}
};

int main() {
	float CPUstart, CPUstop;
	float GPUstart, GPUstop;
	float CPUtime = 0.0f;
	float GPUtime = 0.0f;

	int CPUresult = 0;
	int GPUresult = 0;

	int a = 3;

	thrust::host_vector<int> hostVectorA(32<<20);
	thrust::host_vector<int> hostVectorB(32<<20);
	thrust::host_vector<int> hostVectorC(32<<20);

	thrust::generate(hostVectorA.begin(), hostVectorA.end(), rand);
	thrust::generate(hostVectorB.begin(), hostVectorB.end(), rand);

	CPUstart = clock();

	thrust::transform(hostVectorA.begin(), hostVectorA.end(),hostVectorB.begin(), hostVectorC.begin(), saxpy(a));

	CPUstop = clock();
	CPUtime = 1000.*(CPUstop - CPUstart) / CLOCKS_PER_SEC;
	printf("CPU time : %.3f ms\n", CPUtime);


	thrust::device_vector<int> deviceVectorA = hostVectorA;
	thrust::device_vector<int> deviceVectorB = hostVectorB;
	thrust::device_vector<int> deviceVectorC(32<<20);

	GPUstart = clock();

	thrust::transform(deviceVectorA.begin(), deviceVectorA.end(), deviceVectorB.begin(), deviceVectorC.begin(), saxpy(a));

	GPUstop = clock();
	GPUtime = 1000.*(GPUstop - GPUstart) / CLOCKS_PER_SEC;
	printf("GPU time : %.3f ms\n", GPUtime);

	printf("ArraySize :  %d \n", 32<<22);
	printf("Rate : %.3f \n", CPUtime/GPUtime);

	return 0;
}

