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

int main() {
	float CPUstart, CPUstop;
	float GPUstart, GPUstop;
	float CPUtime = 0.0f;
	float GPUtime = 0.0f;

	int CPUresult = 0;
	int GPUresult = 0;

	thrust::host_vector<int> hostVector(32<<22);
	thrust::generate(hostVector.begin(), hostVector.end(), rand);

	thrust::device_vector<int> deviceVector = hostVector;

	CPUstart = clock();

	CPUresult = thrust::reduce(hostVector.begin(), hostVector.end(), 0, thrust::plus<int>());

	CPUstop = clock();
	CPUtime = 1000.*(CPUstop - CPUstart) / CLOCKS_PER_SEC;
	printf("CPU time : %.3f ms\n", CPUtime);

	GPUstart = clock();

	GPUresult = thrust::reduce(deviceVector.begin(), deviceVector.end(), 0, thrust::plus<int>());

	GPUstop = clock();
	GPUtime = 1000.*(GPUstop - GPUstart) / CLOCKS_PER_SEC;
	printf("GPU time : %.3f ms\n", GPUtime);

	printf("ArraySize :  %d \n", 32<<22);
	printf("Rate : %.3f \n", CPUtime/GPUtime);

	return 0;
}

