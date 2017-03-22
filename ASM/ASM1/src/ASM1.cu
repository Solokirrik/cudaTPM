/*
 ============================================================================
 Name        : Sem1.cu
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define HT 0.001
#define HX 0.1
#define HX2 0.01
#define TIME_IN_SEC 10

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *oldData, float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx == 0) {
		data[idx] = oldData[idx] + 5;
	} else if (idx < vectorSize - 1) {
		data[idx] = ((oldData[idx + 1] - 2 * oldData[idx] + oldData[idx - 1]) * HT / HX2) + oldData[idx];
	}
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *oldGPUData;
	float *gpuData;
	float *t;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&oldGPUData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(oldGPUData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	for (int i = 0; i < TIME_IN_SEC / HT; i++)
	{
		reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (oldGPUData, gpuData, size);
		t = gpuData;
		gpuData = oldGPUData;
		oldGPUData = t;
	}
	CUDA_CHECK_RETURN(cudaMemcpy(rc, oldGPUData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(oldGPUData));
	CUDA_CHECK_RETURN(cudaFree(gpuData));

	for (int i = 0; i < size; i++)
	{
		std::cout << rc[i] << std::endl;
	}
	return rc;
}

void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0.0;
}

int main(void)
{
	static const int WORK_SIZE = 1024;
	float *data = new float[WORK_SIZE];

	initialize (data, WORK_SIZE);

	float *recGpu = gpuReciprocal(data, WORK_SIZE);

	/* Free memory */
	delete[] data;
	delete[] recGpu;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
