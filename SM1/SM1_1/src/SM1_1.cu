/*
 ============================================================================
 Name        : SM1.cu
 Author      : Solo
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <numeric>
#include <stdlib.h>
#include <iostream>
#include <fstream>
//#include <string>
#include <iomanip>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const int time_s = 10;
static const float dx = 0.1;
static const float dx2 = 0.01;
static const float dt = 0.001;
static const float st_len = 100.0;

__global__ void reciprocalKernel(float *k_cur, float *k_next, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx == 0) {
		k_next[idx] = k_cur[idx] + 5;
	} else if (idx < vectorSize - 1) {
		k_next[idx] = (k_cur[idx + 1] - 2 * k_cur[idx] + k_cur[idx - 1]) * dt / dx2 + k_cur[idx];
	}
}


float *gpuReciprocal(float *data, unsigned size)
{
	float ht_count = time_s / dt;
	float GPUTime = 0.0f;

	float *buf;
	float *GPUcur;
	float *GPUnext;
	float *rc = new float[size];

	cudaEvent_t start, stop;
	size_t mem_size = sizeof(float) * size;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&GPUcur, mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&GPUnext, mem_size));
	CUDA_CHECK_RETURN(cudaMemcpy(GPUcur, data, mem_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(GPUnext, data, mem_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start , 0);

	for (int i = 0; i < ht_count; i++)
	{
		reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (GPUcur, GPUnext, size);
		buf = GPUcur;
		GPUcur = GPUnext;
		GPUnext = buf;
	}

	cudaEventRecord ( stop , 0);
	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &GPUTime, start, stop);
	printf("GPU time: %.3f mS\n", GPUTime);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, GPUnext, sizeof(float)*size, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(GPUcur));
	CUDA_CHECK_RETURN(cudaFree(GPUnext));

	return rc;
}


void initData(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0.0;
}

int main(void)
{
	static const int WORK_SIZE = st_len / dx;
	float *data = new float[WORK_SIZE];

	initData (data, WORK_SIZE);

	float *recGpu = gpuReciprocal(data, WORK_SIZE);

	for (int i = 0; i < WORK_SIZE; i++){
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4) << std::setw(15)
			<< recGpu[i];
	}

	delete[] data;
	delete[] recGpu;

	return 0;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
