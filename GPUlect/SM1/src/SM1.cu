/*
 ============================================================================
 Name        : SM1.cu
 Author      : TishenkovKirill
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

#define ST_LEN	1000.0
#define DTIME	10.0
#define DCRD	0.01
#define DCRD2	0.001
#define HTIME	0.0001


/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, float *stData, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx == 0)
		data[idx] = stData[idx] + 5;
	else if (idx < vectorSize)
		data[idx] = (stData[idx+1] - 2*stData[idx] + stData[idx-1])*HTIME/DCRD2 + stData[idx];
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, float *stData, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;
	float *gpuData2;
	float *buffer;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData2, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData2, stData, sizeof(float)*size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&buffer, sizeof(float)*size));
	

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	for(int i = 0; i < DTIME; i++){
		reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, gpuData2, size);
		buffer = gpuData;
		gpuData = gpuData2;
		gpuData2 = buffer;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	CUDA_CHECK_RETURN(cudaFree(gpuData2));
	return rc;
}

void initialize(float *data, float *stData, unsigned size)
{
	for (unsigned i = 0; i < size; ++i){
		data[i] = 0;
		stData[i] = 0;
	}
}


int main(void)
{
	static const int WORK_SIZE = ST_LEN*DCRD;
	float *data = new float[WORK_SIZE];
	float *stData = new float[WORK_SIZE];

	initialize (data, stData, WORK_SIZE);

	float *recGpu = gpuReciprocal(data, stData, WORK_SIZE);

	for(int i = 0; i < WORK_SIZE; i++)
		std::cout << recGpu[i] << std::endl;

	delete[] data;
	delete[] recGpu;
	delete[] stData;
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

