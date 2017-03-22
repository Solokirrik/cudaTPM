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
//#include <iomanip>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const int time_s = 10;
static const float st_len = 100.0;
static const float hx = 0.1;
static const float hx2 = 0.01;
static const float ht = 0.001;

__global__ void reciprocalKernel(float *data, float *oData, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx == 0) {
		data[idx] = oData[idx] + 5;
	} else if (idx < vectorSize - 1) {
		data[idx] = (oData[idx + 1] - 2 * oData[idx] + oData[idx - 1]) * ht / hx2 + oData[idx];
	}
}


float *gpuReciprocal(float *data, unsigned size)
{
	float ht_count = time_s / ht;
	//float **rt = new float*[ht_count];
	float GPUTime = 0.0f;
	float *rc = new float[size];
	float *gpuData2;
	float *gpuData;
	float *buf;

	cudaEvent_t start, stop;
	size_t mem_size = sizeof(float) * size;
	//std::ofstream a_file ( "text_out0.txt" );

	//for(int i = 0; i < ht_count; i++)
	//	rt[i] = new float[size];

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData2, mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, mem_size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData2, data, mem_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, mem_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	cudaEventCreate ( &start );
	cudaEventCreate ( &stop );
	cudaEventRecord (start , 0);

	for (int i = 0; i < ht_count; i++)
	{
		reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, gpuData2, size);
		buf = gpuData;
		//CUDA_CHECK_RETURN(cudaMemcpy(rt[i], gpuData, mem_size, cudaMemcpyDeviceToHost));
		gpuData = gpuData2;
		gpuData2 = buf;
	}

	cudaEventRecord ( stop , 0);
	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &GPUTime, start, stop);
	printf("GPU time: %.3f mS\n", GPUTime);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData2, sizeof(float)*size, cudaMemcpyDeviceToHost));

//	for(int i = 0; i < size; i++){
//		for(int j = 0; j < size; j++){
//			a_file << rt[i][j] << " ";
//		}
//		a_file << std::endl;
//	}
//	a_file.close();

	CUDA_CHECK_RETURN(cudaFree(gpuData2));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
//	for(int i = 0; i < ht_count; i++)
//			delete[] rt[i];

	return rc;
}


void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0.0;
}

int main(void)
{
	static const int WORK_SIZE = st_len / hx;
	float *data = new float[WORK_SIZE];

	initialize (data, WORK_SIZE);

	float *recGpu = gpuReciprocal(data, WORK_SIZE);

	for (int i = 0; i < WORK_SIZE; i++){
		std::cout << recGpu[i] << std::endl;
	}

	/* Free memory */
	delete[] data;
	delete[] recGpu;

	return 0;
}


// Check the return value of the CUDA runtime API call and exit the application if the call has failed.
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
