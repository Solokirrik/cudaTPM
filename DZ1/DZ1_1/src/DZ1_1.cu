/*
 ============================================================================
 Name        : DZ1_1.cu
 Author      : Solo
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <numeric>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const int time_s = 10;
static const float st_len = 100.0;
static const float ht = 0.001;
static const float hx = 0.1;
static const float hx2 = 0.01;
static const float hthx2 = ht/hx2;

__global__ void reciprocalKernel(float *k_cur, float *k_next, unsigned k_size) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx == 0) {
		k_cur[idx] = k_next[idx] + 5 * ht ;
	} else if (idx < k_size - 1) {
		//k_cur[idx] = - hthx2 * k_next[idx + 1] + (2 * hthx2 + 1) * k_next[idx] - hthx2 * k_next[idx - 1] ;
		k_next[idx] = k_cur[idx] + hthx2*(k_next[idx + 1] - 2 * k_next[idx] + k_next[idx - 1] );
	}
}


float *gpuReciprocal(float *hostData, unsigned size)
{
	float ht_count = time_s / ht;
	//float **rt = new float*[ht_count];
	float GPUTime = 0.0f;
	float *rc = new float[size];
	float *devNext;
	float *devCur;
	float *buf;

	cudaEvent_t start, stop;
	size_t mem_size = sizeof(float) * size;
	//std::ofstream a_file ( "text_out0.txt" );

	//for(int i = 0; i < ht_count; i++)
	//	rt[i] = new float[size];

	CUDA_CHECK_RETURN(cudaMalloc((void **)&devNext, mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devCur, mem_size));
	CUDA_CHECK_RETURN(cudaMemcpy(devNext, hostData, mem_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devCur, hostData, mem_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 512;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start, 0);

	for (int i = 0; i < ht_count; i++)
	{
		reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (devCur, devNext, size);
		buf = devCur;
		//CUDA_CHECK_RETURN(cudaMemcpy(rt[i], gpuData, mem_size, cudaMemcpyDeviceToHost));
		devCur = devNext;
		devNext = buf;
	}

	cudaEventRecord ( stop , 0);
	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &GPUTime, start, stop);
	std::cout << std::setprecision(3) << "GPU time: " << GPUTime << " mS"<< std::endl;

	CUDA_CHECK_RETURN(cudaMemcpy(rc, devNext, sizeof(float)*size, cudaMemcpyDeviceToHost));

//	for(int i = 0; i < size; i++){
//		for(int j = 0; j < size; j++){
//			a_file << rt[i][j] << " ";
//		}
//		a_file << std::endl;
//	}
//	a_file.close();

	CUDA_CHECK_RETURN(cudaFree(devNext));
	CUDA_CHECK_RETURN(cudaFree(devCur));
//	for(int i = 0; i < ht_count; i++)
//			delete[] rt[i];

	return rc;
}


void initData(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0.0;
}

int main(void)
{
	static const int WORK_SIZE = st_len / hx;
	float *data = new float[WORK_SIZE];

	initData (data, WORK_SIZE);

	float *recGpu = gpuReciprocal(data, WORK_SIZE);

	for (int i = 0; i < WORK_SIZE; i++){
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4) << std::setw(15)
			<< recGpu[i];
	}
	std::cout << std::endl;

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
