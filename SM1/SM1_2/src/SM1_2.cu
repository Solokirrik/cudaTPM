/*
 ============================================================================
 Name        : SM1_2.cu
 Author      : Solo
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const float dx2 = 0.1;
static const float dy2 = 0.1;
static const float dt = 0.1;
static const float dt2 = 0.01;
static const float proc_time = 10;

static const int mem_ln = 19;
static const int mem_wd = 19;

__global__ void reciprocalKernel(float *k_next, float *k_cur, float *k_pressure, unsigned k_LEN, unsigned k_WD) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	k_pressure[idx] = 1.1 * k_pressure[idx];
	if (((idx + 1) % k_LEN != 0) && (idx > k_LEN) && (idx < k_LEN * (k_WD - 1))) {
		k_next[idx] = ((k_cur[idx + 1] - 2 * k_cur[idx] + k_cur[idx - 1]) / dx2 + (k_cur[idx + 1] - 2 * k_cur[idx] + k_cur[idx - 1]) / dy2 + k_pressure[idx] + k_cur[idx]) * dt2;
	}
}

float *gpuReciprocal(float *data, float *pressure, unsigned LENGHT, unsigned WIDTH)
{
	unsigned size = LENGHT * WIDTH;
	float *rc = new float[size];
	size_t data_size = sizeof(float)*size;
	cudaEvent_t start, stop;

	float *buf;
	float *GPUcur;
	float *GPUnext;
	float *GPUpress;

	float GPUTime = 0.0f;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&GPUcur, data_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&GPUnext, data_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&GPUpress, data_size));

	CUDA_CHECK_RETURN(cudaMemcpy(GPUcur, data, data_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(GPUnext, data, data_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(GPUpress, pressure, data_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	cudaEventCreate (&start );
	cudaEventCreate (&stop );
	cudaEventRecord (start , 0);

	for (int i = 0; i < proc_time/dt; i++){
		reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (GPUnext, GPUcur, GPUpress, LENGHT, WIDTH);
		buf = GPUcur;
		GPUcur = GPUnext;
		GPUnext = buf;
	}

	cudaEventRecord ( stop , 0);
	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &GPUTime, start, stop);
	printf("GPU time: %.3f mS\n", GPUTime);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, GPUcur, sizeof(float)*size, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(GPUcur));
	CUDA_CHECK_RETURN(cudaFree(GPUnext));
	CUDA_CHECK_RETURN(cudaFree(GPUpress));

	return rc;
}

// Gaussian distribution
void initPres(float *press, unsigned LENGHT, unsigned WIDTH)
{
	double r;
	double sigma = 2.0;
	double s = 2.0 * pow(sigma, 2);
	double sum = 0.0;

	int len2 = LENGHT / 2;
	int wid2 = WIDTH / 2;

	for (int x = -len2; x < len2 + 1; x++) {
		for (int y = -wid2; y < wid2 + 1; y++) {
			r = sqrt(x * x + y * y);
			press[(x + len2) * LENGHT + (y + wid2)] = exp(-pow(r, 2) / s) / (M_PI * s);
			sum += press[(x + len2) * LENGHT + (y + wid2)];
		}
	}
	for (int i = 0; i < LENGHT; ++i)
		for (int j = 0; j < WIDTH; ++j)
			press[i * LENGHT + j] /= sum;
}

void initData(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0;
}

int main(void)
{
	float *data = new float[mem_ln * mem_wd];
	float *pressure = new float[mem_ln * mem_wd];

	initData(data, mem_ln * mem_wd);
	initPres(pressure, mem_ln, mem_wd);

	float *recGpu = gpuReciprocal(data, pressure, mem_ln, mem_wd);

	// console output
	std::cout << "External pressure" << std::endl;
	for (int i = 0; i < mem_ln; i++) {
		for (unsigned j = 0; j < mem_wd; ++j) {
			std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(10)
				<< pressure[i + j * mem_ln];
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	std::cout << "Result" << std::endl;
	for (int i = 0; i < mem_ln; i++) {
		for (unsigned j = 0; j < mem_wd; ++j) {
			std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(10)
				<< recGpu[i + j * mem_ln];
		}
		std::cout << std::endl;
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
