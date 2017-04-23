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
#include <iostream>
#include <iomanip>
#include <math.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const float a = 0.5;
static const float hx2 = 0.1;
static const float hy2 = 0.1;
static const float ht = 0.1;
static const float ht2 = 0.01;
static const float proc_time = 25;

static const int mem_ln = 99;
static const int mem_wd = 99;

__global__ void reciprocalKernel(float *k_prev, float *k_cur, float *k_next, float *k_pressure, unsigned k_LEN, unsigned k_WD) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

	if ((idx % k_LEN != 0) && ((idx + 1) % k_LEN != 0) && (idx > k_LEN) && (idx < k_LEN * (k_WD - 1))) {
		k_next[idx] = 2 * k_cur[idx] - k_prev[idx]
			        + (a * ((k_cur[idx + 1] - 2 * k_cur[idx] + k_cur[idx - 1]) / hx2
			        + (k_cur[idx + k_LEN] - 2 * k_cur[idx] + k_cur[idx - k_LEN]) / hy2)
			        + k_pressure[idx]) * ht2;
	}
}

float *gpuReciprocal(float *hostData, float *hostPress, unsigned LENGHT, unsigned WIDTH)
{
	int size = LENGHT * WIDTH;
	float *rc;
	rc = new float[size];

	size_t data_size = sizeof(float) * size;
	cudaEvent_t start, stop;

	float *buf;
	float *devPress;
	float *devPrev;
	float *devCur;
	float *devNext;

	float GPUTime = 0.0f;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&devPrev, data_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devCur, data_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devNext, data_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devPress, data_size));

	CUDA_CHECK_RETURN(cudaMemcpy(devPrev, hostData, data_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devCur, hostData, data_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devNext, hostData, data_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devPress, hostPress, data_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaEventCreate (&start );
	cudaEventCreate (&stop );
	cudaEventRecord (start , 0);

	for (int i = 0; i < proc_time/ht; i++){
		reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (devPrev, devCur, devNext, devPress, LENGHT, WIDTH);
		buf = devPrev;
		devPrev = devCur;
		devCur = devNext;
		devNext = buf;
	}

	cudaEventRecord ( stop , 0);
	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &GPUTime, start, stop);
//	std::cout << std::setprecision(3) << "GPU time: " << GPUTime << " mS"<< std::endl;

	CUDA_CHECK_RETURN(cudaMemcpy(rc, devCur, sizeof(float)*size, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(devPrev));
	CUDA_CHECK_RETURN(cudaFree(devCur));
	CUDA_CHECK_RETURN(cudaFree(devNext));
	CUDA_CHECK_RETURN(cudaFree(devPress));

	return rc;
}

// Gaussian distribution
void initGausPres(float *press, unsigned LENGHT, unsigned WIDTH)
{
	double r;
	double sigma = 5.0;
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
	// normalization
//	for (int i = 0; i < LENGHT; ++i)
//		for (int j = 0; j < WIDTH; ++j)
//			press[i * LENGHT + j] /= sum;

	// gain
	for (int i = 0; i < LENGHT; ++i)
		for (int j = 0; j < WIDTH; ++j)
			press[i * LENGHT + j] *= 1500;
}

void initData(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0;
}

int main(void)
{
	float *inpData;
	float *pressure;
	inpData = new float[mem_ln * mem_wd];
	pressure = new float[mem_ln * mem_wd];

	initData(inpData, mem_ln * mem_wd);
	initData(pressure, mem_ln * mem_wd);
	pressure[(mem_ln / 2) * mem_wd + mem_wd / 2] = 10;
//	initGausPres(pressure, mem_ln, mem_wd);

	float *recGpu = gpuReciprocal(inpData, pressure, mem_ln, mem_wd);

//	console output
//	std::cout << "External pressure" << std::endl;
//	for (int i = 0; i < mem_ln; i++) {
//		for (unsigned j = 0; j < mem_wd; ++j) {
//			std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(10)
//				<< pressure[i * mem_ln + j];
//		}
//		std::cout << std::endl;
//	}
//	std::cout << "Result" << std::endl;
	for (unsigned i = 0; i < mem_ln; i++) {
		for (unsigned j = 0; j < mem_wd; ++j) {
			std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(10)
				<< recGpu[i * mem_ln + j];
		}
		std::cout << std::endl;
	}

	delete[] inpData;
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
