/*
 ============================================================================
 Name        : edge_detect.cu
 Author      : Solo
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <math.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const float K[25] = {2, 4, 5, 4, 2,
					  4, 9, 12, 9, 4,
					  5, 12, 15, 12, 5,
					  4, 9, 12, 9, 4,
					  2, 4, 5, 4, 2};
static const float Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
static const float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__global__ void smoothingKernel(float *data, float *outdata, float *K, unsigned long vectorSize, unsigned k_w, unsigned k_h) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize){
		float sigma_m = 1.0/159;
		int up = -2;
		int bot = 3;
		int left = -2;
		int right = 3;

		int mtrx_w = 5;
		int new_cen = mtrx_w / 2;

		int j = idx % k_w;
		int i = idx / k_w;

		if (i < -up){
			up = -i;
		}
		else if (i >  k_h - bot){
			bot = k_h - i;
		}
		if(j < -left){
			left = -j;
		}
		else if (j > k_w - right){
			right = k_w - j;
		}
			for(int k = up; k < bot; k++){
				for(int l = left; l < right; l++){
					outdata[idx] += sigma_m * K[(k + new_cen)*mtrx_w + l + new_cen] * data[(i + k)*k_w + (j + l)];
				}
			}
	}
}

__global__ void gradientKernel(float *data, float *outdata, float *Gx, float *Gy,
								unsigned long vectorSize, unsigned k_w, unsigned k_h) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize){
		int up = -1;
		int bot = 2;
		int left = -1;
		int right = 2;

		int mtrx_w = 3;
		int new_cen = mtrx_w / 2;
		int j = idx % k_w;
		int i = idx / k_w;
		float Gxx = 0;
		float Gyy = 0;

		if (i < -up){
			up = -i;
		}
		else if (i >  k_h - bot){
			bot = k_h - i;
		}
		if(j < -left){
			left = -j;
		}
		else if (j > k_w - right){
			right = k_w - j;
		}
			for(int k = up; k < bot; k++){
				for(int l = left; l < right; l++){
					Gxx += Gx[(k + new_cen) * mtrx_w + l + new_cen] * data[(i + k)*k_w + (j + l)];
					Gyy += Gy[(k + new_cen) * mtrx_w + l + new_cen] * data[(i + k)*k_w + (j + l)];
					outdata[idx] = sqrt(pow(Gxx, 2) + pow(Gyy, 2));
				}
			}
	}
}

__global__ void minmaxfilterKernel(float *data, float *outdata, unsigned long vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize){
		if(data[idx] < 0){
			outdata[idx] = 0;
		}
		else if(data[idx] > 255){
			outdata[idx] = 255;
		}
		else{
			outdata[idx] = int(data[idx]);
		}
	}
}

float *gpuReciprocal(float *data, unsigned long size, unsigned width, unsigned height)
{
	float *rc = new float[size];
	float *gpuData;
	float *gpuOutData1, *gpuOutData2, *gpuOutData3;
	float *gpuK, *gpuGx, *gpuGy;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuK, sizeof(K)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuGx, sizeof(Gx)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuGy, sizeof(Gy)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuOutData1, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuOutData2, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuOutData3, sizeof(float)*size));

	CUDA_CHECK_RETURN(cudaMemcpy(gpuK, Gx, sizeof(K), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuGx, Gx, sizeof(Gx), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuGy, Gy, sizeof(Gy), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 512;
	const int blockCount = (size+BLOCK_SIZE-1) / BLOCK_SIZE;

	smoothingKernel<<<blockCount, BLOCK_SIZE>>>(gpuData, gpuOutData1, gpuK, size, width, height);
	gradientKernel<<<blockCount, BLOCK_SIZE>>>(gpuOutData1, gpuOutData2, gpuGx, gpuGy, size, width, height);
	minmaxfilterKernel<<<blockCount, BLOCK_SIZE>>>(gpuOutData2, gpuOutData3, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuOutData3, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuGx));
	CUDA_CHECK_RETURN(cudaFree(gpuGy));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	CUDA_CHECK_RETURN(cudaFree(gpuOutData1));
	CUDA_CHECK_RETURN(cudaFree(gpuOutData2));
	CUDA_CHECK_RETURN(cudaFree(gpuOutData3));

	return rc;
}

int main(void)
{
	std::string line;
	unsigned H = 0;
	unsigned W = 0;
	float *data;
	unsigned long WORK_SIZE = 0;

	std::ifstream myFile("cars.txt");
	if (myFile.is_open()) {
		H = std::count(std::istreambuf_iterator<char>(myFile), std::istreambuf_iterator<char>(), '\n');
		myFile.seekg(0, myFile.beg);
		//W = std::count(std::istreambuf_iterator<char>(myFile), std::istreambuf_iterator<char>(), ' ') / H + 1; // 425ms

		getline(myFile, line);
		std::stringstream stream(line);	// 13ms
		int n;
		while (stream >> n) {
			W++;
		}
//		std::cout << "lines:\t" << H << std::endl;
//		std::cout << "values:\t" << W << std::endl;

		WORK_SIZE = W*H;
		data = new float[WORK_SIZE];

		myFile.seekg(0, myFile.beg);
		unsigned long i = 0;
		while (i < WORK_SIZE) {
			myFile >> data[i];
			i++;
		}
	}
	else {
		std::cout << "File not found" << std::endl;
	}
	myFile.close();

	float *recGpu = gpuReciprocal(data, WORK_SIZE, W, H);

	for (int i = 0; i < H; i++) {
		for (unsigned j = 0; j < W; ++j) {
			std::cout << recGpu[i * W + j] << " ";
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

