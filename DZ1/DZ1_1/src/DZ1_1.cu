/*
 ============================================================================
 Name        : DZ1_1.cu
 Author      : Solo
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <iomanip>

using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const int dT = 5;
static const unsigned tot_time = 100;
static const float len = 10.0;
static const float ht = 0.001;
static const float hx = 0.1;

static const float A = -ht/(hx*hx);
static const float B = 2*ht/(hx*hx) + 1;
static const float C = -ht/(hx*hx);

ofstream fout("temp.txt");

// ядро разогрева одного конца
__global__ void kernel_data_heat(float *data, float *newdata, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize - 1){
		newdata[idx] = data[idx];
		if (idx == 0){
			newdata[idx] = data[idx] + dT * 0.1;
		}
	}
}

// ядро расчёта приближения
__global__ void kernel_data_calc(float *newData, float *stbData, float *bufData, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ((idx > 0)&&(idx < vectorSize - 1)){
		newData[idx] = bufData[idx] + (stbData[idx] - (A*bufData[idx+1] + B*bufData[idx] + C*bufData[idx-1]));
	}
	else if(idx == vectorSize - 1){
		newData[idx] = 0;
	}
}

// вывод в файл
void print_to_file(float *recGpu, unsigned long a_width)
{
	for (unsigned long j = 0; j < a_width; ++j) {
		fout << recGpu[j] << " ";
	}
	fout << endl;
}

float *gpuReciprocal(float *hostData, unsigned size)
{
	float GPUTime = 0.0f;
	float *rc = new float[size];

	float *buf;			// буфер обмена обновления и нового приближения
	float *heatedData;	// буфер разогрева конца

	float *devStbData;	// установившиеся значение
	float *devNewData;	// обновлённые значения на шаге приближения
	float *devBufData;	// буфер приближения к базе

	cudaEvent_t start, stop;
	size_t mem_size = sizeof(float) * size;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&devStbData, mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devNewData, mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devBufData, mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&buf, mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&heatedData, mem_size));

	CUDA_CHECK_RETURN(cudaMemcpy(devStbData, hostData, mem_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devNewData, hostData, mem_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devBufData, hostData, mem_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(heatedData, hostData, mem_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start, 0);

	for (unsigned long i = 0; i < tot_time; i++) {
		cudaMemcpy(heatedData, devStbData, mem_size, cudaMemcpyDeviceToDevice);
		kernel_data_heat<<<blockCount, BLOCK_SIZE>>>(devStbData, heatedData, size);
//		обновление базовых значений
		cudaMemcpy(devStbData, heatedData, mem_size, cudaMemcpyDeviceToDevice);
		cudaMemcpy(devBufData, devStbData, mem_size, cudaMemcpyDeviceToDevice);
		cudaMemcpy(devNewData, devStbData, mem_size, cudaMemcpyDeviceToDevice);
//		прогон разогрева от нового базового значения
		for (unsigned long j = 0; j < 1000; j++) {
			kernel_data_calc<<<blockCount, BLOCK_SIZE>>>(devNewData, devStbData, devBufData, size);
//			обновление промежуточных значений температуры
			cudaMemcpy(buf, devBufData, mem_size, cudaMemcpyDeviceToDevice);
			cudaMemcpy(devBufData, devNewData, mem_size, cudaMemcpyDeviceToDevice);
			cudaMemcpy(devNewData, buf, mem_size, cudaMemcpyDeviceToDevice);
		}
//		запись нового базового значения
		cudaMemcpy(devStbData, devNewData, mem_size, cudaMemcpyDeviceToDevice);
		cudaMemcpy(rc, devStbData, mem_size, cudaMemcpyDeviceToHost);
		print_to_file(rc, size);
	}

	cudaEventRecord ( stop , 0);
	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &GPUTime, start, stop);
	cout << "\n" << GPUTime << endl;

	CUDA_CHECK_RETURN(cudaMemcpy(rc, devStbData, mem_size, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(devBufData));
	CUDA_CHECK_RETURN(cudaFree(devNewData));
	CUDA_CHECK_RETURN(cudaFree(devStbData));

	return rc;
}


void initData(float *data, unsigned width)
{
	for (unsigned i = 0; i < width; i++){
		data[i] = 0.0;
	}
	data[0] = 1;
	data[width - 1] = 0;
}

int main(void)
{
	const unsigned a_width = len / hx;
	float *data = new float[a_width];
	float *recGpu;
	initData(data, a_width);

	cout << "Init done" << endl;
	if(fout.is_open()){
		cout << "Writing data" << endl;

		recGpu = gpuReciprocal(data, a_width);

		cout << "\n" << "Data saved" << endl;
	} else {
		cout << "File could not be opened." << endl;
	}

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
