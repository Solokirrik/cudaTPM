#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace gpu;

int main()
{
    float start, stop;
    float time = 0.0f;

    if(getCudaEnabledDeviceCount() < 1) exit(-1);

    Mat src = imread("./cars.jpg",0);
    if(!src.data) exit(-1);

    Mat dst;

    start = clock();

    bilateralFilter(src,dst,-1, 50, 7);

    stop = clock();
    time = 1000.*(stop - start) / CLOCKS_PER_SEC;
    printf("Filter time : %.3f ms\n", time);

    start = clock();

    Canny(dst,dst, 35, 200, 3);

    stop = clock();
    time = 1000.*(stop - start) / CLOCKS_PER_SEC;
    printf("Canny time : %.3f ms\n", time);

    imwrite("outCARs1.jpg",dst);

    gpu::GpuMat device_src(src);
    gpu::GpuMat device_dst;

    start = clock();

    gpu::bilateralFilter(device_src, device_dst, -1, 50, 7);

    stop = clock();
    time = 1000.*(stop - start) / CLOCKS_PER_SEC;
    printf("Filter time : %.3f ms\n", time);

    start = clock();

    gpu::Canny(device_dst, device_dst, 35, 200, 3);

    stop = clock();
    time = 1000.*(stop - start) / CLOCKS_PER_SEC;
    printf("Canny time : %.3f ms\n", time);

    device_dst.download(dst);

    imwrite("outCARs2.jpg",dst);

    return 0;
}

