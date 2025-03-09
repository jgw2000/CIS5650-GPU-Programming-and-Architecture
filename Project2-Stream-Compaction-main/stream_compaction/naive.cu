#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        constexpr int blockSize = 256;

        __global__ void kernExclusiveScan(int n, int offset, int* odata, int* idata) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
                return;

            if (index >= offset)
                odata[index] = idata[index] + idata[index - offset];
            else
                odata[index] = idata[index];
        }

        __global__ void kernShift(int n, int* odata, int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
                return;

			if (index == 0)
				odata[index] = 0;
			else
				odata[index] = idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int blockNum = (n + blockSize - 1) / blockSize;
            dim3 blocksPerGrid(blockNum);

            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int offset = 1; offset < n; offset *= 2) {
                kernExclusiveScan<<<blocksPerGrid, blockSize>>>(n, offset, dev_odata, dev_idata);
                std::swap(dev_idata, dev_odata);
            }

            kernShift<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
