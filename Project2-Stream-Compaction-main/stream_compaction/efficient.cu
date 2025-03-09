#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 256;

        __global__ void kernUpSweep(int n, int stride, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
                return;

            if ((index + 1) % stride == 0) {
				data[index] += data[index - stride / 2];
            }
        }

        __global__ void kernZero(int n, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index == n - 1) {
                data[index] = 0;
            }
        }

        __global__ void kernDownSweep(int n, int stride, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
                return;

            if ((index + 1) % stride == 0) {
				int t = data[index - stride / 2];
				data[index - stride / 2] = data[index];
				data[index] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int n_padded = (1 << ilog2ceil(n));
            int blockNum = (n_padded + blockSize - 1) / blockSize;
            dim3 blocksPerGrid(blockNum);

            int* dev_data;
            cudaMalloc((void**)&dev_data, n_padded * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int d = 0; d < ilog2(n_padded); ++d) {
                kernUpSweep<<<blocksPerGrid, blockSize>>>(n_padded, 1 << (d + 1), dev_data);
            }

            kernZero<<<blocksPerGrid, blockSize>>>(n_padded, dev_data);

			for (int d = ilog2(n_padded) - 1; d >= 0; --d) {
				kernDownSweep<<<blocksPerGrid, blockSize>>>(n_padded, 1 << (d + 1), dev_data);
			}
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int n_padded = (1 << ilog2ceil(n));
            int blockNum = (n + blockSize - 1) / blockSize;
            dim3 blocksPerGrid(blockNum);

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            int* iscans = new int[n];
            int* oscans = new int[n];

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			Common::kernMapToBoolean<<<blocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);

			cudaMemcpy(iscans, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            scan(n, oscans, iscans);
			cudaMemcpy(dev_indices, oscans, n * sizeof(int), cudaMemcpyHostToDevice);

			Common::kernScatter<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

            cudaMemcpy(odata, dev_odata, n * sizeof(n), cudaMemcpyDeviceToHost);

            int remaining = oscans[n - 1];
			if (idata[n - 1] != 0) {
				remaining++;
			}

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            delete[] iscans;
            delete[] oscans;

            return remaining;
        }
    }
}
