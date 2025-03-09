#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            int p = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0)
                    odata[p++] = idata[i];
            }

            timer().endCpuTimer();
            return p;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // scan
            int* scan = new int[n];
            for (int i = 0; i < n; ++i) {
                scan[i] = idata[i] == 0 ? 0 : 1;
            }

            // exclusive prefix sum
            int* exclusivePrefixSum = new int[n];
            exclusivePrefixSum[0] = 0;
            for (int i = 1; i < n; ++i) {
                exclusivePrefixSum[i] = exclusivePrefixSum[i - 1] + scan[i - 1];
            }

            // scatter
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                if (scan[i] != 0) {
                    odata[exclusivePrefixSum[i]] = idata[i];
                    ++cnt;
                }
            }

            delete[] scan;
            delete[] exclusivePrefixSum;

            timer().endCpuTimer();
            return cnt;
        }
    }
}
