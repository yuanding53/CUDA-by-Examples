
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

int main(void) {
	int count;
	cudaGetDeviceCount(&count);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Name:%s \n", prop.name);
	printf("Compute capability: %d.%d \n", prop.major, prop.minor);
	printf("Clock rate: %d \n", prop.clockRate);
	printf("Device copy overlap: ");
	if (prop.deviceOverlap)
		printf("Enabled \n");
	else
		printf("Disabled \n");
	printf("Kernel execition tiomeout: ");
	if (prop.kernelExecTimeoutEnabled)
		printf("Enabled \n");
	else
		printf("Disabled \n");
	printf("\n");
	printf("Total global mem: %lld\n", prop.totalGlobalMem);
	printf("Total constant mem: %ld\n", prop.totalConstMem);
	printf("Max mem pitch: %ld\n", prop.memPitch);
	printf("Texture alignment: %ld\n", prop.textureAlignment);
	printf("\n");
	printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
	printf("Shared mem per block: %ld\n", prop.sharedMemPerBlock);
	printf("Registers per block: %ld\n", prop.regsPerBlock);
	printf("Threads in warp:%d\n", prop.warpSize);
	printf("Max threads per block:%d\n", prop.maxThreadsPerBlock);
	printf("Max thread dimensions:(%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("Max grid dimensions: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("\n");
}
