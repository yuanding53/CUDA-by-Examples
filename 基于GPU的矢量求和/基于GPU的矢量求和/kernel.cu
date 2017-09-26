
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <time.h>

#define N 30

__global__ void add(int *a, int *b, int *c) {
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	double start=clock(), finish;

	//在GPU上分配内存
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	//在CPU上为数组赋值
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i;
	}

	//将数组复制到GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <N, 1 >> > (dev_a, dev_b, dev_c);

	//将数组C从GPU复制到CPU
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	
	//显示结果
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	//释放内存
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	//计算时间
	finish = clock();
	printf("%f seconds\n", (finish - start) / CLOCKS_PER_SEC);

	//利用CPU运行计算运行时间
	start = clock();
	for (int i = 0; i < N; i++) {
		c[i] = a[i] + b[i];
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	finish = clock();
	printf("%f seconds\n", (finish - start) / CLOCKS_PER_SEC);

	return 0;
}
