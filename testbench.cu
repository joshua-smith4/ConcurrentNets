#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

__global__ void addVec(int *a, size_t pitch_a, int *b, size_t pitch_b, int *c, size_t pitch_c, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N)
  {
    int* aElem = (int*)((char*)a + j * pitch_a) + i;
    int* bElem = (int*)((char*)b + j * pitch_b) + i;
    int* cElem = (int*)((char*)c + j * pitch_c) + i;
    *cElem = *aElem + *bElem;
  }
}

int main() {
  const unsigned N = 10;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 10000);
  int **a = (int**)std::malloc(sizeof(int *) * N);
  for (int i = 0; i < N; ++i)
  {
    a[i] = (int*)std::malloc(sizeof(int) * N);
    for (int j = 0; j < N; ++j)
      a[i][j] = rand() % 100;
  }

  int **b = (int**)std::malloc(sizeof(int *) * N);
  for (int i = 0; i < N; ++i)
  {
    b[i] = (int*)std::malloc(sizeof(int) * N);
    for (int j = 0; j < N; ++j)
      b[i][j] = rand() % 100;
  }
  // std::generate(vec_a.begin(), vec_a.end(), [&](){ return dis(gen); });
  // std::generate(vec_b.begin(), vec_b.end(), [&](){ return dis(gen); });
  std::cout << "finished generating numbers\n";
  int *d_a, *d_b, *d_c;
  std::size_t pitch_a, pitch_b, pitch_c;
  cudaMallocPitch(&d_a, &pitch_a, sizeof(int)*N, N);
  cudaMallocPitch(&d_b, &pitch_b, sizeof(int)*N, N);
  cudaMallocPitch(&d_c, &pitch_c, sizeof(int)*N, N);

  cudaMemcpy2D(d_a, pitch_a, a, sizeof(int)*N, sizeof(int)*N, N, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_b, pitch_a, a, sizeof(int)*N, sizeof(int)*N, N, cudaMemcpyHostToDevice);

  const unsigned THREADS_PER_BLOCK_X = 10;
  const unsigned THREADS_PER_BLOCK_Y = 10;
  const unsigned NUM_BLOCKS_X =
      (N + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X;
  const unsigned NUM_BLOCKS_Y =
      (N + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y;

  dim3 gridDim(NUM_BLOCKS_Y, NUM_BLOCKS_X);
  dim3 blockDim(THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_X);
  addVec<<<gridDim, blockDim>>>(d_a, pitch_a, d_b, pitch_b, d_c, pitch_c, N);

  int **c = (int**)std::malloc(sizeof(int *) * N);
  for(int i = 0; i < N; ++i)
  {
    c[i] = (int*)std::malloc(sizeof(int)*N);
  }
  cudaMemcpy2D(c, sizeof(int)*N, d_c, pitch_c, sizeof(int) * N, N, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  for (int i = 0; i < N; ++i)
  {
    for(int j = 0; j < N; ++i)
    {
      assert(c[i][j] == a[i][j] + b[i][j]);
    }
  }
  std::cout << "Passed!\n";
  for (int i = 0; i < N; ++i)
  {
    for(int j = 0; j < N; ++i)
    {
      std::cout << a[i][j] << " ";
    }
    std::cout << std::endl;
  }
  for (int i = 0; i < N; ++i)
  {
    for(int j = 0; j < N; ++i)
    {
      std::cout << b[i][j] << " ";
    }
    std::cout << std::endl;
  }
  for (int i = 0; i < N; ++i)
  {
    for(int j = 0; j < N; ++i)
    {
      std::cout << c[i][j] << " ";
    }
    std::cout << std::endl;
  }

  for(int i = 0; i < N; ++i)
  {
    std::free(a[i]);
  }
  std::free(a);
  for(int i = 0; i < N; ++i)
  {
    std::free(b[i]);
  }
  std::free(b);
  for(int i = 0; i < N; ++i)
  {
    std::free(c[i]);
  }
  std::free(c);
}
