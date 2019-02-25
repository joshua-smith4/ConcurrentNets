#include <iostream>
#include <cstdlib>

template <unsigned N>
struct ParallelHistAccumulator
{
public:
  unsigned* bins[N];
  inline __device__ void init(unsigned numBins)
  {
    for(auto i = 0u; i < N; ++i)
    {
      bins[i] = new unsigned[numBins];
    }
  }
  void cleanup()
  {
    for(auto i = 0u; i < N; ++i)
    {
      delete[] bins[i];
    }
  }
  unsigned incrementBin(unsigned bin)
  {
    return 0u;
  }
};

int main()
{
  ParallelHistAccumulator<20> a;
  std::cout << sizeof(a) << "\n";
  a.init(20);
  std::cout << sizeof(a) << "\n";
  a.cleanup();
  return 0;
}
