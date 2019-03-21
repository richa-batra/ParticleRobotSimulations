#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// #include <cuda_runtime.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// CUDA utilities and system includes
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h>
#include <helper_functions.h>

#include <iostream>
#include <vector>
#include <numeric>

#include "opencv2/opencv.hpp"

cv::VideoWriter writer;
cv::Mat frame;
const double FPS = 20.0;
int i = 0;

#define BLOCK_SIZE 16    // block size

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void PostprocessKernel(uchar4* dst, uchar3* dst1, unsigned int imgWidth, unsigned int imgHeight)
{
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int bw = blockDim.x;
  unsigned int bh = blockDim.y;
  // Non-normalized U, V coordinates of input texture for current thread.
  unsigned int u = (bw * blockIdx.x) + tx;
  unsigned int v = (bh * blockIdx.y) + ty;

  // Early-out if we are beyond the texture coordinates for our texture.
  if (u > imgWidth-1) return;
  if (v > imgHeight-1) return;

  unsigned int index = (v * imgWidth) + u;
  unsigned int index1 = ((imgHeight - v - 1) * imgWidth) + u;

  uchar4 color = tex2D(texRef, u, v);

  dst[index] = color;
  dst1[index1] = make_uchar3(color.z, color.y, color.x);
}

uchar4* g_dstBuffer = NULL;
uchar3* g_dstBuffer1 = NULL;
size_t g_BufferSize = 0;

void PostprocessCUDA(cudaGraphicsResource_t& dst, cudaGraphicsResource_t& src,
	unsigned int width, unsigned int height, unsigned int interval,
    const char* outputFilename)
{

  cudaGraphicsResource_t resources[2] = { src, dst };

  // Map the resources so they can be used in the kernel.
  checkCudaErrors(cudaGraphicsMapResources(2, resources));

  cudaArray* srcArray;
  cudaArray* dstArray;

  // Get a device pointer to the OpenGL buffers
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&srcArray, src, 0, 0));
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&dstArray, dst, 0, 0));

  // Map the source texture to a texture reference.
  checkCudaErrors(cudaBindTextureToArray(texRef, srcArray));

  // Destination buffer to store the result of the postprocess effect.
  size_t bufferSize = width * height * sizeof(uchar4);
  size_t bufferSize1 = width * height * sizeof(uchar3);
  if (g_BufferSize != bufferSize)
    {
      if (g_dstBuffer != NULL)
        {
          cudaFree(g_dstBuffer);
        }
      if (g_dstBuffer1 != NULL)
        {
          cudaFree(g_dstBuffer1);
        }
      // Only re-allocate the global memory buffer if the screen size changes,
      // or it has never been allocated before (g_BufferSize is still 0)
      g_BufferSize = bufferSize;
      checkCudaErrors(cudaMalloc(&g_dstBuffer, g_BufferSize));
      checkCudaErrors(cudaMalloc(&g_dstBuffer1, bufferSize1));
    }

  if (!writer.isOpened()) {
    frame = cv::Mat(height, width, CV_8UC3, cvScalar(0));
    int x = writer.open(outputFilename,
                        cv::VideoWriter::fourcc('X', 'V','I', 'D'),
                        FPS, frame.size());
  }

  // Compute the grid size
  unsigned int blocksW = (unsigned int)ceilf(width / (float)BLOCK_SIZE);
  unsigned int blocksH = (unsigned int)ceilf(height / (float)BLOCK_SIZE);
  dim3 gridDim(blocksW, blocksH, 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  PostprocessKernel << < gridDim, blockDim >> >(g_dstBuffer, g_dstBuffer1, width, height);
  if(i%interval==0){
    checkCudaErrors(cudaMemcpy(frame.data, g_dstBuffer1, bufferSize1, cudaMemcpyDeviceToHost));
    writer.write(frame);
  }
  // Copy the destination back to the source array
  checkCudaErrors(cudaMemcpyToArray(dstArray, 0, 0, g_dstBuffer, bufferSize, cudaMemcpyDeviceToDevice));

  // Unbind the texture reference
  checkCudaErrors(cudaUnbindTexture(texRef));

  // Unmap the resources again so the texture can be rendered in OpenGL
  checkCudaErrors(cudaGraphicsUnmapResources(2, resources));
  i = i + 1;
}

void PostprocessFinish() {
}
