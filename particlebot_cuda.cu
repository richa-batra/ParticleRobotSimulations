#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particlebot_kernel_impl.cuh"

extern "C"
{

  void cudaInit(int argc, char **argv)
  {
    int devID;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
      {
        printf("No CUDA Capable devices found, exiting...\n");
        exit(EXIT_SUCCESS);
      }
  }

  void cudaGLInit(int argc, char **argv)
  {
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaGLDevice(argc, (const char **)argv);
  }

  void allocateArray(void **devPtr, size_t size)
  {
    checkCudaErrors(cudaMalloc(devPtr, size));
  }

  void freeArray(void *devPtr)
  {
    checkCudaErrors(cudaFree(devPtr));
  }

  void threadSync()
  {
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void copyArrayToDevice(void *device, const void *host, int offset, int size)
  {
    checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
  }

  void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
  {
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                 cudaGraphicsMapFlagsNone));
  }

  void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
  {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
  }

  void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
  {
    void *ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                         *cuda_vbo_resource));
    return ptr;
  }

  void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
  {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
  }

  void copyArrayFromDevice(void *host, const void *device,
                           struct cudaGraphicsResource **cuda_vbo_resource, int size)
  {
    if (cuda_vbo_resource)
      {
        device = mapGLBufferObject(cuda_vbo_resource);
      }

    checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

    if (cuda_vbo_resource)
      {
        unmapGLBufferObject(*cuda_vbo_resource);
      }
  }

  void setParameters(SimParams *hostParams)
  {
    // copy parameters to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    checkCudaErrors(cudaMemcpyToSymbol(x1obs, hostParams->x1obs, sizeof(float)*hostParams->nobstacles));
    checkCudaErrors(cudaMemcpyToSymbol(x2obs, hostParams->x2obs, sizeof(float)*hostParams->nobstacles));
    checkCudaErrors(cudaMemcpyToSymbol(y1obs, hostParams->y1obs, sizeof(float)*hostParams->nobstacles));
    checkCudaErrors(cudaMemcpyToSymbol(y2obs, hostParams->y2obs, sizeof(float)*hostParams->nobstacles));

    checkCudaErrors(cudaMemcpyToSymbol(x_cir_obs, hostParams->x_cir_obs, sizeof(float)*hostParams->n_cir_obstacles));
    checkCudaErrors(cudaMemcpyToSymbol(y_cir_obs, hostParams->y_cir_obs, sizeof(float)*hostParams->n_cir_obstacles));
    checkCudaErrors(cudaMemcpyToSymbol(r_cir_obs, hostParams->r_cir_obs, sizeof(float)*hostParams->n_cir_obstacles));
  }

  //Round a / b to nearest higher integer value
  uint iDivUp(uint a, uint b)
  {
    return (a % b != 0) ? (a / b + 1) : (a / b);
  }

  // compute grid and thread block size for a given number of elements
  void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
  {
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
  }

  // compute grid and thread block size for a given number of elements
  void computeGridSize2(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
  {
    numThreads = blockSize;
    numBlocks = iDivUp(n, numThreads);
  }

  void integrateSystem(float *pos,
                       float *vel,
                       float *rad,
                       float deltaTime,
                       uint nCells,
                       float time)
  {
    thrust::device_ptr<float2> d_pos2((float2 *)pos);
    thrust::device_ptr<float2> d_vel2((float2 *)vel);
    thrust::device_ptr<float> d_rad((float *)rad);

    thrust::for_each(
                     thrust::make_zip_iterator(thrust::make_tuple(d_pos2, d_vel2, d_rad)),
                     thrust::make_zip_iterator(thrust::make_tuple(d_pos2+nCells, d_vel2+nCells, d_rad+nCells)),
                     integrate_functor(deltaTime, time));
  }

  void calcHash(uint  *gridParticlebotHash,
                uint  *gridParticlebotIndex,
                float *pos,
                int    nCells)
  {
    uint numThreads, numBlocks;
    computeGridSize(nCells, 256, numBlocks, numThreads);

    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(gridParticlebotHash,
                                           gridParticlebotIndex,
                                           (float2 *) pos
                                           );

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
  }

  void updateRad_light_wave(float *pos,
                            float *absForce_a,float *absForce_r,
                            float *rad,
                            float *phase,
                            float time, float deltaTime, int *dead,
                            int    nCells)
  {
    uint numThreads, numBlocks;
    computeGridSize(nCells, 256, numBlocks, numThreads);

    // execute the kernel
    updateRad_light_wave <<< numBlocks, numThreads >>>((float2 *)pos, (float*)absForce_a, (float*)absForce_r, (float*)rad, phase,  time, deltaTime, dead);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
  }


  void curand_setup(curandState *state, int N) {
    uint numThreads, numBlocks;
    computeGridSize(N, 256, numBlocks, numThreads);
    curand_setup_kernel << < numBlocks, numThreads >> > (state, N);
  }

  void add_normal_noise(curandState *state, float *val, float std, int N) {
    uint numThreads, numBlocks;
    computeGridSize(N, 256, numBlocks, numThreads);
    add_normal_noise_kernel << < numBlocks, numThreads >> > (state, val, std, N);
  }

  void updatePhase(float *pos,
                   float *phase,
                   float spacing,
                   float max_d,
                   float min_d,
                   int nCells)

  {
    uint numThreads, numBlocks;
    computeGridSize(nCells, 256, numBlocks, numThreads);
    updatePhase <<< numBlocks, numThreads >>>((float2 *)pos, phase, spacing, max_d, min_d);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
  }


  void updateCol(float *rad,
                 float *col,
                 int    nCells, float *pos, float *phase, int *dead)
  {
    uint numThreads, numBlocks;
    computeGridSize(nCells, 256, numBlocks, numThreads);

    // execute the kernel
    updateCol_k <<< numBlocks, numThreads >>>((float*)rad, (float4 *)col, (float2 *)pos, phase, dead);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
  }

  void calcCOG(float *pos, float *temppos, float *temppos1,
               int    nCells, float time, int hist_steps, float hist_int)
  {
    int ind = ((int)(time/hist_int ) )% hist_steps;
    uint numThreads, numBlocks;
    int nCells_new;
    nCells_new = nCells;
    computeGridSize2(nCells, 64, numBlocks, numThreads);
    float mul = 1.0f/float(nCells);
    uint smemSize = sizeof(float)*2*(numThreads+1);
    // execute the kernel
    if(numBlocks==1){
      calcCOG1<64> <<< numBlocks, numThreads, smemSize >>>((float2 *)pos, (float2 *) temppos1, nCells, mul);
    }
    else{
      calcCOG<64> <<< numBlocks, numThreads, smemSize >>>((float2 *)pos, (float2 *) temppos, nCells);
    }

    while(numBlocks>1){
      nCells_new = (int)(numBlocks);

      computeGridSize2(nCells_new, 64, numBlocks, numThreads);
      smemSize = sizeof(float)*2*(numThreads+1);
      if(numBlocks== 1){
        calcCOG1<64> <<< numBlocks, numThreads, smemSize >>>((float2 *)temppos, (float2 *) temppos1, nCells_new, mul);
      }
      else{
        calcCOG<64> <<< numBlocks, numThreads, smemSize >>>((float2 *)temppos, (float2 *) temppos1, nCells_new);
        checkCudaErrors(cudaMemcpy(temppos, temppos1, 2*sizeof(float)*nCells_new, cudaMemcpyDeviceToDevice));
      }


    }
    checkCudaErrors(cudaMemcpy(pos+2*(ind+nCells), temppos1, 2*sizeof(float), cudaMemcpyDeviceToDevice));

    //float2 temp;
    //checkCudaErrors(cudaMemcpy(&temp, temppos1, 2*sizeof(float), cudaMemcpyDeviceToHost));
    //printf("centroid ind: %d, x: %f, y: %f\n", ind, temp.x, temp.y);

    getLastCudaError("Kernel execution failed");
  }


  void reorderDataAndFindCellStart(uint  *cellStart,
                                   uint  *cellEnd,
                                   float *sortedPos,
                                   float *sortedVel,
                                   float *sortedRad,
                                   uint  *gridParticlebotHash,
                                   uint  *gridParticlebotIndex,
                                   float *oldPos,
                                   float *oldVel,
                                   float *oldRad,
                                   uint   nCells,
                                   uint   numCells)
  {
    uint numThreads, numBlocks;
    computeGridSize(nCells, 256, numBlocks, numThreads);

    // set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
    checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, nCells*sizeof(float2)));
    checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, nCells*sizeof(float2)));
#endif

    uint smemSize = sizeof(uint)*(numThreads+1);
    reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
                                                                       cellStart,
                                                                       cellEnd,
                                                                       (float2 *) sortedPos,
                                                                       (float2 *) sortedVel,
                                                                       sortedRad,
                                                                       gridParticlebotHash,
                                                                       gridParticlebotIndex,
                                                                       (float2 *) oldPos,
                                                                       (float2 *) oldVel,
                                                                       oldRad
                                                                       );
    getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
    checkCudaErrors(cudaUnbindTexture(oldPosTex));
    checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
  }



  void collide(float *newVel,
               float *absForce_a,
               float *absForce_r,
               float *sortedPos,
               float *sortedVel,
               float *sortedRad,
               uint  *gridParticlebotIndex,
               uint  *cellStart,
               uint  *cellEnd,
               uint   nCells,
               uint   numCells, float deltaTime)
  {
#if USE_TEX
    checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, nCells*sizeof(float4)));
    checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, nCells*sizeof(float4)));
    checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
    checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

    // thread per particlebot
    uint numThreads, numBlocks;
    computeGridSize(nCells, 64, numBlocks, numThreads);

    // execute the kernel
    collideD<<< numBlocks, numThreads >>>((float2 *)newVel,
                                          (float *)absForce_a,
                                          (float *)absForce_r,
                                          (float2 *)sortedPos,
                                          (float2 *)sortedVel,
                                          sortedRad,
                                          gridParticlebotIndex,
                                          cellStart,
                                          cellEnd,
                                          deltaTime);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

#if USE_TEX
    checkCudaErrors(cudaUnbindTexture(oldPosTex));
    checkCudaErrors(cudaUnbindTexture(oldVelTex));
    checkCudaErrors(cudaUnbindTexture(cellStartTex));
    checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
  }

  void sortParticlebots(uint *dGridParticlebotHash, uint *dGridParticlebotIndex, uint nCells)
  {
    thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticlebotHash),
                        thrust::device_ptr<uint>(dGridParticlebotHash + nCells),
                        thrust::device_ptr<uint>(dGridParticlebotIndex));
  }

}   // extern "C"
