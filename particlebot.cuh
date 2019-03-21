/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <curand.h>
#include <curand_kernel.h>


extern "C"
{

  void cudaInit(int argc, char **argv);

  void allocateArray(void **devPtr, int size);
  void freeArray(void *devPtr);

  void threadSync();

  void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
  void copyArrayToDevice(void *device, const void *host, int offset, int size);
  void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
  void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
  void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
  void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


  void setParameters(SimParams *hostParams);

  void integrateSystem(float *pos,
                       float *vel,
                       float *rad,
                       float deltaTime,
                       uint nCells,
                       float time);

  void calcHash(uint  *gridParticlebotHash,
                uint  *gridParticlebotIndex,
                float *pos,
                int    nCells);

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
                                   uint   numCells);




  void updateRad_light_wave(float *pos,
                            float* absForce_a,
                            float* absForce_r,
                            float *rad,
                            float *phase,
                            float time,
                            float deltaTime,
                            int*  dead,
                            int    nCells);

  void curand_setup(curandState *state, int N);

  void add_normal_noise(curandState *state,
                        float *val,
                        float std,
                        int N);

  void updatePhase(float *pos,
                   float *phase,
                   float spacing,
                   float max_d,
                   float min_d,
                   int nCells);



  void updateCol(float *rad,
                 float *col,
                 int nCells,
                 float *pos,
                 float *phase,
                 int* dead);

  void collide(float *newVel,
               float* absForce_a,
               float* absForce_r,
               float *sortedPos,
               float *sortedVel,
               float *sortedRad,
               uint  *gridParticlebotIndex,
               uint  *cellStart,
               uint  *cellEnd,
               uint   nCells,
               uint   numCells,
               float deltaTime);

  void calcCOG(float *pos,
               float *temppos,
               float *temppos1,
               int    nCells,
               float time,
               int hist_steps,
               float hist_int);

  void sortParticlebots(uint *dGridParticlebotHash,
                        uint *dGridParticlebotIndex,
                        uint nCells);

}
