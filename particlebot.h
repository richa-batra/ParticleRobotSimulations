#ifndef __PARTICLEBOT_H__
#define __PARTICLEBOT_H__


#include <helper_functions.h>
#include "particlebot_kernel.cuh"
#include "vector_functions.h"
#include <curand.h>
#include <curand_kernel.h>

#include <cstdio>

// Particlebot system class
class Particlebot
{
 public:
  Particlebot(SimParams params);
  ~Particlebot();


  void update(float deltaTime, float sort_interval);
  void reset();

  float *getArray(ParticlebotArray array);
  void setArray(ParticlebotArray array, const float *data, int start, int count);


  unsigned int getCurrentReadBuffer() const
  {
    return posVbo;
  }
  unsigned int getColorBuffer()       const
  {
    return colorVBO;
  }
  unsigned int getRadBuffer()       const
  {
    return radVbo;
  }
  void *getCudaPosVBO()              const
  {
    return (void *)cudaPosVBO;
  }
  void *getCudaColorVBO()            const
  {
    return (void *)cudaColorVBO;
  }
  void *getCudaRadVBO()            const
  {
    return (void *)cudaRadVBO;
  }

  void dumpParticlebot(uint start, uint count, FILE *fp,
                       float dump_interval, uint testing, float light_x,
                       float light_y);

  void loadFromFile(uint start, uint count, FILE *fp,
                    float dump_interval);

  float2 getWorldOrigin()
  {
    return params.worldOrigin;
  }
  float2 getCellSize()
  {
    return params.cellSize;
  }


 protected: // methods
  Particlebot() {}
  uint createVBO(uint size);

  void _initialize();
  void _finalize();

  void initGrid(uint2 size, float spacing, float jitter,
                uint numParticles);
  void initHexGrid(uint numParticles, float spacing);
 protected: // data


  // CPU data
  float *hPos;              // particle positions
  float *hVel;              // particle velocities
  float *hRad;             // particle radii
  int *hDead;

  uint  *hCellStart;
  uint  *hCellEnd;

  // GPU data
  float *dPos;
  float *dVel;
  float *dAbsForce_a;
  float *dAbsForce_r;
  int *dDead;
  curandState *dState;

  float *dRad;             // particle radii
  float *dfreq;
  float *hfreq;
  float *dphase;
  float *hphase;


  float *dSortedPos;
  float *tempPos1;
  float *tempPos2;

  float *dSortedVel;
  float *dSortedRad;


  // grid data for sorting method
  uint  *dGridParticleHash; // grid hash value for each particle
  uint  *dGridParticleIndex;// particle index for each particle
  uint  *dCellStart;        // index of start of each cell in sorted list
  uint  *dCellEnd;          // index of end of cell

  uint   posVbo;            // vertex buffer object for particle positions
  uint   colorVBO;          // vertex buffer object for colors
  uint   radVbo;            // vertex buffer object for particle radii

  float  time;

  float *cudaPosVBO;        // these are the CUDA deviceMem Pos
  float *cudaColorVBO;      // these are the CUDA deviceMem Color
  float *cudaRadVBO;        // these are the CUDA deviceMem Radii, only used if not using opengl and dumping data to disk instead

  struct cudaGraphicsResource *cuda_posvbo_resource; // handles OpenGL-CUDA exchange
  struct cudaGraphicsResource *cuda_colorvbo_resource; // handles OpenGL-CUDA exchange
  struct cudaGraphicsResource *cuda_radvbo_resource; // handles OpenGL-CUDA exchange
  //struct cudaGraphicsResource *m_cuda_timevbo_resource; // handles OpenGL-CUDA exchange

  // params
  SimParams params;
  uint2 particlebotConfigSize;

};

#endif
