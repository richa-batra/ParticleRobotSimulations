#ifndef _PARTICLEBOT_KERNEL_H_
#define _PARTICLEBOT_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "helper_math.h"
#include "math_constants.h"
#include "particlebot_kernel.cuh"

#if USE_TEX

texture<float2, 1, cudaReadModeElementType> oldPosTex;
texture<float2, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticlebotHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif


// All functions should be changed to use this params instead of passed
// parameters

// simulation parameters in constant memory
__constant__ SimParams params;
__constant__ float x1obs[10];
__constant__ float x2obs[10];
__constant__ float y1obs[10];
__constant__ float y2obs[10];
__constant__ float x_cir_obs[10];
__constant__ float y_cir_obs[10];
__constant__ float r_cir_obs[10];

__global__ void curand_setup_kernel(curandState *state, int N) {
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  for (int i = idx; i < N; i += gridDim.x*blockDim.x) {
    curand_init(params.seed, i, 0, &state[i]);
  }
}

__global__ void add_normal_noise_kernel(curandState *my_curandstate, float *val, float std, int N) {
  int idx = threadIdx.x + blockDim.x*blockIdx.x;

  float noise;
  for (int i = idx; i < N; i += gridDim.x*blockDim.x) {
    noise = std*curand_normal(my_curandstate + i);
    val[i] += noise;
  }
}

struct integrate_functor
{
  float deltaTime;
  float time;

  __host__ __device__
  integrate_functor(float delta_time, float time1) : deltaTime(delta_time), time(time1) {}

  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  {
    volatile float2 posData = thrust::get<0>(t);
    volatile float2 velData = thrust::get<1>(t);
    float2 pos = make_float2(posData.x, posData.y);
    float2 vel = make_float2(velData.x, velData.y);
    float rad = thrust::get<2>(t);


    // new position = old position + velocity * deltaTime
    pos += vel * deltaTime;

    if (pos.x > 64.0f - rad)
      {
        pos.x = 64.0f - rad;
        vel.x *= params.boundaryDamping;
      }

    if (pos.x < -64.0f + rad)
      {
        pos.x = -64.0f + rad;
        vel.x *= params.boundaryDamping;
      }

    if (pos.y > 64.0f - rad)
      {
        pos.y = 64.0f - rad;
        vel.y *= params.boundaryDamping;
      }

    if (pos.y < -64.0f + rad)
      {
        pos.y = -64.0f + rad;
        vel.y *= params.boundaryDamping;
      }

    // store new position and velocity
    thrust::get<0>(t) = pos;
    thrust::get<1>(t) = vel;
  }
};

// calculate position in uniform grid
__device__ int2 calcGridPos(float2 p)
{
  int2 gridPos;
  gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
  gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
  return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int2 gridPos)
{
  gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
  gridPos.y = gridPos.y & (params.gridSize.y-1);
  return __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}



__global__
void updateRad_light_wave(float2 *pos, float* absForce_a, float* absForce_r,               // input: positions
                          float *rad, float* phase,
                          float time, float deltaTime, int* dead)
{
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (index >= params.nCells) return;

  if (dead[index]) return;
  if(phase[index] > 10000000.0f) return;

  float time1 = time + phase[index];
  if(time1<0)
    time1=time1+100*(params.Nx + 1)*params.rise_period;
  if(time1>=(params.Nx+1)*params.rise_period)
    time1 = time1 - ((params.Nx+1)*params.rise_period)*
      floor(time1/((params.Nx+1)*params.rise_period));
  if(time1>=2*params.rise_period)
    return;
  float target_r = 0;
  if(time1<=params.rise_period){
    target_r = params.min_radius+
      (params.max_radius-params.min_radius)/params.rise_period * time1;
  }
  else{
    target_r = params.max_radius+
      (params.min_radius-params.max_radius)/params.rise_period * (time1-params.rise_period);
  }
  float dr1 = target_r - rad[index];
  float dr = 0;
  float max_speed = 0.1;
  float torque = dr1*params.constraint*rad[index] / max_speed / params.max_radius / deltaTime;
  torque = fminf(torque, params.constraint);
  // Figure out dr, the actual change of radius, based on dr1 (the desired change in radius),
  // absForce_r, absForce_a, constraint_constant and **deltaTime**
  if(dr1>0){
    if (torque/rad[index] > absForce_r[index]) {
      dr = max_speed*params.max_radius/params.constraint*(torque / rad[index] - absForce_r[index])*deltaTime;
    }
    //dr = fmin(dr, params.max_radius*deltaTime);
  }
  else{
    if (params.constrained_contraction) {
      if (-params.constraint_contraction*dr1 > absForce_a[index] * rad[index]) {
        dr = (params.constraint_contraction*dr1 + absForce_a[index] * rad[index]) / (params.constraint_contraction);
      }
      dr = fmax(dr, -params.max_radius*deltaTime);
    }
    else {
      dr = dr1;
    }
  }
  dr = rad[index]+dr;
  if(dr>params.max_radius) dr=params.max_radius;
  if(dr<params.min_radius) dr=params.min_radius;
  rad[index] = dr;
}


__device__ int checkIntersectionLine(float x0, float y0,float x1, float y1,
                                     float x3, float y3, float x4, float y4){
  if(abs((x4-x3)/(x1-x0)) == abs((y4-y3)/(y1-y0)))
    return 0;
  float t, t1;
  if(abs(y4-y3)>0){
    t = (x3 - x0 - (y3-y0)*(x3-x4)/(y3-y4)) * ((y3-y4)/((x1-x0)*(y3-y4)-(y1-y0)*(x3-x4)));
    if(t<=0 || t>=1)
      return 0;
    t1 = (y3-y0-t*(y1-y0))/(y3-y4);
    if(t1<=0 || t1>=1)
      return 0;
  }
  else if(abs(x4-x3)>0) {
    t = (y3 - y0 - (x3-x0)*(y3-y4)/(x3-x4)) * ((x3-x4)/((y1-y0)*(x3-x4)-(x1-x0)*(y3-y4)));
    if(t<=0 || t>=1)
      return 0;
    t1 = (x3-x0-t*(x1-x0))/(x3-x4);
    if(t1<=0 || t1>=1)
      return 0;
  }
  else{
    return 0;
  }
  return 1;
}

__device__ int checkIntersectionCircle(float light_x, float light_y, float pos_x, float pos_y,
                                       float obs_x, float obs_y, float obs_r){
  float A, B, C, D, C1, C2, C3, C4, C5, C6, R1, R2;
  C1 = powf(light_x, 2) + powf(light_y, 2);
  C2 = powf(pos_x, 2) + powf(pos_y, 2);
  C3 = powf(obs_x, 2) + powf(obs_y, 2);

  C4 = light_x*pos_x + light_y*pos_y;
  C5 = light_x*obs_x + light_y*obs_y;
  C6 = pos_x*obs_x + pos_y*obs_y;

  A = C1 + C2 - 2*C4;
  B = -2*C1 + 2*C4 + 2*C5 - 2*C6;
  C = C1 + C3 - 2*C5 - powf(obs_r,2);

  D = powf(B,2)-4*A*C;
  if(D >= 0){
    R1 = (-B+powf(D,0.5f))/2/A;
    R2 = (-B-powf(D,0.5f))/2/A;
    if(R1>0 && R1 <1)
      return 1;
    if(R2>0 && R2 <1)
      return 1;
  }
  return 0;
}

__device__ int checkIntersection(float pos_x, float pos_y){
  // Check shadow of circular obstacles
  for(int i=0; i<params.n_cir_obstacles; i++){
    if(checkIntersectionCircle(params.light_x, params.light_y, pos_x, pos_y,
                               x_cir_obs[i], y_cir_obs[i], r_cir_obs[i]))
      return 1;
  }

  // Check shadow rectangular walls
  for(int i=0; i<params.nobstacles; i++){
    if(checkIntersectionLine(params.light_x, params.light_y, pos_x, pos_y,
                             x1obs[i], y1obs[i], x1obs[i], y2obs[i])) //left
      return 1;
    if(checkIntersectionLine(params.light_x, params.light_y, pos_x, pos_y,
                             x1obs[i], y2obs[i], x2obs[i], y2obs[i])) //top
      return 1;
    if(checkIntersectionLine(params.light_x, params.light_y, pos_x, pos_y,
                             x2obs[i], y2obs[i], x2obs[i], y1obs[i])) //right
      return 1;
    if(checkIntersectionLine(params.light_x, params.light_y, pos_x, pos_y,
                             x2obs[i], y1obs[i], x1obs[i], y1obs[i])) //bottom
      return 1;
  }
  return 0;
}

__global__
void updatePhase(float2 *pos, float* phase,
                 float spacing, float max_d, float min_d)
{
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (index >= params.nCells) return;

  int light_visible = 1;
  float2 light = make_float2(params.light_x, params.light_y);
  float dist = length(pos[index]-light);

  if(params.light_shadow){
    if(checkIntersection(pos[index].x, pos[index].y))
      light_visible = 0;
  }
  if(!light_visible){
    //dist = max_d;//phase[index] = 9999999999.0f;  // if Phase[index] > 10000000.0f bots won't modulate
    if(params.light_shadow==1) phase[index] = -(params.Nx-1)*params.rise_period;
    if(params.light_shadow==2) phase[index] = 9999999999.0f;
  }
  else{
    phase[index]= (min_d- dist)/(spacing)*params.rise_period;
  }
  // Noise

}



// Calculate COG
template <unsigned int blockSize>
__global__ void calcCOG(float2 *g_idata, float2 *g_odata, int n)
{
  extern __shared__ float2 sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize) + tid;
  unsigned int gridSize = blockSize*gridDim.x;
  sdata[tid] = make_float2(0.0f, 0.0f);
  while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
  __syncthreads();
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >= 64) { sdata[tid] += sdata[tid + 32]; __syncthreads(); }
    if (blockSize >= 32) { sdata[tid] += sdata[tid + 16]; __syncthreads(); }
    if (blockSize >= 16) { sdata[tid] += sdata[tid + 8]; __syncthreads(); }
    if (blockSize >= 8) { sdata[tid] += sdata[tid + 4]; __syncthreads(); }
    if (blockSize >= 4) { sdata[tid] += sdata[tid + 2]; __syncthreads(); }
    if (blockSize >= 2) { sdata[tid] += sdata[tid + 1]; __syncthreads(); }
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

template <unsigned int blockSize>
__global__ void calcCOG1(float2 *g_idata, float2 *g_odata, int n, float mul)
{
  extern __shared__ float2 sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize) + tid;
  unsigned int gridSize = blockSize*gridDim.x;
  sdata[tid] = make_float2(0.0f, 0.0f);
  while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
  __syncthreads();
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >= 64) { sdata[tid] += sdata[tid + 32]; __syncthreads(); }
    if (blockSize >= 32) { sdata[tid] += sdata[tid + 16]; __syncthreads(); }
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8]; __syncthreads(); }
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4]; __syncthreads(); }
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2]; __syncthreads(); }
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1]; __syncthreads(); }
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0]*mul;
    g_odata[blockIdx.x].y = g_odata[blockIdx.x].y+2000.0f;   // Used in shader to distinguish centroids

  }

}

__device__ float hue2rgb(float p, float q, float t){
  if(t < 0) t += 1;
  if(t > 1) t -= 1;
  if(t < 1.0/6.0) return p + (q - p) * 6.0 * t;
  if(t < 1.0/2.0) return q;
  if(t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
  return p;
}
__device__ void hslToRgb(float h, float s, float l, float &r, float &g, float &b){


  if(s == 0){
    r = l; // achromatic
    g = l;
    b = l;
  }else{

    float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
    float p = 2.0 * l - q;
    r = hue2rgb(p, q, h + 1.0/3.0);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1.0/3.0);
  }
}

__device__ void rgbToHsl(float r, float g, float b, float &h, float &s, float &l){

  float max = fmaxf(fmaxf(r, g), b);
  float min = fminf(fminf(r, g), b);
  h = (max + min) / 2;
  s = (max + min) / 2;
  l = (max + min) / 2;

  if(max == min){
    h = s = 0; // achromatic
  }else{
    float d = max - min;
    s = l > 0.5 ? d / (2.0 - max - min) : d / (max + min);
    if(max==r)
      h = (g - b) / d + (g < b ? 6.0 : 0.0);
    else if(max==g)
      h = (b - r) / d + 2.0;
    else
      h = (r - g) / d + 4.0;
    h /= 6.0;
  }

}

// Update color
__global__
void updateCol_k(float *rad, float4 *col, float2* pos, float* phase, int* dead)
{
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (index >= params.nCells) return;
  float radd = FETCH(rad, index);
  float4 col1 = FETCH(col, index);
  if (dead[index]) {
    col1.x = 0.0f;
    col1.y = 0.0f;
    col1.z = 0.0f;
  }
  else {
    col1.x = 30.0f / 255.0f;// *(params.max_radius - radd) / (params.max_radius - params.min_radius); //red component
    col1.y = (20.0f + (200.0f - 20.0f)*powf(params.max_radius - radd, 2.0f) / powf(params.max_radius - params.min_radius, 2.0f))/ 255.0f; //green component
    col1.z = (30.0f + (210.0f - 30.0f)*powf(radd-params.min_radius, 0.5f) / powf(params.max_radius - params.min_radius, 0.5f)) / 255.0f; //blue component

    //float h = 0, s = 0, l = 0;
    //rgbToHsl(col1.x, col1.y, col1.z, h, s, l);
    //hslToRgb(h, s, l / 2.0, col1.x, col1.y, col1.z);

    //if (radd < (params.min_radius + params.max_radius) / 2) {
    //	col1.y = 1.0f;     // green component
    //	col1.z = 2.0f*(radd - params.min_radius) / (params.max_radius - params.min_radius);     // blue component
    //}
    //else {
    //	col1.y = 2.0f*(params.max_radius - radd) / (params.max_radius - params.min_radius);     // green component
    //	col1.z = 1.0f;     // blue component
    //}
    if (params.display_shadow) {
      if (checkIntersection(pos[index].x, pos[index].y)) {
        //if(phase[index]>10000000.0f){
        float h = 137;
        float s = 36;
        float l = 42;
        rgbToHsl(col1.x, col1.y, col1.z, h, s, l);
        hslToRgb(h, s, l / 2.0, col1.x, col1.y, col1.z);
      }
    }
  }
  col[index] = col1;
}

// calculate grid hash value for each particlebot
__global__
void calcHashD(uint   *gridParticlebotHash,  // output
               uint   *gridParticlebotIndex, // output
               float2 *pos)               // input: positions

{
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (index >= params.nCells) return;

  volatile float2 p = pos[index];

  // get address in grid
  int2 gridPos = calcGridPos(make_float2(p.x, p.y));
  uint hash = calcGridHash(gridPos);

  // store grid hash and particlebot index
  gridParticlebotHash[index] = hash;
  gridParticlebotIndex[index] = index;
}

// rearrange particlebot data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float2 *sortedPos,        // output: sorted positions
                                  float2 *sortedVel,        // output: sorted velocities
                                  float *sortedRad,         // output: sorted radii
                                  uint   *gridParticlebotHash, // input: sorted grid hashes
                                  uint   *gridParticlebotIndex,// input: sorted particlebot indices
                                  float2 *oldPos,           // input: sorted position array
                                  float2 *oldVel,           // input: sorted velocity array
                                  float *oldRad)            // input: *(un)*sorted radii array

{
  extern __shared__ uint sharedHash[];    // blockSize + 1 elements
  uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

  uint hash;

  // handle case when no. of particlebots not multiple of block size
  if (index < params.nCells)
    {
      hash = gridParticlebotHash[index];

      // Load hash data into shared memory so that we can look
      // at neighboring particlebot's hash value without loading
      // two hash values per thread
      sharedHash[threadIdx.x+1] = hash;

      if (index > 0 && threadIdx.x == 0)
        {
          // first thread in block must load neighbor particlebot hash
          sharedHash[0] = gridParticlebotHash[index-1];
        }
    }

  __syncthreads();

  if (index < params.nCells)
    {
      // If this particlebot has a different cell index to the previous
      // particlebot then it must be the first particlebot in the cell,
      // so store the index of this particlebot in the cell.
      // As it isn't the first particlebot, it must also be the cell end of
      // the previous particlebot's cell

      if (index == 0 || hash != sharedHash[threadIdx.x])
        {
          cellStart[hash] = index;

          if (index > 0)
            cellEnd[sharedHash[threadIdx.x]] = index;
        }

      if (index == params.nCells - 1)
        {
          cellEnd[hash] = index + 1;
        }

      // Now use the sorted index to reorder the pos and vel data
      uint sortedIndex = gridParticlebotIndex[index];
      float2 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
      float2 vel = FETCH(oldVel, sortedIndex);       // see particlebot_kernel.cuh
      float rad = FETCH(oldRad, sortedIndex);
      sortedRad[index] = rad;
      sortedPos[index] = pos;
      sortedVel[index] = vel;
    }


}

// collide two spheres using DEM method
__device__
void collideSpheres(float2 posA, float2 posB,
                    float2 velA, float2 velB,
                    float radiusA, float radiusB,
                    float attraction, float2 &force,
                    float &forcea, float &forcer)
{
  // calculate relative position
  float2 relPos = posB - posA;

  float dist = length(relPos);
  float collideDist = radiusA + radiusB;
  float2 tempforce = make_float2(0.0f,0.0f);

  if (dist < collideDist)
    {
      float2 norm = relPos / dist;

      // relative velocity
      // float2 relVel = (massB*velB-massA*velA)/massA;
      float2 relVel = velB - velA;
      // relative tangential velocity
      float2 tanVel = relVel - (dot(relVel, norm) * norm);

      // spring force
      tempforce += (-params.spring*(collideDist - dist) * norm);
      // dashpot (damping) force
      tempforce += params.damping*relVel;
      // tangential shear force
      tempforce += params.shear*tanVel;

      force += tempforce;
      forcer += length(tempforce);
    }
  //else if (massA == massB){
  else {
    // Attraction Force
    float int1 = 0.0009;
    float int2 = 0.0019;
    float min_attr = 2.5f;
    if ((dist - collideDist) < int1) {
      tempforce += min_attr*(relPos / dist);
    }
    else if ((dist - collideDist) < int2) {
      tempforce += (min_attr+
                    (attraction/__powf(int2,2.0f) - min_attr)/(int2-int1)* ((dist-collideDist) - int1) )*(relPos / dist);
    }
    else {
      tempforce += (attraction * (relPos / dist) / __powf(dist - collideDist, 2.0f));
    }
    force += tempforce;
    forcea += length(tempforce);
  }
}

// collide a particlebot against all other particlebots in a given cell
__device__
void collideCell(int2    gridPos,
                 uint    index,
                 float2  pos,
                 float2  vel,
                 float   rad,
                 float2 *oldPos,
                 float2 *oldVel,
                 float  *oldRad,
                 uint   *cellStart,
                 uint   *cellEnd,
                 float2 &force,
                 float &forcea,
                 float &forcer,
                 uint   *gridParticlebotIndex
                 )
{
  uint gridHash = calcGridHash(gridPos);

  // get start of bucket for this cell
  uint startIndex = FETCH(cellStart, gridHash);

  //float mass = params.mass;
  //float mass2 = params.mass;
  float attFact1 = 1.0f;
  float attFact2 = 1.0f;
  if (startIndex != 0xffffffff)          // cell is not empty
    {
      // iterate over particlebots in this cell
      uint endIndex = FETCH(cellEnd, gridHash);

      //
      if (params.nDead == -1 && gridParticlebotIndex[index] == params.nCells-1)
        {
          attFact1 = params.attractionFactor;
          //mass = mass*params.massFactor;
        }
      for (uint j=startIndex; j<endIndex; j++)
        {
          //mass2 = params.mass;
          attFact2 = 1.0f;
          if (j != index)                // check not colliding with self
            {
              if (params.nDead == -1 && gridParticlebotIndex[j] == params.nCells-1)
                {
                  attFact2 = params.attractionFactor;
                  //mass2 = mass*params.massFactor;
                }
              float2 pos2 = FETCH(oldPos, j);
              float2 vel2 = FETCH(oldVel, j);
              float rad2 = FETCH(oldRad, j);
              collideSpheres(pos, pos2, vel, vel2, rad, rad2, //mass, mass2,
                             params.attraction*attFact2*attFact1, force, forcea, forcer);
            }
        }
    }
}



__global__
void collideD(float2 *newVel,               // output: new velocity
              float *absForce_a,
              float *absForce_r,
              float2 *oldPos,               // input: sorted positions
              float2 *oldVel,               // input: sorted velocities
              float  *oldRad,               // input: sorted radii
              uint   *gridParticlebotIndex,     // input: sorted particlebot indices
              uint   *cellStart,
              uint   *cellEnd,
              float deltaTime)
{
  uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

  if (index >= params.nCells) return;

  // read particlebot data from sorted arrays
  float2 pos = FETCH(oldPos, index);
  float2 vel = FETCH(oldVel, index);
  float rad = FETCH(oldRad, index);


  // get address in grid
  int2 gridPos = calcGridPos(pos);

  // examine neighbouring cells
  float2 force = make_float2(0.0f);

  uint originalIndex = gridParticlebotIndex[index];

  float absforce_a = 0.0f;
  float absforce_r = 0.0f * absForce_r[originalIndex];


  for (int y=-2; y<=2; y++)
    {
      for (int x=-2; x<=2; x++)
        {
          int2 neighbourPos = gridPos + make_int2(x, y);
          collideCell(neighbourPos, index, pos, vel, rad,
                      oldPos, oldVel, oldRad, cellStart,
                      cellEnd, force, absforce_a, absforce_r, gridParticlebotIndex);
        }
    }
  // Obstacles
  float2 dir, relVel, tanVel, temp_force;
  for (int i = 0; i<params.n_cir_obstacles; i++) {
    float dist_2 = powf(pos.x - x_cir_obs[i], 2) + powf(pos.y - y_cir_obs[i], 2);
    if (dist_2 < powf(rad + r_cir_obs[i], 2)) {
      // dir is normal direction
      dir = make_float2(-pos.x + x_cir_obs[i], -pos.y + y_cir_obs[i]);
      dir = dir / length(dir);
      //pos = make_float2(x_cir_obs[i], y_cir_obs[i]) - (r_cir_obs[i] + rad)*dir;

      // relative velocity
      relVel = -vel;

      // relative tangential velocity
      tanVel = relVel - (dot(relVel, dir) * dir);

      temp_force = make_float2(0.0f, 0.0f);
      // spring force
      temp_force += (2.0f*params.spring*(rad + r_cir_obs[i] - powf(dist_2, 0.5f)) * (-dir));// +0.25f*dir2));
      // dashpot (damping) force
      temp_force += params.damping*relVel;
      // tangential shear force
      temp_force += params.shear*tanVel;

      force += temp_force;
      absforce_r += length(temp_force);
    }
  }
  int intersect = 0;
  float overlapdist = 0.0f;
  for (int i = 0; i<params.nobstacles; i++) {
    intersect = 0;
    if (pos.y > y1obs[i] && pos.y < y2obs[i]) {
      if (pos.x > x1obs[i] - rad && pos.x < x2obs[i] - rad) {
        intersect = 1;
        dir = make_float2(1.0f, 0.0f);
        overlapdist = pos.x - x1obs[i] + rad;
      }
      if (pos.x < x2obs[i] + rad && pos.x > x1obs[i] + rad) {
        intersect = 1;
        dir = make_float2(-1.0f, 0.0f);
        overlapdist = -pos.x + x2obs[i] + rad;
      }
    }
    else if (pos.x > x1obs[i] && pos.x < x2obs[i]) {
      if (pos.y > y1obs[i] - rad && pos.y < y2obs[i] - rad) {
        intersect = 1;
        dir = make_float2(0.0f, 1.0f);
        overlapdist = pos.y - y1obs[i] + rad;
      }
      if (pos.y < y2obs[i] + rad && pos.y > y1obs[i] + rad) {
        intersect = 1;
        dir = make_float2(0.0f, -1.0f);
        overlapdist = -pos.y + y2obs[i] + rad;
      }
    }
    else if (powf(pos.x - x2obs[i], 2) + powf(pos.y - y2obs[i], 2) < powf(rad, 2)) {
      dir = make_float2(pos.x - x2obs[i], pos.y - y2obs[i]);
      dir = -dir / length(dir);
      intersect = 1;
      overlapdist = rad - powf(powf(pos.x - x2obs[i], 2) + powf(pos.y - y2obs[i], 2), 0.5f);
    }
    else if (powf(pos.x - x1obs[i], 2) + powf(pos.y - y2obs[i], 2) < powf(rad, 2)) {
      dir = make_float2(pos.x - x1obs[i], pos.y - y2obs[i]);
      dir = -dir / length(dir);
      intersect = 1;
      overlapdist = rad - powf(powf(pos.x - x1obs[i], 2) + powf(pos.y - y2obs[i], 2), 0.5f);
    }
    else if (powf(pos.x - x1obs[i], 2) + powf(pos.y - y1obs[i], 2) < powf(rad, 2)) {
      dir = make_float2(pos.x - x1obs[i], pos.y - y1obs[i]);
      dir = -dir / length(dir);
      intersect = 1;
      overlapdist = rad - powf(powf(pos.x - x1obs[i], 2) + powf(pos.y - y1obs[i], 2), 0.5f);
    }
    else if (powf(pos.x - x2obs[i], 2) + powf(pos.y - y1obs[i], 2) < powf(rad, 2)) {
      dir = make_float2(pos.x - x2obs[i], pos.y - y1obs[i]);
      dir = -dir / length(dir);
      intersect = 1;
      overlapdist = rad - powf(powf(pos.x - x2obs[i], 2) + powf(pos.y - y1obs[i], 2), 0.5f);
    }
    if (intersect) {
      relVel = -vel;

      // relative tangential velocity
      tanVel = relVel - (dot(relVel, dir) * dir);

      temp_force = make_float2(0.0f, 0.0f);
      // spring force
      temp_force += (-2.0f*params.spring* overlapdist * dir);
      // dashpot (damping) force
      temp_force += params.damping*relVel;
      // tangential shear force
      temp_force += params.shear*tanVel;

      force += temp_force;
      absforce_r += length(temp_force);
    }
  }


  float friction = params.friction;
  float gravity = params.gravity;
  //if the particlebot is not moving does it overcome the static friction
  if(params.nDead == -1 && gridParticlebotIndex[index] == params.nCells-1)
    {
      friction *= params.frictionFactor;  // Adjust friction for weight
      gravity *= params.massFactor;       // Adjust effective gravity for transport weight
    }
  if(length(vel)<0.000001f && length(force)<
     (2.0f*friction*gravity))
    force = make_float2(0.0f);

  //update velocity
  if(params.nDead == -1 && gridParticlebotIndex[index] == params.nCells-1)
    vel = vel + force/params.massFactor*deltaTime;
  else
    vel = vel + force*deltaTime;

  //Apply kinetic friction once direction of motion is known
  if (length(vel) < (friction*gravity
                     *deltaTime))
    vel = make_float2(0.0f);
  else
    vel -= (friction*gravity*
            deltaTime)*(vel / length(vel));

  // write new velocity back to original unsorted location
  newVel[originalIndex] = vel;
  absForce_a[originalIndex] = absforce_a;
  absForce_r[originalIndex] = absforce_r;
}
#endif
