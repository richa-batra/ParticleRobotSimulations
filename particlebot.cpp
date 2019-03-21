#include <GL/glew.h>
#include <GL/freeglut.h>

#include "particlebot.h"
#include "particlebot.cuh"
#include "particlebot_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

using namespace std;

inline float frand()
{
  return rand() / (float)RAND_MAX;
}

inline float length(float x, float y){
  return powf(powf(x, 2.0f) + powf(y, 2.0f), 0.5f);
}

Particlebot::Particlebot(SimParams simparams) :
  hPos(0),
  hVel(0),
  dPos(0),
  dVel(0),
  time(0)
{
  params = simparams;

  particlebotConfigSize.x = particlebotConfigSize.y = 0;

  _initialize();
}

Particlebot::~Particlebot()
{
  _finalize();
}

// create a color ramp
//void colorRamp(float t, float *r)
//{
//    const int ncolors = 7;
//    float c[ncolors][3] =
//    {
//        { 1.0, 0.0, 0.0, },
//        { 1.0, 0.5, 0.0, },
//        { 1.0, 1.0, 0.0, },
//        { 0.0, 1.0, 0.0, },
//        { 0.0, 1.0, 1.0, },
//        { 0.0, 0.0, 1.0, },
//        { 1.0, 0.0, 1.0, },
//    };
//    t = t * (ncolors-1);
//    int i = (int) t;
//    float u = t - floor(t);
//    r[0] = lerp(c[i][0], c[i+1][0], u);
//    r[1] = lerp(c[i][1], c[i+1][1], u);
//    r[2] = lerp(c[i][2], c[i+1][2], u);
//}

void
Particlebot::_initialize()
{
  // allocate host storage
  hPos = new float[(params.nCells+params.centroid_steps+1)*2];
  hVel = new float[params.nCells*2];
  memset(hPos, 0, (params.nCells+params.centroid_steps+1)*2*sizeof(float));
  memset(hVel, 0, params.nCells*2*sizeof(float));

  hRad = new float[params.nCells+params.centroid_steps+1];
  hDead = new int[params.nCells];
  hphase = new float[params.nCells];
  hfreq = new float[params.nCells];
  memset(hRad, 0, (params.nCells+params.centroid_steps)*sizeof(float));
  memset(hDead, 0, (params.nCells) * sizeof(int));
  memset(hphase, 0 , params.nCells * sizeof(float));
  memset(hfreq, 0, params.nCells * sizeof(float));

  hCellStart = new uint[params.numCells];
  memset(hCellStart, 0, params.numCells*sizeof(uint));

  hCellEnd = new uint[params.numCells];
  memset(hCellEnd, 0, params.numCells*sizeof(uint));

  // allocate GPU data
  unsigned int memSize = sizeof(float) * 2 * params.nCells;


  posVbo = createVBO(memSize+sizeof(float) * 2 * (params.centroid_steps+1));
  registerGLBufferObject(posVbo, &cuda_posvbo_resource);
  radVbo = createVBO(memSize+sizeof(float)*(params.centroid_steps+1));
  registerGLBufferObject(radVbo, &cuda_radvbo_resource);

  //radius dependant color
  //fill color buffer
  colorVBO = createVBO((params.nCells+params.centroid_steps+1)*4*sizeof(float));
  registerGLBufferObject(colorVBO, &cuda_colorvbo_resource);
  glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
  float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
  float *ptr = data;

  for (uint i=0; i<params.nCells; i++)
    {
      //float t = i / (float) params.nCells;

      //colorRamp(t, ptr);
      //ptr+=3;
      // *ptr++ = 1.0f;

      *ptr++ = 1.0f;
      *ptr++ = 1.0f;
      *ptr++ = 1.0f;
      *ptr++ = 1.0f;
    }
  for(uint i=0;i<params.centroid_steps;i++){
    *ptr++ = 1.0f;
    *ptr++ = 0.0f;
    *ptr++ = 0.0f;
    *ptr++ = 0.8f;
  }
  *ptr++ = 1.0f;
  *ptr++ = 0.0f;
  *ptr++ = 0.0f;
  *ptr++ = 1.0f;

  glUnmapBuffer(GL_ARRAY_BUFFER);


  allocateArray((void **)&dVel, memSize);
  allocateArray((void **)&dSortedPos, memSize);
  allocateArray((void **)&tempPos1, memSize);
  allocateArray((void **)&tempPos2, memSize);
  allocateArray((void **)&dSortedVel, memSize);
  allocateArray((void **)&dSortedRad, sizeof(float) * params.nCells);
  allocateArray((void **)&dphase, sizeof(float) * params.nCells);
  allocateArray((void **)&dfreq, sizeof(float) * params.nCells);
  allocateArray((void **)&dAbsForce_a, sizeof(float) * params.nCells);
  allocateArray((void **)&dAbsForce_r, sizeof(float) * params.nCells);
  allocateArray((void **)&dGridParticleHash, params.nCells*sizeof(uint));
  allocateArray((void **)&dGridParticleIndex, params.nCells*sizeof(uint));
  allocateArray((void **)&dCellStart, params.numCells*sizeof(uint));
  allocateArray((void **)&dCellEnd, params.numCells*sizeof(uint));

  allocateArray((void **)&dDead, sizeof(int) * params.nCells);
  allocateArray((void **)&dState, sizeof(curandState) * params.nCells);

  setParameters(&params);

  curand_setup(dState, params.nCells);
}



void
Particlebot::update(float deltaTime, float sort_interval)
{

  if (time > params.max_time) {
    exit(0);
  }

  if (time >= params.time_to_dead && time < params.time_to_dead+deltaTime) {
    int count = 0;
    int i;
    vector<int> inds;
    inds.reserve(params.nCells);
    for (i = 0; i < params.nCells; i++)
      inds.push_back(i);

    while (count < params.nDead) {
      i = rand() % inds.size();
      hDead[inds[i]] = 1;
      inds.erase(inds.begin() + i);
      count++;
    }

    copyArrayToDevice(dDead, hDead, 0 * sizeof(int), params.nCells * sizeof(int));
  }

  float *dPos;
  float *dRad;
  float *dCol;

  dPos = (float *)mapGLBufferObject(&cuda_posvbo_resource);
  dRad = (float *)mapGLBufferObject(&cuda_radvbo_resource);
  dCol = (float *)mapGLBufferObject(&cuda_colorvbo_resource);
  unmapGLBufferObject(cuda_posvbo_resource);
  unmapGLBufferObject(cuda_radvbo_resource);
  unmapGLBufferObject(cuda_colorvbo_resource);

  if (time - params.centroid_int*floor(time / params.centroid_int) < deltaTime) {
    calcCOG(dPos, tempPos1, tempPos2, params.nCells, time, params.centroid_steps, params.centroid_int);
  }

  if (params.control == LIGHT_WAVE ) {
    if (time - params.phase_update_interval*floor(time / params.phase_update_interval) < deltaTime)
      {
        copyArrayFromDevice(hPos, dPos, 0, sizeof(float) * 2 * params.nCells);
        float min_d = 0;
        float max_d = 0;
        float dist = 0;
        for (uint i = 0; i < params.nCells; i++) {
          dist = powf(powf(params.light_x - hPos[i * 2], 2) + powf(params.light_y - hPos[i * 2 + 1], 2), 0.5f);
          if (i == 0) {
            max_d = dist;
            min_d = dist;
          }
          else {
            min_d = (min_d < dist ? min_d : dist);
            max_d = (max_d > dist ? max_d : dist);
          }
        }
        float spacing = 2.0f*params.min_radius;
        //float spacing = (max_s-min_s)/particlebotConfigSize.x;
        //if (params.config == CONFIG_HEX || params.config == CONFIG_BLOB)
        //	spacing = 1.73205080757f * params.min_radius;
        updatePhase(dPos, dphase, spacing, max_d, min_d, params.nCells);
        if (params.phase_std) {
          add_normal_noise(dState, dphase, params.phase_std, params.nCells);
        }
      }
    if (time >= 0)
      {
        updateRad_light_wave(dPos, dAbsForce_a, dAbsForce_r, dRad, dphase, time,
                             deltaTime, dDead, params.nCells);
      }
  }


  integrateSystem(
                  dPos,
                  dVel,
                  dRad,
                  deltaTime,
                  params.nCells,
                  time);

  updateCol(dRad, dCol, params.nCells, dPos, dphase, dDead);

  if (time - sort_interval*floor(time / sort_interval) < deltaTime){
    // calculate grid hash
    calcHash(
             dGridParticleHash,
             dGridParticleIndex,
             dPos,
             params.nCells);

    // sort particles based on hash
    sortParticlebots(dGridParticleHash, dGridParticleIndex, params.nCells);


  }

  // reorder particle arrays into sorted order and
  // find start and end of each cell
  reorderDataAndFindCellStart(
                              dCellStart,
                              dCellEnd,
                              dSortedPos,
                              dSortedVel,
                              dSortedRad,
                              dGridParticleHash,
                              dGridParticleIndex,
                              dPos,
                              dVel,
                              dRad,
                              params.nCells,
                              params.numCells);

  collide(
          dVel,
          dAbsForce_a, dAbsForce_r,
          dSortedPos,
          dSortedVel,
          dSortedRad,
          dGridParticleIndex,
          dCellStart,
          dCellEnd,
          params.nCells,
          params.numCells, deltaTime);


  time = time + deltaTime;
}


void
Particlebot::dumpParticlebot(uint start, uint count, FILE *fp, float dump_interval, uint testing, float light_x, float light_y)
{
  float sumX = 0.0f;
  float sumY = 0.0f;

  if (time - dump_interval*floor(time / dump_interval) > 0.01f)
    return;
  copyArrayFromDevice(hPos, 0, &cuda_posvbo_resource, sizeof(float) * 2 * count);
  copyArrayFromDevice(hVel, dVel, 0, sizeof(float) * 2 * count);
  copyArrayFromDevice(hRad, 0, &cuda_radvbo_resource, sizeof(float)*count);

  // Write Column header row if called for the first time
  if (time == 0) {
    fprintf(fp, "Seed, %u\n", params.seed);
    fprintf(fp, "Time,");
    if(testing){
      for (uint i = start; i < start + count; i++)
        {
          fprintf(fp, "Particlebot_%d_xpos, Particlebot_%d_ypos,", i, i);
        }
      for (uint i = start; i < start + count; i++)
        {
          fprintf(fp, "Particlebot_%d_xvel, Particlebot_%d_yvel,", i, i);
        }
      for (uint i = start; i < start + count; i++)
        {
          fprintf(fp, "Particlebot_%d_rad,", i);
        }
    }
    fprintf(fp,"Centroid X, Centroid Y, Distance");
    fprintf(fp, "\n");
  }
  // Write time as the first column
  fprintf(fp, "%f,", time);

  if(testing){
    // Write Positions
    for (uint i = start; i < start + count; i++)
      {
        fprintf(fp,"%f, %f,", hPos[i * 2 + 0], hPos[i * 2 + 1]);
      }
    //Velocities
    for (uint i = start; i < start + count; i++)
      {
        fprintf(fp,"%f, %f,", hVel[i * 2 + 0], hVel[i * 2 + 1]);
      }
    // Radii
    for (uint i = start; i < start + count; i++)
      {
        fprintf(fp, "%f,", hRad[i]);
      }
  }
  //write centroid location and distance from light
  for (uint i = start; i < start + count; i++)
    {
      sumX += hPos[i * 2 + 0];
      sumY += hPos[i * 2 + 1];
    }
  fprintf(fp,"%f, %f, %f,", sumX/(float)count, sumY/(float)count,
          powf(powf(sumX/(float)count-light_x,2.0)+powf(sumY/(float)count-light_y,2.0),0.5));

  fprintf(fp, "\n");
  printf("%f %f %f \n", time, sumX / (float)count, sumY / (float)count);
}

void
Particlebot::loadFromFile(uint start, uint count, FILE *fp, float dump_interval)
{
  fseek(fp, 0, SEEK_SET);
  int c = fgetc(fp);
  int bytes = 1;
  int line_start_1 = 0;
  int line_start_2 = 0;
  while(c!=EOF){
    if (c=='\n'){
      line_start_1 = line_start_2;
      line_start_2 = bytes;
    }
    c = fgetc(fp);
    bytes+=1;
  }
  fseek(fp, line_start_1-bytes, SEEK_END);


  fscanf(fp, "%f,", &time);

  for (uint i = start; i < start + count; i++)
    {
      fscanf(fp,"%f, %f,", &hPos[i * 2 + 0], &hPos[i * 2 + 1]);
    }
  //Velocities
  for (uint i = start; i < start + count; i++)
    {
      fscanf(fp,"%f, %f,", &hVel[i * 2 + 0], &hVel[i * 2 + 1]);
    }
  // Radii
  for (uint i = start; i < start + count; i++)
    {
      fscanf(fp, "%f,", &hRad[i]);
    }

  setArray(RADII, hRad, 0, params.nCells);
  setArray(POSITION, hPos, 0, params.nCells);
  setArray(VELOCITY, hVel, 0, params.nCells);

  printf("Time = %f\n", time);

}

void
Particlebot::initGrid(uint2 size, float spacing, float jitter, uint nCells)
{
  float xs = size.x*spacing/2.0f;
  float ys = size.y*spacing/2.0f;
  for (uint y=0; y<size.y; y++)
    {

      for (uint x=0; x<size.x; x++)
        {
          uint i = (y*size.x) + x;

          if (i < nCells)
            {
              hPos[i*2] = (spacing * x) + params.min_radius - xs + (frand()*2.0f-1.0f)*jitter;
              hPos[i * 2 + 1] = 0;// (spacing * y) + params.max_radius - ys + (frand()*2.0f - 1.0f)*jitter;

              hVel[i*2] = 0.0f;
              hVel[i*2+1] = 0.0f;

            }
        }
    }
}

void
Particlebot::initHexGrid(uint nCells, float spacing)
{
  float dirs[7][3] = { { 1.0,0.0,0.0 },
                       { 0.5,0.0,powf(3,0.5f)*0.5f },
                       { -0.5,0.0,powf(3,0.5f)*0.5f },
                       { -1.0,0.0,0.0 },
                       { -0.5,0.0,-powf(3,0.5f)*0.5f },
                       { 0.5,0.0,-powf(3,0.5f)*0.5f },
                       { 1.0,0.0,0.0 } };
  float ws[6][6] = { { -1.0,1.0,0.0,0,0,0 },
                     { 0,-1,1,0,0,0 },
                     { 0,0,-1,1,0,0 },
                     { 0,0,0,-1,1,0 },
                     { 0,0,0,0,-1,1 },
                     { 1,0,0,0,0,-1 } };
  int i = 0;
  hPos[i * 2] = 0.0f;
  hPos[i * 2 + 1] = 0.0f;


  hVel[i * 2] = 0.0f;
  hVel[i * 2 + 1] = 0.0f;

  i++;
  int n_ring = 1;
  while (i<nCells) {
    for (int k = 0; k<6; k++) {
      for (int j = 0; j<n_ring; j++) {
        hPos[i * 2] = dirs[k][0] * (n_ring - j)*spacing + dirs[k + 1][0] * spacing*j;
        hPos[i * 2 + 1] = dirs[k][2] * (n_ring - j)*spacing + dirs[k + 1][2] * spacing*j;

        hVel[i * 2] = 0.0f;
        hVel[i * 2 + 1] = 0.0f;
        i++;
        if (i == nCells) break;
      }
      if (i == nCells) break;
    }
    n_ring++;
  }
  particlebotConfigSize.x = particlebotConfigSize.y = n_ring*2;
  printf("%d\n", particlebotConfigSize.x);
}



void
Particlebot::reset()
{
  time = 0;
  switch (params.config)
    {
    default:
    case CONFIG_BLOB_UPLEFT:
      {
        assert(params.nCells == 10);
        float r = params.min_radius;
        float theta = -CUDART_PI_F/180.0f*30.0f;

        for(int i = 0; i < params.nCells; i++){
          hVel[i*2+0] = 0;
          hVel[i*2+1] = 0;
        }

        hPos[1] = r;
        hPos[3] = r;
        hPos[5] = -r;
        hPos[7] = -r;
        hPos[9] = -(1.0f+powf(3.0f,0.5f))*r;
        hPos[11] = 0.0f;
        hPos[13] = 0.0f;
        hPos[15] = 2.0f*r;
        hPos[17] = 2.0f*r;
        hPos[19] = (1.0f+powf(3.0f,0.5f))*r;

        hPos[0] = -r;
        hPos[2] = r;
        hPos[4] = -r;
        hPos[6] = r;
        hPos[8] = 0.0f;
        hPos[10] = -(1.0f+powf(3.0f,0.5f))*r;
        hPos[12] = (1.0f+powf(3.0f,0.5f))*r;
        hPos[14] = -(1.0f+powf(3.0f,0.5f))*r;
        hPos[16] = (1.0f+powf(3.0f,0.5f))*r;
        hPos[18] = 0.0f;


        particlebotConfigSize.x = 4;
        particlebotConfigSize.y = 4;


      }
      break;
    case CONFIG_LIGHTTEST_7:
      {
        assert(params.nCells == 10);
        float r = params.min_radius;
        float theta = -CUDART_PI_F / 180.0f*30.0f;

        for (int i = 0; i < params.nCells; i++) {
          hVel[i * 2 + 0] = 0;
          hVel[i * 2 + 1] = 0;
        }

        hPos[1] = 0;
        hPos[3] = r;
        hPos[5] = -r;
        hPos[7] = r;
        hPos[9] = 2.0f*r;
        hPos[11] = -r;
        hPos[13] = -2.0f*r;
        hPos[14] = powf(3.0f, 0.5f)*r;
        hPos[16] = 0.0f;
        hPos[18] = powf(3.0f, 0.5f)*2.0f*r;

        hPos[0] = 0.0f;
        hPos[2] = -powf(3.0f, 0.5f)*r;
        hPos[4] = powf(3.0f, 0.5f)*r;
        hPos[6] = powf(3.0f, 0.5f)*r;
        hPos[8] = 0.0f;
        hPos[10] = -powf(3.0f, 0.5f)*r;
        hPos[12] = 0;
        hPos[15] = 3.0f*r;
        hPos[17] = 4.0f*r;
        hPos[19] = 2.0f*r;


        particlebotConfigSize.x = 4;
        particlebotConfigSize.y = 4;


      }
      break;
    case CONFIG_BLOB:
      {
        assert(params.nCells == 10);
        float r = params.min_radius;
        float theta = -CUDART_PI_F / 180.0f*30.0f;

        for (int i = 0; i < params.nCells; i++) {
          hVel[i * 2 + 0] = 0;
          hVel[i * 2 + 1] = 0;
        }

        hPos[0] = r;
        hPos[2] = r;
        hPos[4] = -r;
        hPos[6] = -r;
        hPos[8] = -(1.0f + powf(3.0f, 0.5f))*r;
        hPos[10] = 0.0f;
        hPos[12] = 0.0f;
        hPos[14] = 2.0f*r;
        hPos[16] = 2.0f*r;
        hPos[18] = (1.0f + powf(3.0f, 0.5f))*r;

        hPos[1] = -r;
        hPos[3] = r;
        hPos[5] = -r;
        hPos[7] = r;
        hPos[9] = 0.0f;
        hPos[11] = -(1.0f + powf(3.0f, 0.5f))*r;
        hPos[13] = (1.0f + powf(3.0f, 0.5f))*r;
        hPos[15] = -(1.0f + powf(3.0f, 0.5f))*r;
        hPos[17] = (1.0f + powf(3.0f, 0.5f))*r;
        hPos[19] = 0.0f;


        particlebotConfigSize.x = 4;
        particlebotConfigSize.y = 4;


      }
      break;
    case CONFIG_RANDOM:
      {
        vector< vector<vector<int> > > grid_indices;;
        vector<int>::iterator it;
        for(int i=0; i<params.gridSize.x; i++){
          vector< vector<int> > temp;
          for(int j=0; j<params.gridSize.y; j++){
            vector<int> temp1;
            temp.push_back(temp1);
          }
          grid_indices.push_back(temp);
        }
        particlebotConfigSize.x = (int) ceilf(powf((float)params.nCells, 1.0f / 2.0f));
        int p = 0, v = 0;
        int xg = 0, yg=0, xgs=0, ygs=0;
        uint placed = 0;
        uint start_ind = 0;
        uint max_unsuccessful_placements = 200;
        uint unsuccessful_placements = 0;
        hPos[p++] = 5.0;
        hPos[p++] = 0.0;
        hVel[v++] = 0.0f;
        hVel[v++] = 0.0f;
        xg = ((int)floor((0.0f - params.worldOrigin.x) / params.cellSize.x)) & (params.gridSize.x-1);
        yg = ((int)floor((0.0f - params.worldOrigin.y) / params.cellSize.y)) & (params.gridSize.y-1);
        grid_indices[xg][yg].push_back(0);
        float x =0, y=0, theta = 0, r = 0;
        float old_theta = 0;
        float min_x = 9999999.0;
        float increment_theta = 2 * CUDART_PI_F / 360.0 * 10.0;
        int k;
        for(uint i=1; i<params.nCells; i++)
          {
            printf("Placing %d th disc\n", i);
            if (i == 2) {
              int j = rand() % 2;
              float2 dir = make_float2(hPos[2] - hPos[0], hPos[3] - hPos[1]);
              float l = length(dir.x, dir.y);
              dir.y = dir.y / l;
              dir.x = dir.x / l;
              if (j) {
                dir = make_float2(dir.y, -dir.x);
              }
              else {
                dir = make_float2(-dir.y, dir.x);
              }
              x = (hPos[2] + hPos[0]) / 2.0f + dir.x * params.min_radius;
              y = (hPos[3] + hPos[1]) / 2.0f + dir.y * params.min_radius;
              if (x < min_x)
                min_x = x;

              hPos[p++] = x;
              hPos[p++] = y;

              hVel[v++] = 0.0f;
              hVel[v++] = 0.0f;
              xg = ((int)floor((hPos[2*i] - params.worldOrigin.x) / params.cellSize.x)) & (params.gridSize.x-1);
              yg = ((int)floor((hPos[2*i+1] - params.worldOrigin.y) / params.cellSize.y)) & (params.gridSize.y-1);
              grid_indices[xg][yg].push_back(i);
              continue;
            }
            placed = 0;
            r = params.min_radius;
            while (!placed) {
              // Random int between 0 and i-1 as the starting particlebot
              start_ind = (uint)rand()%i;
              placed = 1;
              if (unsuccessful_placements == max_unsuccessful_placements) {
                unsuccessful_placements = 0;
                r += params.min_radius;
              }
              // Random Direction
              theta = 2 * frand() * CUDART_PI_F;
              x = hPos[2*start_ind] + 2*r*cos(theta);
              y = hPos[2*start_ind+1] + 2*r*sin(theta);
              xgs = ((int)floor((x - params.worldOrigin.x) / params.cellSize.x)) & (params.gridSize.x-1);
              ygs = ((int)floor((y - params.worldOrigin.y) / params.cellSize.y)) & (params.gridSize.y-1);
              for(xg=xgs-1; xg<=xgs+1 & placed; xg++){
                for(yg=ygs-1; yg<=ygs+1 & placed; yg++){
                  for(it = grid_indices[xg][yg].begin(); it != grid_indices[xg][yg].end() & placed; it++){
                    if (length(x - hPos[2 * (*it)], y - hPos[2 * (*it) + 1]) < 2*1.0*params.min_radius) {
                      placed = 0;
                      unsuccessful_placements++;
                      break;
                    }
                  }
                }
              }
              if (!placed)
                continue;
              old_theta = theta;
              int flag = 0;
              // Successfully placed it. Start pivoting till overlap
              while (theta - old_theta < 2 * CUDART_PI_F) {
                theta += increment_theta;
                x = hPos[2 * start_ind] + 2 * r*cos(theta);
                y = hPos[2 * start_ind + 1] + 2 * r*sin(theta);
                xgs = ((int)floor((x - params.worldOrigin.x) / params.cellSize.x)) & (params.gridSize.x-1);
                ygs = ((int)floor((y - params.worldOrigin.y) / params.cellSize.y)) & (params.gridSize.y-1);
                for(xg=xgs-1; xg<=xgs+1; xg++){
                  for(yg=ygs-1; yg<=ygs+1; yg++){
                    for(it = grid_indices[xg][yg].begin(); it != grid_indices[xg][yg].end(); it++){
                      if (length(x - hPos[2 * (*it)], y - hPos[2 * (*it) + 1]) < 2 * 1.0*params.min_radius) {
                        flag = 1;
                        break;
                      }
                    }
                  }
                }
                if (flag) {
                  theta -= increment_theta;
                  break;
                }
              }
              x = hPos[2 * start_ind] + 2 * r*cos(theta);
              y = hPos[2 * start_ind + 1] + 2 * r*sin(theta);
            }
            if (x < min_x)
              min_x = x;
            if (params.nDead == -1 && i == params.nCells - 1)
              {
                x = min_x -	1*params.min_radius*params.radFactor-2*params.min_radius;
                y = 0;
              }

            hPos[p++] = x;
            hPos[p++] = y;
            int temp = (int)floor((x - params.worldOrigin.x) / params.cellSize.x);
            float temp1 = (x - params.worldOrigin.x) / params.cellSize.x;
            xg = ((int)floor((x - params.worldOrigin.x) / params.cellSize.x)) & (params.gridSize.x-1);
            yg = ((int)floor((y - params.worldOrigin.y) / params.cellSize.y)) & (params.gridSize.y-1);
            grid_indices[xg][yg].push_back(i);
            hVel[v++] = 0.0f;
            hVel[v++] = 0.0f;
          }
      }
      break;
    case CONFIG_GRID:
      {
        float jitter = params.max_radius*0.01f;
        uint s = (int)ceilf(powf((float)params.nCells, 1.0f / 2.0f));
        particlebotConfigSize.x = particlebotConfigSize.y  = s;
        initGrid(particlebotConfigSize, params.min_radius*2.0f, jitter, params.nCells);
      }
      break;
    case CONFIG_HEX:
      {
        particlebotConfigSize.x = (int) ceilf(powf((float)params.nCells, 1.0f / 2.0f));
        initHexGrid(params.nCells, params.min_radius*2.0f);
      }
      break;
    case CONFIG_LINE:
      {
        float jitter = params.max_radius*0.00f;
        particlebotConfigSize.x = params.nCells;
        particlebotConfigSize.y = 1;
        initGrid(particlebotConfigSize, params.min_radius*2.0f, jitter, params.nCells);
      }
      break;
    }
  if(!params.Nx)
    params.Nx = particlebotConfigSize.x;
  // Initial allocation stategy of radii (Position based)
  float xs = ceilf(powf((float) params.nCells, 1.0f / 2.0f)) * params.max_radius*0.5f;
  for(uint i=0;i<params.centroid_steps;i++){
    hRad[params.nCells + i] = params.centroid_radius;
    hPos[(params.nCells+i)*2] = -5000.0f;
  }
  hRad[params.nCells+params.centroid_steps] = 0.0f;
  for(uint i=0;i<params.nCells;i++){
    xs = (xs>hPos[i * 2] ? xs : hPos[i * 2]);
  }
  for (uint i = 0; i < params.nCells; i++) {
    hRad[i] = params.min_radius;
    if (params.nDead == -1 && i == params.nCells - 1) {
      hRad[i] = params.min_radius*params.radFactor;
      hDead[i] = 1;
    }
    hphase[i] = 0;
  }


  copyArrayToDevice(dDead, hDead, 0 * sizeof(int), params.nCells * sizeof(int));
  setArray(RADII, hRad, 0, params.nCells+params.centroid_steps+1);

  setArray(PHASE, hphase, 0, params.nCells);
  setArray(FREQUENCY, hfreq, 0, params.nCells);
  setArray(POSITION, hPos, 0, params.nCells+params.centroid_steps+1);
  setArray(VELOCITY, hVel, 0, params.nCells);
}

float* Particlebot::getArray(ParticlebotArray array)
{
  float *hdata = 0;
  float *ddata = 0;
  struct cudaGraphicsResource *cuda_vbo_resource = 0;

  switch (array)
    {
    default:
    case POSITION:
      hdata = hPos;
      ddata = dPos;
      cuda_vbo_resource = cuda_posvbo_resource;
      break;

    case VELOCITY:
      hdata = hVel;
      ddata = dVel;
      break;

    case RADII:
      hdata = hRad;
      ddata = dRad;
      cuda_vbo_resource = cuda_radvbo_resource;
      break;
    }

  copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, params.nCells*4*sizeof(float));
  return hdata;
}

void
Particlebot::setArray(ParticlebotArray array, const float *data, int start, int count)
{
  switch (array)
    {
    default:
    case POSITION:
      unregisterGLBufferObject(cuda_posvbo_resource);
      glBindBuffer(GL_ARRAY_BUFFER, posVbo);
      glBufferSubData(GL_ARRAY_BUFFER, start*2*sizeof(float), count*2*sizeof(float), data);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      registerGLBufferObject(posVbo, &cuda_posvbo_resource);
      break;

    case VELOCITY:
      copyArrayToDevice(dVel, data, start*2*sizeof(float), count*2*sizeof(float));
      break;

    case PHASE:
      copyArrayToDevice(dphase, data, start * sizeof(float), count * sizeof(float));
      break;
    case FREQUENCY:
      copyArrayToDevice(dfreq, data, start * sizeof(float), count * sizeof(float));
      break;

    case RADII:
      unregisterGLBufferObject(cuda_radvbo_resource);
      glBindBuffer(GL_ARRAY_BUFFER, radVbo);
      glBufferSubData(GL_ARRAY_BUFFER, start*1*sizeof(float), count*1*sizeof(float), data);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      registerGLBufferObject(radVbo, &cuda_radvbo_resource);
      break;
    }
}



uint
Particlebot::createVBO(uint size)
{
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return vbo;
}

void
Particlebot::_finalize()
{
  delete [] hPos;
  delete [] hVel;
  delete [] hCellStart;
  delete [] hCellEnd;
  delete [] hRad;
  delete [] hphase;
  delete [] hfreq;

  freeArray(dVel);
  freeArray(dSortedPos);
  freeArray(tempPos1);
  freeArray(tempPos2);
  freeArray(dSortedVel);
  freeArray(dSortedRad);
  freeArray(dphase);
  freeArray(dfreq);

  freeArray(dGridParticleHash);
  freeArray(dGridParticleIndex);
  freeArray(dCellStart);
  freeArray(dCellEnd);

  unregisterGLBufferObject(cuda_colorvbo_resource);
  unregisterGLBufferObject(cuda_posvbo_resource);
  glDeleteBuffers(1, (const GLuint *)&posVbo);
  glDeleteBuffers(1, (const GLuint *)&colorVBO);
  unregisterGLBufferObject(cuda_radvbo_resource);
  glDeleteBuffers(1, (const GLuint *)&radVbo);

}
