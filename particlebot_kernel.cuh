


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

#ifndef PARTICLEBOT_KERNEL_H
#define PARTICLEBOT_KERNEL_H

#define USE_TEX 0

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"

typedef unsigned int uint;

enum ParticlebotConfig
  {
    CONFIG_RANDOM,
    CONFIG_GRID,
    CONFIG_BLOB,
    CONFIG_BLOB_UPLEFT,
    CONFIG_HEX,
    CONFIG_LINE,
    CONFIG_LIGHTTEST_7,
    _NUM_CONFIGS
  };

enum ParticlebotArray
  {
    POSITION,
    VELOCITY,
    RADII,
    PHASE,
    FREQUENCY,
    DEAD
  };

enum ParticlebotControl
  {
    LIGHT_WAVE
  };

// simulation parameters
struct SimParams
{
  uint2 gridSize;
  uint numCells;

  float2 worldOrigin;
  float2 cellSize;

  uint nCells;
  int nDead;
  uint maxParticlebotsPerCell;

  float gravity;
  float spring;
  float damping;
  float shear;
  float attraction;
  float boundaryDamping;
  float friction;

  float massFactor;
  float frictionFactor;
  float radFactor;
  float attractionFactor;

  float constraint;
  float constraint_contraction;
  int centroid_steps;
  float centroid_int;
  float centroid_radius;
  float light_x;
  float light_y;
  float phase_update_interval;
  ParticlebotControl control;
  ParticlebotConfig config;
  float min_radius;
  float max_radius;
  float rise_period;
  float freq;

  int nobstacles;
  float *x1obs;
  float *x2obs;
  float *y1obs;
  float *y2obs;

  int n_cir_obstacles;
  float *x_cir_obs;
  float *y_cir_obs;
  float *r_cir_obs;

  int Nx;

  float phase_std;
  unsigned seed;

  uint light_shadow;
  uint testing;
  uint constrained_contraction;
  uint display_shadow;
  float time_to_dead;
  float max_time;
};

#endif
