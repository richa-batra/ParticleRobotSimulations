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

#ifndef __RENDER__
#define __RENDER__

class ParticlebotRenderer
{
 public:
  ParticlebotRenderer();
  ~ParticlebotRenderer();

  void setPositions(float *p, int n);
  void setVertexBuffer(unsigned int v, int n);
  void setColorBuffer(unsigned int v)
  {
    colorVBO = v;
  }
  void setRadBuffer(unsigned int v)
  {
    radVbo = v;
  }

  void setCentroidSteps(int steps)
  {
    centroid_steps = steps;
  }
  void setFOV(float x)
  {
    fov = x;
  }
  void setWindowSize(int w, int h)
  {
    window_w = w;
    window_h = h;
  }

  void setLightPos(float x, float y, float z){
    light_x = x;
    light_y = y;
    light_z = z;
  }

  void display();

 protected: // methods
  void _initGL();
  void _drawPoints();
  GLuint _compileProgram(const char *vsource, const char *fsource);

 protected: // data
  float *pos;
  int nCells;
  int centroid_steps;

  float pointSize;
  float fov;
  int window_w, window_h;

  float light_x;
  float light_y;
  float light_z;

  GLuint program;

  GLuint vbo;
  GLuint colorVBO;
  GLuint radVbo;
};

#endif //__ RENDER_PARTICLES__
