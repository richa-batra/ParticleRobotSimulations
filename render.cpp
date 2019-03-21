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


#include <math.h>
#include <assert.h>
#include <stdio.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>


#include "render.h"
#include "shaders.h"

#ifndef PI
#define PI    3.1415926535897932384626433832795
#endif

ParticlebotRenderer::ParticlebotRenderer()
  : pos(0),
    nCells(0),
    pointSize(1.0f),
    program(0),
    vbo(0),
    colorVBO(0),
    centroid_steps(500)
{
  _initGL();
}

ParticlebotRenderer::~ParticlebotRenderer()
{
  pos = 0;
}

void ParticlebotRenderer::setPositions(float *p, int n)
{
  pos = p;
  nCells = n;
}

void ParticlebotRenderer::setVertexBuffer(unsigned int v, int n)
{
  vbo = v;
  nCells = n;
}

void ParticlebotRenderer::_drawPoints()
{

  glEnableVertexAttribArray(glGetAttribLocation(program, "pos"));
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(
                        glGetAttribLocation(program, "pos"), // attribute
                        2,                 // number of elements per vertex
                        GL_FLOAT,          // the type of each element
                        GL_FALSE,          // take our values as-is (no normalization)
                        0,                 // no extra data between each position
                        0                  // offset of first element
                        );

  glEnableVertexAttribArray(glGetAttribLocation(program, "rad"));
  glBindBuffer(GL_ARRAY_BUFFER, radVbo);
  glVertexAttribPointer(
                        glGetAttribLocation(program, "rad"), // attribute
                        1,                 // number of elements per vertex
                        GL_FLOAT,          // the type of each element
                        GL_FALSE,          // take our values as-is (no normalization)
                        0,                 // no extra data between each position
                        0                  // offset of first element
                        );

  if (colorVBO)
    {
      glEnableVertexAttribArray(glGetAttribLocation(program, "color"));
      glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
      glVertexAttribPointer(
                            glGetAttribLocation(program, "color"), // attribute
                            4,                 // number of elements per vertex
                            GL_FLOAT,          // the type of each element
                            GL_FALSE,          // take our values as-is (no normalization)
                            0,                 // no extra data between each position
                            0                  // offset of first element
                            );
    }

  glDrawArrays(GL_POINTS, 0, nCells+centroid_steps);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDisableVertexAttribArray(glGetAttribLocation(program, "pos"));
  glDisableVertexAttribArray(glGetAttribLocation(program, "rad"));
  glDisableVertexAttribArray(glGetAttribLocation(program, "color"));
}

void ParticlebotRenderer::display()
{
  glEnable(GL_POINT_SPRITE_ARB);
  glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glDepthMask(GL_TRUE);
  glEnable(GL_DEPTH_TEST);

  glUseProgram(program);
  glUniform1f(glGetUniformLocation(program, "pointScale"), window_h / tanf(fov*0.5f*(float)PI/180.0f));
  glUniform3f(glGetUniformLocation(program, "lightDir"), light_x, light_y, light_z);

  glColor3f(1, 1, 1);
  _drawPoints();

  glUseProgram(0);
  glDisable(GL_POINT_SPRITE_ARB);

}

GLuint
ParticlebotRenderer::_compileProgram(const char *vsource, const char *fsource)
{
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

  glShaderSource(vertexShader, 1, &vsource, 0);
  glShaderSource(fragmentShader, 1, &fsource, 0);

  glCompileShader(vertexShader);
  glCompileShader(fragmentShader);

  GLuint glprogram = glCreateProgram();

  glAttachShader(glprogram, vertexShader);
  glAttachShader(glprogram, fragmentShader);

  glLinkProgram(glprogram);

  // check if program linked
  GLint success = 0;
  glGetProgramiv(glprogram, GL_LINK_STATUS, &success);

  if (!success)
    {
      char temp[256];
      glGetProgramInfoLog(glprogram, 256, 0, temp);
      printf("Failed to link program:\n%s\n", temp);
      glDeleteProgram(glprogram);
      glprogram = 0;
    }

  return glprogram;
}

void ParticlebotRenderer::_initGL()
{
  program = _compileProgram(vertexShader, spherePixelShader);

#if !defined(__APPLE__) && !defined(MACOSX)
  glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
  glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}
