/*
 * Code for Particle Robotics
 * Author: Richa Batra (richa.batra@columbia.edu)

 ======================================
 Adapted from NVIDIA CUDA Sample Code.

 * This software contains source code
 * provided by NVIDIA Corporation
 --------------------------------------

 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#include <GL/freeglut.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> 
#include <helper_functions.h>

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "particlebot.h"
#include "render.h"
#include "particlebot_kernel.cuh"

#include <iostream>
#include <string>

#include "postprocess.h"
#include "opencv2/opencv.hpp"

#include <time.h>


#define DISPLAY_INTERVAL 600     // No of timesteps after which to update display
#define VIDEO_INTERVAL 1        // No of display steps after which to write a frame to video. 
// For example, if timestep = 0.01 seconds, DISPLAY_INTERVAL = 600 and VIDEO_INTERVAL=10,
// one frame of the output video will correspond to 0.01*600*10 = 60 seconds

const char* folder_name = "./";  // Folder (must be created beforehand) where the output csv and video files will be written.

char* csv_filename = (char *)malloc(sizeof(char) * 300);
char* outputFilename = (char *)malloc(sizeof(char) * 300);
const uint width = 1920 , height = 1080;

FILE *fp;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, 0 };
float camera_rot[] = { 0, 0, 0 };
float camera_trans_lag[] = { 0, 0, 0 };
float camera_rot_lag[] = { 0, 0, 0 };
int mode = 0;
bool displayEnabled = true;
enum { M_VIEW = 0, M_MOVE };
float camera_y = 10;
float camera_x = 0;
float light_radius = 1.75f;
int numIterations = 0; // run until exit
int iterations = 1;
float timestep;
float sort_interval;
float dump_interval;
SimParams params;

Particlebot *particlebotSystem = 0;

ParticlebotRenderer *renderer = 0;

float modelView[16];
unsigned int frameCount = 0;

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

#define SRC_BUFFER  0
#define DST_BUFFER  1

int g_iGLUTWindowHandle = 0;
int g_iWindowPositionX = 0;
int g_iWindowPositionY = 0;
int g_iWindowWidth = width;
int g_iWindowHeight = height;

int g_iImageWidth = g_iWindowWidth;
int g_iImageHeight = g_iWindowHeight;

float g_fRotate[3] = { 0.0f, 0.0f, 0.0f };  // Rotation parameter for scene object.

GLuint g_GLFramebuffer = 0;                  // Frame buffer object for off-screen rendering.
GLuint g_GLColorAttachment0 = 0;            // Color texture to attach to frame buffer object.
GLuint g_GLDepthAttachment = 0;             // Depth buffer to attach to frame buffer object.
GLuint g_GLPostprocessTexture = 0;          // This is where the result of the post-process effect will go.
                                            // This is also the final texture that will be blit to the back buffer for viewing.

// The CUDA Graphics Resource is used to map the OpenGL texture to a CUDA
// buffer that can be used in a CUDA kernel.
// We need 2 resource: One will be used to map to the color attachment of the
//   framebuffer and used read-only from the CUDA kernel (SRC_BUFFER),
//   the second is used to write the postprocess effect to (DST_BUFFER).
cudaGraphicsResource_t g_CUDAGraphicsResource[2] = { 0, 0 };

// Create a framebuffer object that is used for offscreen rendering.
void CreateFramebuffer( GLuint& framebuffer, GLuint colorAttachment0, GLuint depthAttachment );
void DeleteFramebuffer( GLuint& framebuffer );

void CreatePBO( GLuint& bufferID, size_t size );
void DeletePBO( GLuint& bufferID );

void CreateTexture( GLuint& texture, unsigned int width, unsigned int height );
void DeleteTexture( GLuint& texture );

void CreateDepthBuffer( GLuint& depthBuffer, unsigned int width, unsigned int height );
void DeleteDepthBuffer( GLuint& depthBuffer );

// Links a OpenGL texture object to a CUDA resource that can be used in the CUDA kernel.
void CreateCUDAResource( cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags );
void DeleteCUDAResource( cudaGraphicsResource_t& cudaResource );

// Create a pixel buffer object
void CreatePBO( GLuint& bufferID, size_t size )
{
// Make sure the buffer doesn't already exist
DeletePBO( bufferID );

 glGenBuffers( 1, &bufferID );
 glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID );
 glBufferData( GL_PIXEL_UNPACK_BUFFER, size, NULL, GL_STREAM_DRAW );

 glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
}

void DeletePBO(  GLuint& bufferID )
{
  if ( bufferID != 0 )
    {
      glDeleteBuffers( 1, &bufferID );
      bufferID = 0;
    }
}

// Create a texture resource for rendering to.
void CreateTexture( GLuint& texture, unsigned int width, unsigned int height )
{
  // Make sure we don't already have a texture defined here
  DeleteTexture( texture );

  glGenTextures( 1, &texture );
  glBindTexture( GL_TEXTURE_2D, texture );

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Create texture data (4-component unsigned byte)
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );

  // Unbind the texture
  glBindTexture( GL_TEXTURE_2D, 0 );
}

void DeleteTexture( GLuint& texture )
{
  if ( texture != 0 )
    {
      glDeleteTextures(1, &texture );
      texture = 0;
    }
}

void CreateDepthBuffer( GLuint& depthBuffer, unsigned int width, unsigned int height )
{
  // Delete the existing depth buffer if there is one.
  DeleteDepthBuffer( depthBuffer );

  glGenRenderbuffers( 1, &depthBuffer );
  glBindRenderbuffer( GL_RENDERBUFFER, depthBuffer );

  glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );

  // Unbind the depth buffer
  glBindRenderbuffer( GL_RENDERBUFFER, 0 );
}

void DeleteDepthBuffer( GLuint& depthBuffer )
{
  if ( depthBuffer != 0 )
    {
      glDeleteRenderbuffers( 1, &depthBuffer );
      depthBuffer = 0;
    }
}

void CreateFramebuffer( GLuint& framebuffer, GLuint colorAttachment0, GLuint depthAttachment )
{
  // Delete the existing framebuffer if it exists.
  DeleteFramebuffer( framebuffer );

  glGenFramebuffers( 1, &framebuffer );
  glBindFramebuffer( GL_FRAMEBUFFER, framebuffer );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorAttachment0, 0 );
  glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthAttachment );

  // Check to see if the frame buffer is valid
  GLenum fboStatus = glCheckFramebufferStatus( GL_FRAMEBUFFER );
  if ( fboStatus != GL_FRAMEBUFFER_COMPLETE )
    {
      std::cerr << "ERROR: Incomplete framebuffer status." << std::endl;
    }

  // Unbind the frame buffer
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

void DeleteFramebuffer( GLuint& framebuffer )
{
  if ( framebuffer != 0 )
    {
      glDeleteFramebuffers( 1, &framebuffer );
      framebuffer = 0;
    }
}

void Postprocess()
{
  PostprocessCUDA( g_CUDAGraphicsResource[DST_BUFFER], g_CUDAGraphicsResource[SRC_BUFFER], g_iImageWidth, g_iImageHeight, VIDEO_INTERVAL, outputFilename );
}

void DisplayImage( GLuint texture, unsigned int x, unsigned int y, unsigned int width, unsigned int height )
{
  glBindTexture(GL_TEXTURE_2D, texture);
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glMatrixMode( GL_MODELVIEW);
  glLoadIdentity();

  glPushAttrib( GL_VIEWPORT_BIT );
  glViewport(x, y, width, height );

  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
  glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
  glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
  glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
  glEnd();

  glPopAttrib();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glDisable(GL_TEXTURE_2D);
}

void CreateCUDAResource( cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags )
{
  // Map the GL texture resource with the CUDA resource
  cudaGraphicsGLRegisterImage( &cudaResource, GLtexture, GL_TEXTURE_2D, mapFlags );
}

void DeleteCUDAResource( cudaGraphicsResource_t& cudaResource )
{
  if ( cudaResource != 0 )
    {
      cudaGraphicsUnregisterResource( cudaResource );
      cudaResource = 0;
    }
}

// initialize particle system
void initParticlebotSystem()
{
  particlebotSystem = new Particlebot(params);
  particlebotSystem->reset();

  renderer = new ParticlebotRenderer();
  renderer->setCentroidSteps(params.centroid_steps);
  renderer->setColorBuffer(particlebotSystem->getColorBuffer());
  renderer->setLightPos(-params.light_x, 0.1f, params.light_y);

}


// initialize OpenGL
void initGL(int *argc, char **argv)
{
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(width, height);
  glutCreateWindow("Particlebot Simulation");

  //Init GLEW
  glewInit();
  GLboolean gGLEW = glewIsSupported(
                                    "GL_VERSION_3_1 "
                                    "GL_ARB_pixel_buffer_object "
                                    "GL_ARB_framebuffer_object "
                                    "GL_ARB_copy_buffer "
                                    );
  if ( !gGLEW ) return;

#if defined (WIN32)

  if (wglewIsSupported("WGL_EXT_swap_control"))
    {
      // disable vertical sync
      wglSwapIntervalEXT(0);
    }
#endif
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.25, 0.25, 0.25, 1.0);

  glutReportErrors();

  cudaGLSetGLDevice(0);
}


void display()
{
  // Bind the framebuffer that we want to use as the render target.
  glBindFramebuffer( GL_FRAMEBUFFER, g_GLFramebuffer );


  particlebotSystem->dumpParticlebot(0, params.nCells, fp, dump_interval, params.testing, params.light_x,params.light_y);
  particlebotSystem->update(timestep, sort_interval);

  if (renderer)
    {
      renderer->setVertexBuffer(particlebotSystem->getCurrentReadBuffer(), params.nCells);
      renderer->setRadBuffer(particlebotSystem->getRadBuffer());
    }

  // render
  if (frameCount%DISPLAY_INTERVAL == 0)
    {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // view transform
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      gluLookAt(camera_x, camera_y, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 1.0);

      glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
      glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
      glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

      glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

      // cube
      glPushMatrix();
      glColor3f(1.0f, 1.0f, 1.0f);
      glBegin(GL_POLYGON);//begin drawing of polygon
      glVertex3f(-64.0f, 0.0f, -64.0f);//first vertex
      glVertex3f(-64.0f, 0.0f, 64.0f);//second vertex
      glVertex3f(64.0f, 0.0f, 64.0f);//third vertex
      glVertex3f(64.0f, 0.0f, -64.0f);//fourth vertex
      glEnd();//end drawing of polygon
      glColor3f(1.0, 0.0, 0.0);
      glLineWidth(0.5);
      glPopMatrix();


      glPushMatrix();
      glTranslatef(-params.light_x, 0.01f, params.light_y);
      glColor3f(0.8f, 0.8f, 0.0);
      glutSolidSphere(light_radius, 20, 10);
      glPopMatrix();

      // Cylinder Obstacles
      for(int i=0; i<params.n_cir_obstacles; i++){
        glPushMatrix();
        glColor3f(0.2, 0.2, 0.2);
        GLUquadricObj *quadratic;
        quadratic = gluNewQuadric();
        glTranslatef(*(params.x_cir_obs+i)*-1.0f, 0.01f, *(params.y_cir_obs+i));
        glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        gluDisk(quadratic,0,*(params.r_cir_obs+i),32,32);
        glPopMatrix();
      }

      // Box OBstacles
      glPushMatrix();
      glColor3f(0.2, 0.2, 0.2);
      glBegin(GL_QUADS);
      for(int i=0; i<params.nobstacles; i++){
        // Top
        glVertex3f(*(params.x1obs+i)*-1.0f, 0.02f, *(params.y1obs+i));//first vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, 0.02f, *(params.y1obs+i));//second vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, 0.02f, *(params.y2obs+i));//third vertex
        glVertex3f(*(params.x1obs+i)*-1.0f, 0.02f, *(params.y2obs+i));//fourth vertex

        // Bottom
        glVertex3f(*(params.x1obs+i)*-1.0f, -0.02f, *(params.y1obs+i));//first vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, -0.02f, *(params.y1obs+i));//second vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, -0.02f, *(params.y2obs+i));//third vertex
        glVertex3f(*(params.x1obs+i)*-1.0f, -0.02f, *(params.y2obs+i));//fourth vertex

        // Left
        glVertex3f(*(params.x1obs+i)*-1.0f, 0.02f, *(params.y1obs+i));//first vertex
        glVertex3f(*(params.x1obs+i)*-1.0f, 0.02f, *(params.y2obs+i));//second vertex
        glVertex3f(*(params.x1obs+i)*-1.0f, -0.02f, *(params.y2obs+i));//third vertex
        glVertex3f(*(params.x1obs+i)*-1.0f, -0.02f, *(params.y1obs+i));//fourth vertex

        // Right
        glVertex3f(*(params.x2obs+i)*-1.0f, 0.02f, *(params.y1obs+i));//first vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, 0.02f, *(params.y2obs+i));//second vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, -0.02f, *(params.y2obs+i));//third vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, -0.02f, *(params.y1obs+i));//fourth vertex

        // Front
        glVertex3f(*(params.x1obs+i)*-1.0f, 0.02f, *(params.y1obs+i));//first vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, 0.02f, *(params.y1obs+i));//second vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, -0.02f, *(params.y1obs+i));//third vertex
        glVertex3f(*(params.x1obs+i)*-1.0f, -0.02f, *(params.y1obs+i));//fourth vertex

        // Back
        glVertex3f(*(params.x1obs+i)*-1.0f, 0.02f, *(params.y2obs+i));//first vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, 0.02f, *(params.y2obs+i));//second vertex
        glVertex3f(*(params.x2obs+i)*-1.0f, -0.02f, *(params.y2obs+i));//third vertex
        glVertex3f(*(params.x1obs+i)*-1.0f, -0.02f, *(params.y2obs+i));//fourth vertex
      }
      glEnd();
      glPopMatrix();


      renderer->display();

      // Unbind the framebuffer so we render to the back buffer again.
      glBindFramebuffer(GL_FRAMEBUFFER, 0);

      Postprocess();

      // Blit the image full-screen
      DisplayImage(g_GLPostprocessTexture, 0, 0, g_iWindowWidth, g_iWindowHeight);

      glutSwapBuffers();

      glutReportErrors();

    }
  glutPostRedisplay();
  frameCount++;
}

inline float frand()
{
  return rand() / (float)RAND_MAX;
}


void reshape(int w, int h)
{
  h = std::max(h, 1);

  g_iWindowWidth = w;
  g_iWindowHeight = h;

  g_iImageWidth = w;
  g_iImageHeight = h;

  // Create a surface texture to render the scene to.
  CreateTexture( g_GLColorAttachment0, g_iImageWidth, g_iImageHeight );
  // Create a depth buffer for the frame buffer object.
  CreateDepthBuffer( g_GLDepthAttachment, g_iImageWidth, g_iImageHeight );
  // Attach the color and depth textures to the framebuffer.
  CreateFramebuffer( g_GLFramebuffer, g_GLColorAttachment0, g_GLDepthAttachment );

  // Create a texture to render the post-process effect to.
  CreateTexture( g_GLPostprocessTexture, g_iImageWidth, g_iImageHeight );

  // Map the color attachment to a CUDA graphics resource so we can read it in a CUDA a kernel.
  CreateCUDAResource( g_CUDAGraphicsResource[SRC_BUFFER], g_GLColorAttachment0, cudaGraphicsMapFlagsReadOnly );
  // Map the post-process texture to the CUDA resource so it can be
  // written in the kernel.
  CreateCUDAResource( g_CUDAGraphicsResource[DST_BUFFER], g_GLPostprocessTexture, cudaGraphicsMapFlagsWriteDiscard );

  glutPostRedisplay();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float)w / (float)h, 0.1, 100.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);

  if (renderer)
    {
      renderer->setWindowSize(w, h);
      renderer->setFOV(60.0);
    }
}

void mouse(int button, int state, int x, int y)
{
  int mods;

  if (state == GLUT_DOWN)
    {
      buttonState |= 1 << button;
    }
  else if (state == GLUT_UP)
    {
      buttonState = 0;
    }

  mods = glutGetModifiers();

  if (mods & GLUT_ACTIVE_SHIFT)
    {
      buttonState = 2;
    }
  else if (mods & GLUT_ACTIVE_CTRL)
    {
      buttonState = 3;
    }

  ox = x;
  oy = y;

  glutPostRedisplay();
}

// transform vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + m[12];
  r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + m[13];
  r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
  r[1] = v[0] * m[4] + v[1] * m[5] + v[2] * m[6];
  r[2] = v[0] * m[8] + v[1] * m[9] + v[2] * m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
  float x[4];
  x[0] = v[0] - m[12];
  x[1] = v[1] - m[13];
  x[2] = v[2] - m[14];
  x[3] = 1.0f;
  ixform(x, r, m);
}


void cleanup()
{
  PostprocessFinish();
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  //srand((unsigned)time(NULL)); // set it to specific int for reproducible results

  std::atexit(cleanup);

  // quadrilateral obstacle
  params.nobstacles = 0;    //max 10 for now

  params.x1obs = (float *)malloc(10*sizeof(float));
  params.x2obs = (float *)malloc(10*sizeof(float));
  params.y1obs = (float *)malloc(10*sizeof(float));
  params.y2obs = (float *)malloc(10*sizeof(float));

  // circular obstacle
  params.n_cir_obstacles = 0;    // max 10 for now

  params.x_cir_obs = (float *)malloc(10 * sizeof(float));
  params.y_cir_obs = (float *)malloc(10 * sizeof(float));
  params.r_cir_obs = (float *)malloc(10 * sizeof(float));


  params.min_radius = 0.0775;
  params.max_radius = 0.1175;

  params.centroid_int = 10;
  params.centroid_radius = 0.05f;
  params.centroid_steps = 24000;

  sort_interval = 180.0f;
  dump_interval = 60.0f;
  params.testing = 0; //0 outputs centroid and 1 outputs every cell position and velocity

  params.friction = 0.4;
  params.spring = 1000.0f;
  params.damping = 10.0f;
  params.shear = 40.0f;
  params.constraint = 0.5f;
  params.constrained_contraction = 0;
  params.constraint_contraction = 10.0f;
  params.attraction = 3.0f*0.000015884f;
  params.boundaryDamping = -1.0f;
  params.gravity = 9.81*0.566f;

  camera_y = 35;
  light_radius = 0.25f;
  timestep = 0.01f;
  params.nCells = 11;
  params.nDead = -1; // -1 for carrying mass

  // Multiplicative Factors for radius, mass and friction of object to be carried
  // Only Used in nDead == -1
  params.radFactor = 0.2;
  params.massFactor = 1.0;
  params.frictionFactor = 1.0;
  params.attractionFactor = 0.0f;
  params.time_to_dead = 0;
  params.max_time = 720.0;//9 *60*60; //in seconds
  params.seed = (unsigned)time(NULL);

  int cont = 0; //1 is continue existing experiment based on output name

  float gap_pct = 0;
  float max_pack = 16.449;
  float light_fact = 3.0;

  //mass study
  /*if (argc > 1) {
    params.seed = (unsigned)strtol(argv[1], NULL, 10);
    if (argc > 2) {
    params.nCells = strtol(argv[2], NULL, 10);
    if (argc > 3) {
    params.radFactor = strtof(argv[3], NULL);
    if (argc > 4) {
    params.frictionFactor = strtof(argv[4], NULL);
    }
    }
    }
    }*/


  //gap study
  if (argc > 1) {
    params.seed = (unsigned)strtol(argv[1], NULL, 10);
    if (argc > 2) {
      params.nCells = strtol(argv[2], NULL, 10);
      if (argc > 3) {
        light_fact = strtof(argv[3], NULL);
        if (argc > 4) {
          gap_pct = strtof(argv[4], NULL);
        }
      }
    }
  }
  

  
  srand(params.seed);
  //set packing diameter in meters
  if (params.nCells <= 11)
    {
      max_pack = 0.591;
      light_fact = 16;
    }
  else if (params.nCells <= 101)
    {
      max_pack = 1.718;
      light_fact = 8;
    }
  else if (params.nCells <= 1001)
    {
      max_pack = 5.264;
      light_fact = 4;
    }
  else if (params.nCells <= 10001)
    {
      max_pack = 16.449;
      light_fact = 2;
    }
  if (gap_pct > 0)
    {
      params.nobstacles = 2;

      *(params.x1obs) = -max_pack*0.75 - params.min_radius * 2;
      *(params.x2obs) = -max_pack*0.75;
      *(params.y1obs) = max_pack*gap_pct / 200;
      *(params.y2obs) = 500.0f;

      *(params.x1obs + 1) = -max_pack*0.75 - params.min_radius * 2;
      *(params.x2obs + 1) = -max_pack*0.75;
      *(params.y1obs + 1) = -500.0f;
      *(params.y2obs + 1) = -max_pack*gap_pct / 200;

      params.light_x = -max_pack * light_fact;
    }

  if (params.radFactor < 1)
    {
      params.radFactor *= max_pack / (2 * params.min_radius);
    }

  params.light_x = -max_pack*3.5;
  params.light_y = 0;
  camera_y = max_pack * light_fact;
  camera_x = max_pack * 0;

  // 0 -> Obstacle does not affect light transmission
  // 1 -> Light is blocked by obstacles, cells in shadow
  //      modulate last
  // 2 -> Light is blocked by obstacles, cells in shadow
  //      do not modulate
  params.light_shadow = 1;


  char *temp_filename = (char *)malloc(sizeof(char) * 300);
  sprintf(temp_filename, "mass2_N_%d_A_%g_M_%g_R_%g_F_%g_seed_%u",
          params.nCells, params.attractionFactor, params.massFactor,
          params.radFactor==1?params.radFactor:params.radFactor*2*params.min_radius/max_pack,
          params.frictionFactor, params.seed);
  if (params.attractionFactor)
    sprintf(temp_filename, "gap_N_%d_pct_%f_seed_%u_display_int_600_int_10",
            params.nCells, gap_pct, params.seed);
  if (params.light_shadow == 0) {
    sprintf(csv_filename, "%s%s_no_shadow.csv", folder_name, temp_filename);
    if(cont)
      sprintf(outputFilename, "%s%s_no_shadow_%d.avi", folder_name, temp_filename, cont);
    else
      sprintf(outputFilename, "%s%s_no_shadow.avi", folder_name, temp_filename);
  }
  if (params.light_shadow == 1) {
    sprintf(csv_filename, "%s%s_modulate_last.csv", folder_name, temp_filename);
    if(cont)
      sprintf(outputFilename, "%s%s_modulate_last_%d.avi", folder_name, temp_filename, cont);
    else
      sprintf(outputFilename, "%s%s_modulate_last.avi", folder_name, temp_filename);
  }
  if (params.light_shadow == 2) {
    sprintf(csv_filename, "%s%s_no_modulate.csv", folder_name, temp_filename);
    if(cont)
      sprintf(outputFilename, "%s%s_no_modulate_%d.avi", folder_name, temp_filename, cont);
    else
      sprintf(outputFilename, "%s%s_no_modulate.avi", folder_name, temp_filename);
  }

  params.rise_period = 2;
  params.phase_std = 0.3f*params.rise_period;
  params.config = CONFIG_RANDOM;


  params.display_shadow = 0;
  params.phase_update_interval = 12;
  params.control = LIGHT_WAVE;
  params.Nx = 5;

  params.freq = 0.5f / 25;//
  
  if (params.nDead == -1 && params.max_radius * 0.5 * params.radFactor > 2 * params.max_radius) {
    params.cellSize.x = params.cellSize.y = params.max_radius * 0.5 * params.radFactor + 4 * params.max_radius;
  }
  else {
    params.cellSize.x = params.cellSize.y = params.max_radius * 2;
  }
  params.gridSize.x = params.gridSize.y = 512;      //

  params.numCells = params.gridSize.x*params.gridSize.y;
  
  params.worldOrigin = make_float2(-64.0f, -64.0f);
  if (cont) {
    fp = fopen(csv_filename, "r");
  }
  else {
    fp = fopen(csv_filename, "w+");
  }
  frameCount = 0;

  initGL(&argc, argv);
  cudaGLInit(argc, argv);


  initParticlebotSystem();

  if (cont) {
    particlebotSystem->loadFromFile(0, params.nCells, fp, dump_interval);
    fp = fopen(csv_filename, "a");
  }
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);

  glutCloseFunc(cleanup);

  glutMainLoop();


}
