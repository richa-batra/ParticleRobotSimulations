#ifndef __POSTPROCESS_H__
#define __POSTPROCESS_H__

void PostprocessCUDA( cudaGraphicsResource_t& dst, cudaGraphicsResource_t& src,
                      unsigned int width, unsigned int height, unsigned int interval, const char* outputFilename );

void PostprocessFinish();

#endif
