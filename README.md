# ParticleRobotSimulations

CUDA-based simulation code for

[**Particle robotics based on statistical mechanics of loosely coupled components**](https://doi.org/10.1038/s41586-019-1022-9)  
*Shuguang Li, Richa Batra, David Brown, Hyun-Dong Chang, Nikhil Ranganathan, Chuck Hoberman, Daniela Rus & Hod Lipson*


## Installation Steps

#### Ubuntu
* Install build-essential package
```
sudo apt-get install build-essentials
```
* Install [NVIDIA CUDA ToolKit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* Install GL/GLEW related libraries
```
    sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libglew-dev
```
* Install OpenCV
```
sudo apt-get install libopencv-dev
```
* Compile the Code
```
make
```
---
#### Windows

* Install Visual Studio (with C/C++ development module)
* Install [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
* Install [OpenCV](https://github.com/opencv/opencv)
* Install [FreeGLUT](http://freeglut.sourceforge.net/)
* Install [GLEW](http://glew.sourceforge.net/)
* Set the following Environment Variables (System Properties->Advanced->Environment Variables) to point to your installation locations for the above libraries
    ```
    CUDA_PATH
    OpenCV_DIR
    GLEW_DIR
    FREEGLUT_DIR
    ```
* Add 
    ```
     %FREEGLUT_DIR%\bin\x64;%GLEW_DIR%\bin\Release\x64;
    ```
    to your User PATH Variable (System Properties->Advanced->Environment Variables)
* Open ParticleBot.sln with Visual Studio and Build Solution

## Author

[Richa Batra](mailto:richa.batra@columbia.edu)
