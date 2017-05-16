#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//////////////////////////////////////////////////////////////////////////////////////////
// constants 
//////////////////////////////////////////////////////////////////////////////////////////
__device__ const int SCREEN_WIDTH	= 500;
__device__ const int SCREEN_HEIGHT = 500;

const int FPS_MAX = 60;//anything bigger than 60 fps is pointless on most machines as the monitor won't take more

static int BOID_MAX = 0;

__device__ const float BOID_SPEED_MAX = 2.0f;
__device__ const float BOID_RESPECT_DIST = 15.0f; //boid to boid spacing
__device__ const float BOID_ROTATE_MAX = 0.00000000001f;
__device__ const float BOID_MOUSE_FEAR = 25.0f; //boid distance from mouse
__device__ const float BOID_COHESION_WEIGHTING = 0.0025f;
__device__ const float BIOD_SIGHT_RANGE = 25.0f; //max distance to consider boids as nearby

const bool DEBUG_RUN_TOP_SPEED = true; //override FPS cap
const bool DEBUG_AVERAGE_POS = false;
const bool DEBUG_HIGHLIGHT_FIRST_BOID = false;
const bool DEBUG_PRINT_POS_OF_FIRST_BOID = false;
const bool DEBUG_PRINT_POS_OF_ALL_BOIDS = false;
const float DEBUG_UPDATE_FREQUENCY = 1.0f/FPS_MAX; //override FPS
