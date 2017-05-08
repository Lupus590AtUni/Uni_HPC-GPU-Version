// GraphicsTemplate.cpp
// 
//////////////////////////////////////////////////////////////////////////////////////////
// includes 
//////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
using std::cout;
using std::cerr;
#include <tchar.h>
#include <windows.h>
#include "GL/glut.h"
#include "globals.cuh"
#include "cRenderClass.cuh"
#include <vector>
using std::vector;
//
#include "NA_Boid.cuh"
#include "NA_MathsLib.cuh"
#include "NA_Timer.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

//////////////////////////////////////////////////////////////////////////////////////////
// externals 
//////////////////////////////////////////////////////////////////////////////////////////
extern cRenderClass graphics;
extern NA_MathsLib na_maths;

vector<NA_Boid> boidList; //not really a list

//////////////////////////////////////////////////////////////////////////////////////////
// renderScene() - render the scene
//////////////////////////////////////////////////////////////////////////////////////////
void renderScene()
{
	for (int i = 0; i < BOID_MAX; i++)
	{
		boidList[i].draw();
	}

	if (DEBUG_AVERAGE_POS)
	{
		NA_Vector sumPosition;
		for (int i = 0; i < BOID_MAX; i++)
		{
			sumPosition.x += boidList[i].position.x;
			sumPosition.y += boidList[i].position.y;
		}
		//convert to average
		sumPosition.x = sumPosition.x / (BOID_MAX);
		sumPosition.y = sumPosition.y / (BOID_MAX);

		graphics.setPointSize(10);
		graphics.setColour(0.0f, 0.0f, 1.0f);
		graphics.drawPixel(sumPosition.x, sumPosition.y);
	}

	// render the scene
	graphics.render();
}

//////////////////////////////////////////////////////////////////////////////////////////
// update() - update function
//////////////////////////////////////////////////////////////////////////////////////////
void update()
{
	static bool first = true;

	if (first)
		renderScene();
	first = false;

	//renderScene();
	//cout << "first render done\n";
	// add any update code here...
	static NA_Timer fpsCap;//wait if FPS is too high (if boids move too fast)
	fpsCap.restart();
	if (DEBUG_PRINT_POS_OF_ALL_BOIDS)
		fpsCap.setDuration(DEBUG_UPDATE_FREQUENCY);
	else
		fpsCap.setDuration(1.0f / FPS_MAX);


	if (first && !DEBUG_RUN_TOP_SPEED)
		fpsCap.wait();

	for (int i = 0; i < BOID_MAX; i++)
	{
		boidList[i].update();
	}

	if (DEBUG_PRINT_POS_OF_ALL_BOIDS || DEBUG_PRINT_POS_OF_FIRST_BOID) system("cls");
	for (int i = 0; i < BOID_MAX; i++)
	{
		boidList[i].postUpdate();
		if (DEBUG_PRINT_POS_OF_ALL_BOIDS || DEBUG_PRINT_POS_OF_FIRST_BOID && i == 0) cout << "pos: " << boidList[i].position.x << " " << boidList[i].position.y << "\n";
	}



	//cout << "updates done, waiting\n";

	extern void debugMouse();
	//debugMouse();
	//cout << "mouse scary? " << graphics.mouseIsScary << "\n";

	if (!DEBUG_RUN_TOP_SPEED) fpsCap.wait();

	// always re-render the scene..
	renderScene();
	//cout << " post render done\n";
}



////////////////////////////////////////////////////////////////////////////////////////
// _tmain() - program entry point
////////////////////////////////////////////////////////////////////////////////////////
//int _tmain(int argc, _TCHAR* argv[])
//{	
//	// init glut stuff..
//	graphics.create(argc, argv);
//
//	// good place for one-off initialisations and objects creation..
//
//	//make all boids
//	na_maths.seedDice();
//	for (int i = 0; i < BOID_MAX; i++)
//	{
//		NA_Boid temp;
//		temp.position.x = na_maths.dice(SCREEN_WIDTH);
//		temp.position.y = na_maths.dice(SCREEN_HEIGHT);
//
//		//temp.position.x = 100.0f;
//		//temp.position.y = 100.0f;
//
//		temp.currentVelocity.x = float(na_maths.dice(-100,100))/100.0f;
//		temp.currentVelocity.y = float(na_maths.dice(-100, 100))/100.0f;
//
//		boidList.push_back(temp);
//
//		//cout << "POS: X: " << temp.position.x << " Y: " << temp.position.y << "\n";
//		//cout << "VEL: X: " << temp.currentVelocity.x << " Y: " << temp.currentVelocity.y << "\n";
//
//		//NA_Vector t = temp.currentVelocity;
//		//t.normalise();
//		//cout << "NV: X: " << t.x << " Y: " << t.y << "\n\n";
//
//
//	}
//
//
//	// enter game loop..
//	graphics.loop();	
//
//	return 0;
//}


/////////////////////////////////////////////////////////////////////////
// CUDA Main
/////////////////////////////////////////////////////////////////////////

//general notes
#define CUDA_CALL(x) { const cudaError_t a = (x); if (a != cudaSuccess) {printf("\nCUDA Error: %s (err_num=%d) \n",cudaGetErrorScring(a),a); cudaDeviceReset(); assert(0);}} // Shane Cook - CUDA Programming A Developer's Guide to Parallel Computing with GPUs ISBN:978-0-12-415933-4 P67
//cudaDeviceSynchronize(); //have the CPU wait for the GPU

//cudaKernal<<<num_blocks, num_threads>>>(args ...); //	running the kernal on the GPU

struct psudoVector2
{
	float x;
	float y;
};

struct psudoBoid
{
	psudoVector2 position;
	psudoVector2 currentVelocity;
};



__global__ void cudaBoidUpdate(psudoBoid* globalBoidArray, int loopCount)
{
	printf("kernel launched\n");
	int selfIndex = (int)threadIdx.x; // slightly more readable and means less casting

   // every boid copies it own data into the shared memory
	__shared__ psudoBoid sharedBoidArray[BOID_MAX];
	sharedBoidArray[selfIndex] = globalBoidArray[selfIndex];

	printf("init complete\n");

	psudoBoid localBoidArray[BOID_MAX];


	for (int loop = 0; loop < loopCount; loop++)
	{
		printf("beginning loop %d\n", loop);
		// rebuild cache
		// starting at own boid, copy data into own memory

		int i = selfIndex;
		for (int j = 0; j < BOID_MAX; j++)
		{

			// actual copy
			localBoidArray[i] = sharedBoidArray[i];

			i++; // next boid
			if (i == BOID_MAX) // wrap arround when walking off memeory
				i = 0;
		}
		printf("local cache rebuilt\n");

		// find which boids are in range
		int nearbyBoidIndexer[BOID_MAX]; //save memory with a trick
		int nearbyBoidIndexSize = 0;

		for (int i = 0; i < BOID_MAX; i++)
		{
			if (i == selfIndex)
			{
				//skip
			}
			else
			{
				psudoVector2 temp;
				temp.x = localBoidArray[i].position.x - localBoidArray[selfIndex].position.x;
				temp.y = localBoidArray[i].position.y - localBoidArray[selfIndex].position.y;
				int tempLength = sqrt(temp.x*temp.x + temp.y*temp.y);
				if (tempLength < BIOD_SIGHT_RANGE)
				{
					nearbyBoidIndexer[nearbyBoidIndexSize] = i;
					nearbyBoidIndexSize++;
				}
			}
		}

		printf("sort list made\n");

		// alightment
		psudoVector2 sumVelocity;
		sumVelocity.x = 0;
		sumVelocity.y = 0;

		for (int i = 0; i < nearbyBoidIndexSize; i++)
		{
			sumVelocity.x += localBoidArray[nearbyBoidIndexer[i]].currentVelocity.x;
			sumVelocity.y += localBoidArray[nearbyBoidIndexer[i]].currentVelocity.y;
		}
		// convert to average
		sumVelocity.x = sumVelocity.x / nearbyBoidIndexSize;
		sumVelocity.y = sumVelocity.y / nearbyBoidIndexSize;

		psudoVector2 newVelocity = sumVelocity;

		printf("alignment found\n");

		// cohesion
		psudoVector2 sumPosition;
		sumPosition.x = 0;
		sumPosition.y = 0;

		for (int i = 0; i < nearbyBoidIndexSize; i++) // just realised I could combine this loop with the previous one
		// keeping them seperate to maintain readability
		{
			sumPosition.x += localBoidArray[nearbyBoidIndexer[i]].position.x;
			sumPosition.y += localBoidArray[nearbyBoidIndexer[i]].position.y;
		}
		// convert to average
		sumPosition.x = sumPosition.x / nearbyBoidIndexSize;
		sumPosition.y = sumPosition.y / nearbyBoidIndexSize;

		printf("cohesion done\n");

		// seperation
		for (int i = 0; i < nearbyBoidIndexSize; i++) // another for loop that could be merged?
		{
			if (nearbyBoidIndexer[i] != selfIndex) // skip self
			{
				psudoVector2 temp;
				temp.x = localBoidArray[selfIndex].position.x - localBoidArray[i].position.x;
				temp.y = localBoidArray[selfIndex].position.y - localBoidArray[i].position.y;
				int tempLength = sqrt(temp.x*temp.x + temp.y*temp.y);
				if (tempLength < BOID_RESPECT_DIST)
				{
					newVelocity = temp;
				}
			}
		}

		printf("seperation done\n");

		// STUFF FROM CPU POST UPDATE METHOD

	   // enforce rotation limit
	   // commented out due to bug in NA_Vector::clockwiseAngle - it doesn't give a different value when you mesure from the other vector. Thiss means that the CPU version has this bug
	   /*float newVelocityCurrentVelocityClockwiseAngle; //this is going to get messy - missing my vector library now

		 float newVelocityLenSq = newVelocity.x*newVelocity.x + newVelocity.y*newVelocity.y; // I could possibly do some #defines for readability
		 float currentVelocityLenSq = currentVelocity.x*currentVelocity.x + currentVelocity.y*currentVelocity.y;
		 float dotProduct = newVelocity.x*currentVelocity.x + newVelocity.y*currentVelocity.y;
		 newVelocityCurrentVelocityClockwiseAngle = acos(dotProduct / sqrt(newVelocityLenSq * currentVelocityLenSq));

	   float currentVelocityNewVelocityCClockwiseAngle; //there is a difference, the velocities are swapped

		 float newVelocityLenSq = newVelocity.x*newVelocity.x + newVelocity.y*newVelocity.y; // I could possibly do some #defines for readability
		 float currentVelocityLenSq = currentVelocity.x*currentVelocity.x + currentVelocity.y*currentVelocity.y;
		 float dotProduct = newVelocity.x*currentVelocity.x + newVelocity.y*currentVelocity.y;
		 newVelocityCurrentVelocityClockwiseAngle = acos(dotProduct / sqrt(newVelocityLenSq * currentVelocityLenSq));


	   if (newVelocityCurrentVelocityClockwiseAngle > BOID_ROTATE_MAX && currentVelocityNewVelocityCClockwiseAngle > BOID_ROTATE_MAX)
	   {

		 if (newVelocityCurrentVelocityClockwiseAngle < currentVelocityNewVelocityCClockwiseAngle)//clockwise or counterclockwise?
		 {

		   NA_Matrix r = NA_Matrix(NA_Matrix::types::rotateZ, BOID_ROTATE_MAX);
		   newVelocity = r.matrixXvector(newVelocity);
		 }
		 else
		 {

		   NA_Matrix r = NA_Matrix(NA_Matrix::types::rotateZ, -BOID_ROTATE_MAX);
		   newVelocity = r.matrixXvector(newVelocity);
		 }
	   }*/

	   // enforec speed limit
		float l = sqrt(newVelocity.x*newVelocity.x + newVelocity.y*newVelocity.y);
		if (l > BOID_SPEED_MAX);
		{
			// normalise and then scale
			newVelocity.x = (newVelocity.x / l)*BOID_SPEED_MAX;
			newVelocity.y = (newVelocity.y / l)*BOID_SPEED_MAX;
		}

		printf("obaying the speed limit\n");

		// update position with velocity
		localBoidArray[selfIndex].currentVelocity = newVelocity;
		localBoidArray[selfIndex].position.x += newVelocity.x;
		localBoidArray[selfIndex].position.y += newVelocity.y;

		printf("updated local cache\n");

		// screen wrap
		if (localBoidArray[selfIndex].position.x < 0)
			localBoidArray[selfIndex].position.x += SCREEN_WIDTH;
		if (localBoidArray[selfIndex].position.x > SCREEN_WIDTH)
			localBoidArray[selfIndex].position.x -= SCREEN_WIDTH;

		if (localBoidArray[selfIndex].position.y < 0)
			localBoidArray[selfIndex].position.y += SCREEN_HEIGHT;
		if (localBoidArray[selfIndex].position.y > SCREEN_HEIGHT)
			localBoidArray[selfIndex].position.y -= SCREEN_HEIGHT;

		printf("staying within the world\n");

		printf("waiting for everyone\n");
		__syncthreads();

		// update shared data
		sharedBoidArray[selfIndex] = localBoidArray[selfIndex];

		printf("updated shared info\n");

		//TODO: cuda/opengl interop render

		// wait for all threads (get ready for next round)

		printf("waiting for next loop\n");
		__syncthreads();
	}

	// put stuff back in global memory so that CPU can collect it if wanted
	globalBoidArray[selfIndex] = sharedBoidArray[selfIndex];
	
}




int _tmain(int argc, _TCHAR* argv[])
{

	const int numberOfBlocks = 1;
	const int numberOfThreadsPerBlock = BOID_MAX;
	const int loopCount = 1;

	// set up cuda
	cudaError err = cudaSetDevice(0);
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to set device\n";
		getchar();
		return -1;
	}
	// make all boids
	na_maths.seedDice();
	psudoBoid boidArray[BOID_MAX];
	for (int i = 0; i < BOID_MAX; i++)
	{

		boidArray[i].position.x = na_maths.dice(SCREEN_WIDTH);
		boidArray[i].position.y = na_maths.dice(SCREEN_HEIGHT);

		boidArray[i].currentVelocity.x = float(na_maths.dice(-100, 100)) / 100.0f;
		boidArray[i].currentVelocity.y = float(na_maths.dice(-100, 100)) / 100.0f;

	}

	// tell cuda to allocate space for boids and copy boids to cuda

	psudoBoid* deviceBoidArray;
	err = cudaMalloc((void**)&deviceBoidArray, BOID_MAX * sizeof(psudoBoid));
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to allocate memory on device\n";
		cudaFree(deviceBoidArray);
		getchar();
		return -1;
	}

	err = cudaMemcpy(deviceBoidArray, boidArray, BOID_MAX * sizeof(psudoBoid), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to copy memory to device\n";
		cudaFree(deviceBoidArray);
		getchar();
		return -1;
	}

	// loopCount is a normal variable, no need to cudaMalloc and CudaMemcpy

	// run kernel
	std::cout << "Simulating boids\n";
	cudaBoidUpdate << <numberOfBlocks, numberOfThreadsPerBlock >> >(deviceBoidArray, loopCount);

	

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to launch kernel: " << cudaGetErrorString(err) << "\n";
		cudaFree(deviceBoidArray);
		getchar();
		return -1;
	}

	// wait for GPU to finish
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - cudaDeviceSync returned " << err << " errorString = " << cudaGetErrorString(err) << "\n";
		cudaFree(deviceBoidArray);
		getchar();
		return -1;
	}

	// all ok, cleanup and exit
	cudaFree(deviceBoidArray);
	cout << "all done\n";

	getchar();
	return 0;
}

