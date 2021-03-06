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
#include "globals.cuh"
#include <vector>
using std::vector;


#include "NA_MathsLib.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

//http://www.cplusplus.com/reference/chrono/high_resolution_clock/now/
#include <ctime>
#include <ratio>
#include <chrono>
using namespace std::chrono;

//http://www.cplusplus.com/reference/string/stoi/
#include <string> 



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



__global__ void cudaBoidUpdate(psudoBoid* globalBoidArray, int loopCount, const int BOID_MAX)
{
	bool debug = false;
	bool allThreadsDebug = false;
	int selfIndex = (int)threadIdx.x; // slightly more readable and means less casting
	if ((selfIndex == 0 && debug) || allThreadsDebug) printf("kernel launched\n");
	
  // create shared array for communicating between threads/boids
	__shared__ psudoBoid* sharedBoidArray;
	if(selfIndex == 0) sharedBoidArray = (psudoBoid*)malloc(BOID_MAX * sizeof(psudoBoid));
	__syncthreads();
	sharedBoidArray[selfIndex] = globalBoidArray[selfIndex]; // every boid copies it own data into the shared memory

	if ((selfIndex ==0 && debug) || allThreadsDebug) printf("init complete\n");

	psudoBoid* localBoidArray = (psudoBoid*)malloc(BOID_MAX * sizeof(psudoBoid)); // each thread has its own cache of the shared array, this is likely the cause of 551 boid limit

	int* nearbyBoidIndexer = (int*)malloc(BOID_MAX * sizeof(int));  //save memory while creating short list with a trick
	for (int loop = 0; loop < loopCount; loop++)
	{
		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("beginning loop %d\n", loop);
		// rebuild localBoidArray cache
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
		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("local cache rebuilt\n");

		// find which boids are in range
		
		int nearbyBoidIndexSize = 0;

		for (int i = 0; i < BOID_MAX; i++)
		{
      // CPU version had a bug here
			//if (i == selfIndex)
			//{
				//skip
			//}
			//else
			//{
				psudoVector2 temp;
				temp.x = localBoidArray[i].position.x - localBoidArray[selfIndex].position.x;
				temp.y = localBoidArray[i].position.y - localBoidArray[selfIndex].position.y;
				int tempLength = sqrt(temp.x*temp.x + temp.y*temp.y);
				if (tempLength < BIOD_SIGHT_RANGE) // if the boid is close enough to be seen then add it to the nearby list
				{
					nearbyBoidIndexer[nearbyBoidIndexSize] = i;
					nearbyBoidIndexSize++;
				}
			//}
		}

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("sort list made\n");

		// alightment
		psudoVector2 sumVelocity;
		sumVelocity.x = 0;
		sumVelocity.y = 0;

    // nearbyBoidIndexer trick in action
		for (int i = 0; i < nearbyBoidIndexSize; i++)
		{
			sumVelocity.x += localBoidArray[nearbyBoidIndexer[i]].currentVelocity.x;
			sumVelocity.y += localBoidArray[nearbyBoidIndexer[i]].currentVelocity.y;
		}
		// convert to average
		sumVelocity.x = sumVelocity.x / nearbyBoidIndexSize;
		sumVelocity.y = sumVelocity.y / nearbyBoidIndexSize;

		psudoVector2 newVelocity = sumVelocity;
		
		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("alignment found\n");

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
    
    psudoVector2 temp;
    temp.x = localBoidArray[selfIndex].position.x - sumPosition.x;
    temp.y = localBoidArray[selfIndex].position.y - sumPosition.y;
	
    temp.x *= BOID_COHESION_WEIGHTING;
    temp.y *= BOID_COHESION_WEIGHTING;

    // modify velocity to head towards the average position
    newVelocity.x += temp.x;
    newVelocity.y += temp.y;

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("cohesion done\n");
		
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

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("seperation done\n");

		// STUFF FROM CPU POST UPDATE METHOD

	   // enforce rotation limit
	   // commented out due to bug in NA_Vector::clockwiseAngle - it doesn't give a different value when you mesure from the other vector. Thiss means that the CPU version had this bug
     // also porting this to cuda would have made ugly code
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

	   // enforce speed limit
		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("enforcing the speed limit\n");
		float l = sqrt(newVelocity.x*newVelocity.x + newVelocity.y*newVelocity.y);
		if (l > BOID_SPEED_MAX);
		{
			// normalise and then scale
			newVelocity.x = (newVelocity.x / l)*BOID_SPEED_MAX; // I occasionally had NaN in boid data, I suspect that is was caused here from when the boid filtered itself out of the nearbyBoidIndexer
			newVelocity.y = (newVelocity.y / l)*BOID_SPEED_MAX;
		}

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("obaying the speed limit\n");

		// update position with velocity
		localBoidArray[selfIndex].currentVelocity = newVelocity;
		localBoidArray[selfIndex].position.x += newVelocity.x;
		localBoidArray[selfIndex].position.y += newVelocity.y;

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("updated local cache\n");

		// screen wrap
		if (localBoidArray[selfIndex].position.x < 0)
			localBoidArray[selfIndex].position.x += SCREEN_WIDTH;
		if (localBoidArray[selfIndex].position.x > SCREEN_WIDTH)
			localBoidArray[selfIndex].position.x -= SCREEN_WIDTH;

		if (localBoidArray[selfIndex].position.y < 0)
			localBoidArray[selfIndex].position.y += SCREEN_HEIGHT;
		if (localBoidArray[selfIndex].position.y > SCREEN_HEIGHT)
			localBoidArray[selfIndex].position.y -= SCREEN_HEIGHT;

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("staying within the world\n");

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("waiting for everyone\n");
		__syncthreads();

		// update shared data
		sharedBoidArray[selfIndex] = localBoidArray[selfIndex];

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("updated shared info\n");

		//TODO: cuda/opengl interop render

		// wait for all threads (get ready for next round)

		if ((selfIndex ==0 && debug) || allThreadsDebug) printf("waiting for next loop\n");
		__syncthreads();
	}

  // threads cleanup their local stuff
	free(nearbyBoidIndexer);
	free(localBoidArray);

	// put stuff back in global memory so that CPU can collect it if wanted
	globalBoidArray[selfIndex] = sharedBoidArray[selfIndex];
	__syncthreads();
	if (selfIndex == 0) free(sharedBoidArray);
	
	
}




int _tmain(int argc, _TCHAR* argv[])
{
  
	int loopCount;
	if (argc != 3)
	{
		std::cerr << "usage: " << argv[0] << " <boidCount> <loopCount> \n";
		cout << "errored\n";
		return -1;
	}
	else
	{
    // convert to ints, not sure what happens if can't
		BOID_MAX = std::stoi(argv[1], NULL); //http://www.cplusplus.com/reference/string/stoi/
		loopCount = std::stoi(argv[2], NULL);
	}

	const int numberOfBlocks = 1;
	const int numberOfThreadsPerBlock = BOID_MAX; // expect errors with over 1024 boids

	// set up cuda
	cudaError err = cudaSetDevice(0);
	cudaDeviceReset();
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to set device\n";
		cout << "errored\n";
		cudaDeviceReset();
		return -1;
	}
	// make all boids
	extern NA_MathsLib na_maths;
	na_maths.seedDice(0); // fixed seed matches CPU version
	psudoBoid* boidArray = (psudoBoid*) malloc(BOID_MAX * sizeof(psudoBoid));
	//psudoBoid boidArray[BOID_MAX];
  //set initual values
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
		free(boidArray);
		cout << "errored\n";
		cudaDeviceReset();
		return -1;
	}

	err = cudaMemcpy(deviceBoidArray, boidArray, BOID_MAX * sizeof(psudoBoid), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to copy memory to device\n";
		cudaFree(deviceBoidArray);
		free(boidArray);
		cout << "errored\n";
		cudaDeviceReset();
		return -1;
	}

	// loopCount is a normal variable, no need to cudaMalloc and CudaMemcpy

	// run kernel
	//std::cout << "Simulating boids\n";
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
	cudaBoidUpdate << <numberOfBlocks, numberOfThreadsPerBlock>> >(deviceBoidArray, loopCount, BOID_MAX); // launch the GPU kernel

	

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to launch kernel: " << cudaGetErrorString(err) << "\n";
		cudaFree(deviceBoidArray);
		free(boidArray);
		cout << "errored\n";
		cudaDeviceReset();
		return -1;
	}

	// wait for GPU to finish
	err = cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  
  
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - cudaDeviceSync returned " << err << " errorString = " << cudaGetErrorString(err) << "\n";
		cudaFree(deviceBoidArray);
		free(boidArray);
		cout << "errored\n";
		cudaDeviceReset();
		return -1;
	}

  // test did not error, calculate time taken and printout
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

	cout << time_span.count() << "\n";

	// all ok, cleanup and exit
	cudaFree(deviceBoidArray);
	//cout << "all done\n";

	free(boidArray);
	std::cerr << "all ok\n"; // only here so that the error cvs file is easier to read
	cudaDeviceReset();
	return 0;
}
