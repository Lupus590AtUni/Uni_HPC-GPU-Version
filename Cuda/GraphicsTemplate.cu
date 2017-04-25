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

__global__ void cudaBoidUpdate(psudoBoid* boidArray, int loopCount)
{
	//TODO: offset copy

	// starting at own boid, copy data into own memory
	int i = (int) threadIdx.x;
	for (int j = 0; j < BOID_MAX; j++)
	{
		
		//TODO: actual copy


		i++; // next boid
		if (i == BOID_MAX) // wrap arround when walking off memeory
			i = 0;
	}

	//TODO: find which boids are in range (perhaps should do this before copy as only would need to look at position

	psudoVector2 newVelocity;

	//TODO: alightment

	//TODO: cohesion

	//TODO: seperation

	//TODO: enforce speed/rotation limits

	//TODO: screen wrap

	//TODO: wait for all threads

	//TODO: update global data

	//TODO: cuda/opengl interop render

	//TODO: wait for all threads (get ready for next round)

}



int _tmain(int argc, _TCHAR* argv[])
{

	const int blockCount = 1;
	const int threadCount = BOID_MAX;
	const int loopCount = 10000;

	// set up cuda
	cudaError err = cudaSetDevice(0);
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to set device\n";
		return -1;
	}
	// make all boids
	na_maths.seedDice();
	psudoBoid boidArray[BOID_MAX];
	for (int i = 0; i < BOID_MAX; i++)
	{
		
		boidArray[i].position.x = na_maths.dice(SCREEN_WIDTH);
		boidArray[i].position.y = na_maths.dice(SCREEN_HEIGHT);
	
		boidArray[i].currentVelocity.x = float(na_maths.dice(-100,100))/100.0f;
		boidArray[i].currentVelocity.y = float(na_maths.dice(-100, 100))/100.0f;
	
	}

	// tell cuda to allocate space for boids and copy boids to cuda

	psudoBoid* deviceBoidArray;
	err = cudaMalloc((void**)&deviceBoidArray, BOID_MAX * sizeof(psudoBoid));
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to allocate memory on device\n";
		cudaFree(deviceBoidArray);
		return -1;
	}

	err = cudaMemcpy(deviceBoidArray, boidArray, BOID_MAX * sizeof(psudoBoid), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to copy memory to device\n";
		cudaFree(deviceBoidArray);
		return -1;
	}

	// loopCount is a normal variable, no need to cudaMalloc and CudaMemcpy

	// run kernel
	cudaBoidUpdate << <blockCount, threadCount >> >(deviceBoidArray, loopCount);

	cout << "Simulating boids\n";

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - failed to launch kernel: " << cudaGetErrorString(err)<<"\n";
		cudaFree(deviceBoidArray);
		return -1;
	}

	// wait for GPU to finish
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		cerr << "GraphicsTemplate::_tmain - cudaDeviceSync returned " << err << " errorString = " << cudaGetErrorString(err) << "\n";
		cudaFree(deviceBoidArray);
		return -1;
	}
	
	// all ok, cleanup and exit
	cudaFree(deviceBoidArray);

	return 0;
}

