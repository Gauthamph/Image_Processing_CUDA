#include <iostream>
#include <FreeImage.h>
#include <time.h>

#define THR 50
#define MAX 255
#define MIN 0

__global__ void RgbToGrey(const BYTE* in, BYTE* out, int width, int height)
{

	float red = 0.0f;
	float blue = 0.0f;
	float green = 0.0f;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int pixelIndex = row * width + col;
	// Check if the thread is within the image dimensions
	if (row < height && col < width)
	{
		blue   = in[pixelIndex * 3] * 0.392 + in[pixelIndex * 3 + 1 ] * 0.769 + in[pixelIndex * 3 + 2 ] * 0.189; //blue
		green  = in[pixelIndex * 3] * 0.349 + in[pixelIndex * 3 + 1 ] * 0.686 + in[pixelIndex * 3 + 2 ] * 0.168; //green
		red    = in[pixelIndex * 3] * 0.272 + in[pixelIndex * 3 + 1 ] * 0.534 + in[pixelIndex * 3 + 2 ] * 0.131; //red
	}

	out[pixelIndex * 3] = blue;
	out[pixelIndex * 3 + 1] = green;
	out[pixelIndex * 3 + 2] = red;
}

void checkError(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
		exit(EXIT_FAILURE);
	}
}

int main()
{
	const char* inputFileName = "inout.bmp";
	const char* outputFileName = "output.bmp";

	struct timespec currentTime1,currentTime2;

	// Load input BMP image using FreeImage
	FIBITMAP* inputImage = FreeImage_Load(FIF_BMP, inputFileName, BMP_DEFAULT);
	if (!inputImage)
	{
		std::cerr << "Error loading input image" << std::endl;
		return EXIT_FAILURE;
	}

	// Get image dimensions
	int width = FreeImage_GetWidth(inputImage);
	int height = FreeImage_GetHeight(inputImage);

	// Convert image to 24-bit RGB format
	FIBITMAP* inputImageRGB = FreeImage_ConvertTo24Bits(inputImage);

	// Allocate memory for GPU buffers
	BYTE* d_input, *d_output;
	checkError(cudaMalloc((void**)&d_input, width * height * 3));
	checkError(cudaMalloc((void**)&d_output, width * height * 3));

	// Copy input image data to GPU
	checkError(cudaMemcpy(d_input, FreeImage_GetBits(inputImageRGB), width * height * 3, cudaMemcpyHostToDevice));


	clock_gettime(CLOCK_REALTIME, &currentTime1);
	long milliseconds1 = currentTime1.tv_nsec / 1000000;
	long seconds1 = currentTime1.tv_sec * 1000;

	dim3 blockDim(16, 16); // Adjust block dimensions as needed
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	//printf("blockDim.x %d blockDim.y %d number of grids in x direction %d number of grids in y direction %d \n", blockDim.x, blockDim.y,(width + blockDim.x - 1) / blockDim , (height + blockDim.y - 1) / blockDim.y);



	RgbToGrey<<<gridDim, blockDim>>>(d_input, d_output, width, height);
	checkError(cudaGetLastError());
	checkError(cudaDeviceSynchronize());


	clock_gettime(CLOCK_REALTIME, &currentTime2);
	long milliseconds2 = currentTime2.tv_nsec / 1000000;
	long seconds2 = currentTime2.tv_sec * 1000;

	printf("difference in time is %ld \n",((seconds2+milliseconds2) - (seconds1+milliseconds1)));
	// Copy result from GPU to host
	BYTE* h_output = (BYTE*)malloc(width * height * 3);
	checkError(cudaMemcpy(h_output, d_output, width * height * 3, cudaMemcpyDeviceToHost));


	// Save output image using FreeImage
	FIBITMAP* outputImage = FreeImage_ConvertFromRawBits(h_output, width, height, 3 * width, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);
	FreeImage_Save(FIF_BMP, outputImage, outputFileName);

	// Cleanup
	cudaFree(d_input);
	cudaFree(d_output);
	FreeImage_Unload(inputImage);
	FreeImage_Unload(inputImageRGB);
	free(h_output);

	return EXIT_SUCCESS;
}
