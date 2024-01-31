#include <iostream>
#include <FreeImage.h>
#include <time.h>

#define THR 50
#define MAX 255
#define MIN 0

#define MASK_DIM 3

#define MASK_OFFSET (MASK_DIM / 2)

__constant__ int  mask[MASK_DIM * MASK_DIM];

__global__ void RgbToGrey(const BYTE* in, BYTE* out, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float   temp_r = 0.0f;
    float   temp_g = 0.0f;
    float   temp_b = 0.0f;

    // Check if the thread is within the image dimensions
    if (row < height && col < width)
    {

	int start_r  = row - MASK_OFFSET;
	int start_c  = col - MASK_OFFSET;
        int pixelIndex = row * width + col;
	


	//if(row != 0 && col != 0) {
	 // Iterate over all the rows
  	for (int i = -1; i <= 1; i++) {
    	// Go over each column
    	for (int j = -1; j <= 1; j++) {


//	if((row + i) > 0 && (row + i ) < height) {
//		if((col + j) > 0 && (col + j) < width) {
          // Accumulate result
	
	  //printf(" *G* number p1  is %d \n", (row + i));

	  //printf(" *G* number p2  is %d \n", (col + j));



	  //printf(" *G* number P3  is %d \n", ((i + row) * width + (j + col)) * 3);

	  //printf(" *G* number P4  is %d \n", (i  + row) * MASK_DIM + (j + col) );

          temp_r		+= in[((i + row) * width + (j + col)) * 3] * mask[(i + 1)  * MASK_DIM + (j + 1) ];


          temp_g		+= in[((i + row) * width + (j + col)) * 3 + 1] * mask[(i + 1) * MASK_DIM + (j + 1)];


          temp_b 		+= in[((i + row) * width + (j + col)) * 3 + 2] * mask[((i + 1)  * MASK_DIM + (j + 1))];


	 // temp_r += in[((i + row) * width + (j + col)) * 3] * mask[(i + 1) * MASK_DIM + (j + 1)];
	 // temp_g += in[((i + row) * width + (j + col)) * 3 + 1] * mask[(i + 1) * MASK_DIM + (j + 1)];
	 // temp_b += in[((i + row) * width + (j + col)) * 3 + 2] * mask[(i + 1) * MASK_DIM + (j + 1)];

          //out[((i + 1) * width + (j + 1) * 3) + 1]		+= in[((start_r + i) * width + (start_c + j) * 3) + 1] * mask[(start_r + i) * MASK_DIM + ((start_c + j))];

          //out[((j + 1) * width + (j + 1) * 3) + 2] 		+= in[((start_r + i) * width + (start_c + j) * 3) + 2] * mask[(start_r + i) * MASK_DIM + ((start_c + j))];
	//		}
	//	}
		}
    }
  

	//out[((i + row) * width + (j + col)) * 3] 	= static_cast<BYTE>(temp_r);
	//out[((i + row) * width + (j + col)) * 3 + 1] 	= static_cast<BYTE>(temp_g);
	//out[((i + row) * width + (j + col)) * 3 + 2] 	= static_cast<BYTE>(temp_b);

  	// Write back the result
  	out[(row * width + col) * 3] = temp_r;
	
  	out[(row * width + col) * 3 + 1] = temp_g;
	
  	out[(row * width + col) * 3 + 2] = temp_b;
	//}

	//out[pixelIndex * 3] = (in[pixelIndex * 3] + THR)>MAX? MAX:(in[pixelIndex * 3] + THR); //red

	//out[pixelIndex * 3 + 1] = (in[pixelIndex * 3 + 1 ] + THR)>MAX? MAX:(in[pixelIndex * 3 + 1 ] + THR); //green

	//out[pixelIndex * 3 + 2] = (in[pixelIndex * 3 + 2 ] + THR)>MAX? MAX:(in[pixelIndex * 3 + 2 ] + THR); //blue
	
        //float temp = (in[pixelIndex * 3] * 0.3f + in[pixelIndex * 3 + 1] * 0.59f + in[pixelIndex * 3 + 2] * 0.11f);

        // Store the result in the appropriate channel of the output image
        //out[pixelIndex * 3] = static_cast<BYTE>(temp);
        //out[pixelIndex * 3 + 1] = static_cast<BYTE>(temp);
        //out[pixelIndex * 3 + 2] = static_cast<BYTE>(temp);
    } 
}

#if 0
__global__ void RgbToGrey(const BYTE* in, BYTE* out, int width, int height)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread processes one pixel (three channels) at a time
    int pixelIndex = gid / 3;

    // Check if the pixel index is within the image dimensions
    if (pixelIndex < width * height)
    {
        int row = pixelIndex / width;
        int col = pixelIndex % width;
        int channel = gid % 3;

        float temp = (in[row * width * 3 + col * 3] * 0.3f + in[row * width * 3 + col * 3 + 1] * 0.59f + in[row * width * 3 + col * 3 + 2] * 0.11f);

        // Store the result in the appropriate channel of the output image
        out[row * width * 3 + col * 3 + channel] = static_cast<BYTE>(temp);
    }
}
#endif
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
    const char* inputFileName = "output.bmp";
    const char* outputFileName = "grey.bmp";

     struct timespec currentTime1,currentTime2;

    // Load input BMP image using FreeImage
    FIBITMAP* inputImage = FreeImage_Load(FIF_BMP, inputFileName, BMP_DEFAULT);
    if (!inputImage)
    {
        std::cerr << "Error loading input image" << std::endl;
        return EXIT_FAILURE;
    }

    //int h_mat[3*3];


	//horizontal Edge detection
    int horizontal[9] = {-1,-1,-1,0,0,0,1,1,1};
    int vertical[9] = {-1,0,1,-1,0,1,-1,0,1}; 

     int ch;


     //int *h_mat = (int*)malloc(9 * sizeof(int));

     int h_mat[9];
     printf("1. Horizontal\n 2, Vertical Edge detection \n Enter your  choice\n");
     scanf("%d",&ch);

     switch(ch){
	     case 1: 
		     memcpy((void*)&h_mat,(void*)&horizontal, (9 * sizeof(int)));
		     break;
	     case 2:
		     memcpy((void*)&h_mat,(void*)&vertical, (9 * sizeof(int)));
		     break;

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

    // Launch GPU kernel
    //dim3 blockDim(256, 1);
    //dim3 gridDim((width * height + blockDim.x - 1) / blockDim.x, 1);


    // Size of the mask in bytes
    size_t bytes_m = (MASK_DIM * MASK_DIM * 4);
    
    clock_gettime(CLOCK_REALTIME, &currentTime1);
    long milliseconds1 = currentTime1.tv_nsec / 1000000;
    long seconds1 = currentTime1.tv_sec * 1000;

    dim3 blockDim(16, 16); // Adjust block dimensions as needed
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaMemcpyToSymbol(mask, h_mat, bytes_m);


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
