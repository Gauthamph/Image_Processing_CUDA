#include <iostream>
#include <FreeImage.h>
#include <time.h>

__global__ void RgbToGrey(const BYTE* in, BYTE* out, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the image dimensions
    if (row < height && col < width)
    {
        int pixelIndex = row * width + col;

        float temp = (in[pixelIndex * 3] * 0.3f + in[pixelIndex * 3 + 1] * 0.59f + in[pixelIndex * 3 + 2] * 0.11f);

        // Store the result in the appropriate channel of the output image
        out[pixelIndex * 3] = static_cast<BYTE>(temp);
        out[pixelIndex * 3 + 1] = static_cast<BYTE>(temp);
        out[pixelIndex * 3 + 2] = static_cast<BYTE>(temp);
    }
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
    const char* inputFileName = "input.bmp";
    const char* outputFileName = "grey.bmp";

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

    // Launch GPU kernel
    //dim3 blockDim(256, 1);
    //dim3 gridDim((width * height + blockDim.x - 1) / blockDim.x, 1);

        clock_gettime(CLOCK_REALTIME, &currentTime1);
        long milliseconds1 = currentTime1.tv_nsec / 1000000;
        long seconds1 = currentTime1.tv_sec * 1000;

    dim3 blockDim(16, 16); // Adjust block dimensions as needed
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);




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
