# Image_Processing_CUDA
Image processing on a parallel processing platform (CUDA) by implementing algorithms to speed up execution more than 600% compared to CPU.
Operations are done on BGR data of BMP file, you may need to install Freeimages API for file operations on Images


Executed all programs on nVidia Jetson nano.
You may need freeimage package, on Ubuntu you can install by command 'sudo apt-get install libfreeimage3 libfreeimage-dev'

Steps to Execute

nvcc program_name -o output_name -lfreeimage


