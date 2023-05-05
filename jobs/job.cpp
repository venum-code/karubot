#include <iostream>
#include <fstream>
#include <CL/opencl.hpp>
#include <CL/cl.h>
#include <string>

int main() {
//Step 1: Create an OpenCL context, command queue and program object.
cl_context context;
cl_command_queue queue;
cl_program program;

// create an OpenCL context
cl_platform_id platform = NULL;
cl_device_id device = NULL;
clGetPlatformIDs(1, &platform, NULL);
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

// create a command queue
queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

// create a program object
std::ifstream file("histogram_equalization.cl");
std::string source(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
const char *sourcePtr = source.c_str();
program = clCreateProgramWithSource(context, 1, &sourcePtr, NULL, NULL);
clBuildProgram(program, 1, &device, NULL, NULL, NULL);

//Step 2: Define kernels for histogram calculation, scan and back-projection.
cl_kernel histogram_kernel, scan_kernel, backprojection_kernel;

// create histogram kernel
histogram_kernel = clCreateKernel(program, "histogram", NULL);

// create scan kernel
scan_kernel = clCreateKernel(program, "scan", NULL);

// create back-projection kernel
backprojection_kernel = clCreateKernel(program, "backprojection", NULL);

//Step 3: Allocate memory for input and output images, histograms, and cumulative histograms.
cl_mem input_image, output_image, histogram, cumulative_histogram;

// allocate memory for input and output images
size_t image_size = 1024; 
input_image = clCreateBuffer(context, CL_MEM_READ_ONLY, image_size, NULL, NULL);
output_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_size, NULL, NULL);

// allocate memory for histograms and cumulative histograms
int num_bins = 10; 
size_t histogram_size = num_bins * sizeof(unsigned int);
histogram = clCreateBuffer(context, CL_MEM_READ_WRITE, histogram_size, NULL, NULL);
cumulative_histogram = clCreateBuffer(context, CL_MEM_READ_WRITE, histogram_size, NULL, NULL);

//Step 4: Load the input image from a file into the allocated memory.
// load input image from file
unsigned char* input_image_data = new unsigned char[image_size];
std::ifstream input_file("input_image.raw", std::ios::binary);
input_file.read(reinterpret_cast<char*>(input_image_data), image_size);

// copy input image data to OpenCL device memory
clEnqueueWriteBuffer(queue, input_image, CL_TRUE, 0, image_size, input_image_data, 0, NULL, NULL);

// free host memory
delete[] input_image_data;

//Step 5: Calculate the histogram of input image using a parallel kernel.
// set kernel arguments
// declare the kernel
cl_kernel hist_kernel;
cl_int err = 0;
unsigned int num_pixels = 42; // replace <value> with the actual value
// create the kernel from the program
hist_kernel = clCreateKernel(program, "histogram", &err);

// check for errors in creating the kernel
if (err != CL_SUCCESS) {
    printf("Error: Failed to create kernel!\n");
    return EXIT_FAILURE;
}
clSetKernelArg(hist_kernel, 0, sizeof(cl_mem), &input_image);
clSetKernelArg(hist_kernel, 1, sizeof(cl_mem), &histogram);
clSetKernelArg(hist_kernel, 2, sizeof(unsigned int), &num_pixels);
clSetKernelArg(hist_kernel, 3, sizeof(unsigned int), &num_bins);

// enqueue kernel
size_t global_size = num_bins;
size_t local_size = num_bins; // or use a smaller local size if memory is limited
clEnqueueNDRangeKernel(queue, hist_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
// wait for kernel to finish
clFinish(queue);

//Step 6: Calculate the cumulative histogram from the histogram.
// declare kernel
cl_kernel cumulative_hist_kernel = clCreateKernel(program, "cumulative_hist_kernel", &err);
if (err != CL_SUCCESS) {
    printf("Error: Failed to create kernel!\n");
    return EXIT_FAILURE;
}
// set kernel arguments
clSetKernelArg(cumulative_hist_kernel, 0, sizeof(cl_mem), &histogram);
clSetKernelArg(cumulative_hist_kernel, 1, sizeof(cl_mem), &cumulative_histogram);
clSetKernelArg(cumulative_hist_kernel, 2, sizeof(unsigned int), &num_bins);

// enqueue kernel
// or use a smaller local size if memory is limited
clEnqueueNDRangeKernel(queue, cumulative_hist_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
// wait for kernel to finish
clFinish(queue);

//Step 7: Normalize and scale the cumulative histogram to represent output image intensities.
// find maximum value in cumulative histogram
unsigned int max_val = 0;
for (int i = 0; i < num_bins; i++) {
    if (cumulative_histogram[i] > max_val) {
        max_val = cumulative_histogram[i];
    }
}

// set kernel arguments
clSetKernelArg(normalize_kernel, 0, sizeof(cl_mem), &cumulative_histogram);
clSetKernelArg(normalize_kernel, 1, sizeof(cl_mem), &lut);
clSetKernelArg(normalize_kernel, 2, sizeof(unsigned int), &num_bins);
clSetKernelArg(normalize_kernel, 3, sizeof(unsigned int), &max_val);

// enqueue kernel
size_t global_size = num_bins;
size_t local_size = num_bins; // or use a smaller local size if memory is limited
clEnqueueNDRangeKernel(queue, normalize_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
// wait for kernel to finish
clFinish(queue);

//Step 8: Back-project the input image intensities onto the output image using the LUT.
// set kernel arguments
clSetKernelArg(backproject_kernel, 0, sizeof(cl_mem), &input_image);
clSetKernelArg(backproject_kernel, 1, sizeof(cl_mem), &output_image);
clSetKernelArg(backproject_kernel, 2, sizeof(cl_mem), &lut);
clSetKernelArg(backproject_kernel, 3, sizeof(unsigned int), &num_pixels);

// enqueue kernel
size_t global_size = num_pixels;
size_t local_size = num_pixels; // or use a smaller local size if memory is limited
clEnqueueNDRangeKernel(queue, backproject_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
// wait for kernel to finish
clFinish(queue);

//Step 9: Read the output image from the device and save it to a file.
// allocate memory for output image
unsigned char* output_data = new unsigned char[num_pixels];

// read output image from device
clEnqueueReadBuffer(queue, output_image, CL_TRUE, 0, num_pixels * sizeof(unsigned char), output_data, 0, NULL, NULL);

// write output image to file
stbi_write_png("output.png", width, height, 1, output_data, width);

// free memory
delete[] output_data;

//Step 10: Calculate and report performance metrics such as memory transfer, kernel execution, and total program execution times.
// get profiling info for each kernel
cl_ulong input_read_time = 0, histogram_time = 0, cumulative_histogram_time = 0, normalize_time = 0, backproject_time = 0, output_write_time = 0;
clGetEventProfilingInfo(input_read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &input_read_time, NULL);
clGetEventProfilingInfo(histogram_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &histogram_time, NULL);
clGetEventProfilingInfo(cumulative_histogram_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &cumulative_histogram_time, NULL);
clGetEventProfilingInfo(normalize_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &normalize_time, NULL);
clGetEventProfilingInfo(backproject_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &backproject_time, NULL);
clGetEventProfilingInfo(output_write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &output_write_time, NULL);

// calculate execution times
double memory_transfer_time = (input_read_time - input_enqueue_time + output_write_time - output_enqueue_time) / 1e9;
double kernel_execution_time = (histogram_time - input_read_time + cumulative_histogram_time - histogram_time + normalize_time - cumulative_histogram_time + backproject_time - normalize_time) / 1e9;
double total_execution_time = (output_write_time - input_enqueue_time) / 1e9;

// print execution times
std::cout << "Memory transfer time: " << memory_transfer_time << " seconds\n";
std::cout << "Kernel execution time: " << kernel_execution_time << " seconds\n";
std::cout << "Total program execution time: " << total_execution_time << " seconds\n";

return 0;
}



