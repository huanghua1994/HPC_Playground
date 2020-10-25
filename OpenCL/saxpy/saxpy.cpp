#include "../common/ocl_utils.h"
#include <time.h>

int main(int argc, char **argv)
{
    int vec_len = 1048576;
    if (argc >= 2) vec_len = atoi(argv[1]);
    if (vec_len < 1024) vec_len = 1024;
    printf("OpenCL SAXPY, vector length = %d\n", vec_len);
    
    // Allocate memory on host
    size_t vec_bytes = sizeof(float) * vec_len;
    float *h_x   = (float *) malloc(vec_bytes);
    float *h_y   = (float *) malloc(vec_bytes);
    float *h_ref = (float *) malloc(vec_bytes);
    float alpha  = 1.0;
    srand48(time(NULL));
    for (int i = 0; i < vec_len; i++)
    {
        h_x[i] = (float) drand48();
        h_y[i] = (float) drand48();
        h_ref[i] = h_y[i] + alpha * h_x[i];
    }
    
    cl_int status;
    // Get all OpenCL platforms and choose the first platform
    int n_platform;
    cl_platform_id *platform_ids, platform_id;
    status = cl_get_platform_ids(&n_platform, &platform_ids);
    platform_id = platform_ids[0];

    // Select the first GPU device
    int n_device;
    cl_device_id *device_ids, device_id;
    status = cl_get_device_ids(platform_id, CL_DEVICE_TYPE_GPU, &n_device, &device_ids);
    device_id = device_ids[0];

    // Build a program from the kernel file and create the kernel we need
    cl_context       context;
    cl_program       program;
    cl_kernel        kernel;
    cl_command_queue queue;
    void *user_data = NULL;
    context = clCreateContext(NULL, 1, &device_id, NULL, user_data, &status);
    CL_CHECK_RET(clCreateContext, status);
    status = cl_build_program_from_file("saxpy.cl", context, device_id, &program);
    kernel = clCreateKernel(program, "my_ocl_saxpy", &status);
    CL_CHECK_RET(clCreateKernel, status);
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &status);
    CL_CHECK_RET(clCreateCommandQueue, status);

    // Allocate memory on device and copy data to device
    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, vec_bytes, NULL, &status);
    cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_WRITE, vec_bytes, NULL, &status);
    cl_bool  blk_write = CL_FALSE;
    size_t   write_offset = 0;
    cl_uint  n_wait_event = 0;
    cl_event h2dcpy_x_event, h2dcpy_y_event;
    CL_CHECK_CALL( status = clEnqueueWriteBuffer(queue, d_x, blk_write, write_offset, vec_bytes, h_x, n_wait_event, NULL, &h2dcpy_x_event) );
    CL_CHECK_CALL( status = clEnqueueWriteBuffer(queue, d_y, blk_write, write_offset, vec_bytes, h_y, n_wait_event, NULL, &h2dcpy_y_event) );

    // Set kernel arguments and launch kernel
    CL_CHECK_CALL( status = clSetKernelArg(kernel, 0, sizeof(int),    (void *) &vec_len) );
    CL_CHECK_CALL( status = clSetKernelArg(kernel, 1, sizeof(float),  (void *) &alpha)   );
    CL_CHECK_CALL( status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_x)     );
    CL_CHECK_CALL( status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_y)     );
    cl_uint work_dim = 1;
    size_t  *global_work_offset = NULL;
    size_t  workgroup_size = 64;
    size_t  workspace_size = (vec_len + workgroup_size - 1) / workgroup_size * workgroup_size;
    n_wait_event = 2;
    cl_event wait_events[2] = {h2dcpy_x_event, h2dcpy_y_event};
    cl_event kernel_event;
    status = clEnqueueNDRangeKernel(
        queue, kernel, work_dim, global_work_offset, 
        &workspace_size, &workgroup_size, 
        n_wait_event, wait_events, &kernel_event
    );
    CL_CHECK_RET(clEnqueueNDRangeKernel, status);
    wait_events[0] = NULL;
    wait_events[1] = NULL;

    // Copy result from device to host
    blk_write = CL_TRUE;
    n_wait_event = 1;
    wait_events[0] = kernel_event;
    CL_CHECK_CALL( status = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, vec_bytes, h_y, n_wait_event, wait_events, NULL) );
    
    // Check the results
    int correct_result = 1;
    for (int i = 0; i < vec_len; i++)
    {
        if (h_ref[i] == h_y[i]) continue;
        correct_result = 0;
        printf("Index %d: expected %f, got %f\n", i, h_ref[i], h_y[i]);
        break;
    }
    if (correct_result) printf("Result is correct.\n");

    // Calculate the bandwidth
    cl_ulong time_start, time_end;
    CL_CHECK_CALL( status = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL) );
    CL_CHECK_CALL( status = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,   sizeof(time_end),   &time_end,   NULL) );
    double seconds = (double) (time_end - time_start) / 1000000000.0;
    double giga_bytes = 3.0 * (double) vec_bytes / 1024.0 / 1024.0 / 1024.0;
    double GBs = giga_bytes / seconds; 
    printf("SAXPY used %.3f sec, memory footprint = %.3f GB, bandwidth = %.3f GB/s\n", seconds, giga_bytes, GBs);

    // Free host memory
    free(h_x);
    free(h_y);
    free(h_ref);
    free(platform_ids);
    free(device_ids);
    
    // Free OpenCL resources
    status = clReleaseEvent(h2dcpy_x_event);
    status = clReleaseEvent(h2dcpy_y_event);
    status = clReleaseEvent(kernel_event);
    status = clReleaseKernel(kernel);
    status = clReleaseProgram(program);
    status = clReleaseMemObject(d_x); 
    status = clReleaseMemObject(d_y);
    status = clReleaseCommandQueue(queue);
    status = clReleaseContext(context);
    
    return 0;
}