#include "../common/ocl_utils.h"

int main(int argc, char **argv)
{
    cl_int status;

    cl_uint n_platform, n_device;
    cl_platform_id *platform_ids;
    cl_device_id *device_ids;

    status = cl_get_platform_ids(&n_platform, &platform_ids);
    printf("OpenCL available platforms: %u\n", n_platform);

    for (cl_uint i = 0; i < n_platform; i++)
    {
        printf("========== OpenCL platform %u ==========\n", i);
        cl_print_platform_info(platform_ids[i]);

        printf("---------- CPU devices ----------\n");
        status = cl_get_device_ids(platform_ids[i], CL_DEVICE_TYPE_CPU, &n_device, &device_ids);
        for (cl_uint j = 0; j < n_device; j++)
            cl_print_device_info(device_ids[j]);
        if (n_device > 0) free(device_ids);

        printf("---------- GPU devices ----------\n");
        status = cl_get_device_ids(platform_ids[i], CL_DEVICE_TYPE_GPU, &n_device, &device_ids);
        for (cl_uint j = 0; j < n_device; j++)
            cl_print_device_info(device_ids[j]);
        if (n_device > 0) free(device_ids);

        printf("---------- Accelerator devices ----------\n");
        status = cl_get_device_ids(platform_ids[i], CL_DEVICE_TYPE_ACCELERATOR, &n_device, &device_ids);
        for (cl_uint j = 0; j < n_device; j++)
            cl_print_device_info(device_ids[j]);
        if (n_device > 0) free(device_ids);
    }
    if (n_platform > 0) free(platform_ids);

    return 0;
}