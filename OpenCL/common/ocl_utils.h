#ifndef __OCL_UTILS_H__
#define __OCL_UTILS_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get OpenCL error name
const char *cl_get_error_str(cl_int status);

#define CL_CHECK_CALL(statement)                                        \
    do {                                                                \
        cl_int err = statement;                                         \
        if (err == CL_SUCCESS) break;                                   \
        fprintf(stderr, "[%s:%d] OpenCL error: '%s' returned %s!\n",    \
                __FILE__, __LINE__, #statement, cl_get_error_str(err)); \
        exit(-1);                                                       \
    } while (0) 

#define CL_CHECK_RET(statement, err)                                    \
    do {                                                                \
        if (err == CL_SUCCESS) break;                                   \
        fprintf(stderr, "[%s:%d] OpenCL error: '%s' returned %s!\n",    \
                __FILE__, __LINE__, #statement, cl_get_error_str(err)); \
        exit(-1);                                                       \
    } while (0) 

// Get all OpenCL platform IDs
cl_int cl_get_platform_ids(int *n_platform, cl_platform_id **platform_ids);

// Choose all specified device ids on the given platform
cl_int cl_get_device_ids(
    cl_platform_id platform_id, cl_device_type device_type, 
    int *n_device, cl_device_id **device_ids
);

// Print the information of an OpenCL platform
void cl_print_platform_info(cl_platform_id platform_id);

// Print the information of an OpenCL device
void cl_print_device_info(cl_device_id device_id);

// Read kernel file and create a program with the given cl_context and cl_device_id
cl_int cl_build_program_from_file(
    const char *file_name, cl_context context, 
    cl_device_id device_id, cl_program *program
);

#ifdef __cplusplus
}
#endif

#endif
