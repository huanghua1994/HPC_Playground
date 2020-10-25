#include "ocl_utils.h"

const char *cl_get_error_str(cl_int status)
{
    const char *err_str;
    switch (status)
    {
        case   0: err_str = "CL_SUCCESS";                                   break;
        case  -1: err_str = "CL_DEVICE_NOT_FOUND";                          break;
        case  -2: err_str = "CL_DEVICE_NOT_AVAILABLE";                      break;
        case  -3: err_str = "CL_COMPILER_NOT_AVAILABLE";                    break;
        case  -4: err_str = "CL_MEM_OBJECT_ALLOCATION_FAILURE";             break;
        case  -5: err_str = "CL_OUT_OF_RESOURCES";                          break;
        case  -6: err_str = "CL_OUT_OF_HOST_MEMORY";                        break;
        case  -7: err_str = "CL_PROFILING_INFO_NOT_AVAILABLE";              break;
        case  -8: err_str = "CL_MEM_COPY_OVERLAP";                          break;
        case  -9: err_str = "CL_IMAGE_FORMAT_MISMATCH";                     break;
        case -10: err_str = "CL_IMAGE_FORMAT_NOT_SUPPORTED";                break;
        case -11: err_str = "CL_BUILD_PROGRAM_FAILURE";                     break;
        case -12: err_str = "CL_MAP_FAILURE";                               break;
        case -13: err_str = "CL_MISALIGNED_SUB_BUFFER_OFFSET";              break;
        case -14: err_str = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
        case -15: err_str = "CL_COMPILE_PROGRAM_FAILURE";                   break;
        case -16: err_str = "CL_LINKER_NOT_AVAILABLE";                      break;
        case -17: err_str = "CL_LINK_PROGRAM_FAILURE";                      break;
        case -18: err_str = "CL_DEVICE_PARTITION_FAILED";                   break;
        case -19: err_str = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";             break;
        case -30: err_str = "CL_INVALID_VALUE";                             break;
        case -31: err_str = "CL_INVALID_DEVICE_TYPE";                       break;
        case -32: err_str = "CL_INVALID_PLATFORM";                          break;
        case -33: err_str = "CL_INVALID_DEVICE";                            break;
        case -34: err_str = "CL_INVALID_CONTEXT";                           break;
        case -35: err_str = "CL_INVALID_QUEUE_PROPERTIES";                  break;
        case -36: err_str = "CL_INVALID_COMMAND_QUEUE";                     break;
        case -37: err_str = "CL_INVALID_HOST_PTR";                          break;
        case -38: err_str = "CL_INVALID_MEM_OBJECT";                        break;
        case -39: err_str = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";           break;
        case -40: err_str = "CL_INVALID_IMAGE_SIZE";                        break;
        case -41: err_str = "CL_INVALID_SAMPLER";                           break;
        case -42: err_str = "CL_INVALID_BINARY";                            break;
        case -43: err_str = "CL_INVALID_BUILD_OPTIONS";                     break;
        case -44: err_str = "CL_INVALID_PROGRAM";                           break;
        case -45: err_str = "CL_INVALID_PROGRAM_EXECUTABLE";                break;
        case -46: err_str = "CL_INVALID_KERNEL_NAME";                       break;
        case -47: err_str = "CL_INVALID_KERNEL_DEFINITION";                 break;
        case -48: err_str = "CL_INVALID_KERNEL";                            break;
        case -49: err_str = "CL_INVALID_ARG_INDEX";                         break;
        case -50: err_str = "CL_INVALID_ARG_VALUE";                         break;
        case -51: err_str = "CL_INVALID_ARG_SIZE";                          break;
        case -52: err_str = "CL_INVALID_KERNEL_ARGS";                       break;
        case -53: err_str = "CL_INVALID_WORK_DIMENSION";                    break;
        case -54: err_str = "CL_INVALID_WORK_GROUP_SIZE";                   break;
        case -55: err_str = "CL_INVALID_WORK_ITEM_SIZE";                    break;
        case -56: err_str = "CL_INVALID_GLOBAL_OFFSET";                     break;
        case -57: err_str = "CL_INVALID_EVENT_WAIT_LIST";                   break;
        case -58: err_str = "CL_INVALID_EVENT";                             break;
        case -59: err_str = "CL_INVALID_OPERATION";                         break;
        case -60: err_str = "CL_INVALID_GL_OBJECT";                         break;
        case -61: err_str = "CL_INVALID_BUFFER_SIZE";                       break;
        case -62: err_str = "CL_INVALID_MIP_LEVEL";                         break;
        case -63: err_str = "CL_INVALID_GLOBAL_WORK_SIZE";                  break;
        case -64: err_str = "CL_INVALID_PROPERTY";                          break;
        case -65: err_str = "CL_INVALID_IMAGE_DESCRIPTOR";                  break;
        case -66: err_str = "CL_INVALID_COMPILER_OPTIONS";                  break;
        case -67: err_str = "CL_INVALID_LINKER_OPTIONS";                    break;
        case -68: err_str = "CL_INVALID_DEVICE_PARTITION_COUNT";            break;
        case -69: err_str = "CL_INVALID_PIPE_SIZE";                         break;
        case -70: err_str = "CL_INVALID_DEVICE_QUEUE";                      break;
        default:  err_str = "UNKNOWN_RETURN_CODE";                          break;
    }
    return err_str;
}

// Get all OpenCL platform IDs
cl_int cl_get_platform_ids(int *n_platform, cl_platform_id **platform_ids)
{
    cl_int status;
    CL_CHECK_CALL( status = clGetPlatformIDs(0, NULL, n_platform) );
    if ((*n_platform) == 0)
    {
        fprintf(stderr, "%s, %d: No available OpenCL platform\n", __FILE__, __LINE__);
        *n_platform = 0;
        *platform_ids = NULL;
        return 255;
    }
    *platform_ids = (cl_platform_id *) malloc((*n_platform) * sizeof(cl_platform_id));
    CL_CHECK_CALL( status = clGetPlatformIDs((*n_platform), *platform_ids, NULL) );
    return status;
}

// Choose all specified device ids on the given platform
cl_int cl_get_device_ids(
    cl_platform_id platform_id, cl_device_type device_type, 
    int *n_device, cl_device_id **device_ids
)
{
    cl_int status;
    CL_CHECK_CALL( status = clGetDeviceIDs(platform_id, device_type, 0, NULL, n_device) );
    if ((*n_device) == 0)
    {
        fprintf(stderr, "%s, %d: No available GPU device on the given OpenCL platform\n", __FILE__, __LINE__);
        *n_device = 0;
        *device_ids = NULL;
        return 255;
    }
    *device_ids = (cl_device_id *) malloc(sizeof(cl_device_id) * (*n_device));
    CL_CHECK_CALL( status = clGetDeviceIDs(platform_id, device_type, (*n_device), *device_ids, NULL) );
    return status;
}

// Print the information of an OpenCL platform
void cl_print_platform_info(cl_platform_id platform_id)
{
    const int  n_attr = 5;
    const char *attr_names[5] = {
        "Name", "Vendor", "Version", 
        "Profile", "Extensions"
    };
    const cl_platform_info attr_types[5] = {
        CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, 
        CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS
    };
    printf("Platform information: \n");
    for (int i = 0; i < n_attr; i++)
    {
        size_t info_size;
        clGetPlatformInfo(platform_id, attr_types[i], 0, NULL, &info_size);
        char *info_str = (char *) malloc(info_size);
        clGetPlatformInfo(platform_id, attr_types[i], info_size, info_str, NULL);
        printf("    %-11s: %s\n", attr_names[i], info_str);
        free(info_str);
    }
}

// Print the information of an OpenCL device
void cl_print_device_info(const cl_device_id device_id)
{
    const int  n_attr = 6;
    const char *attr_names[6] = {
        "Name", "Device version", 
        "Driver version", "OpenCL C version", 
        "Memory size", "Computing units"
    };
    const cl_device_info attr_types[6] = {
        CL_DEVICE_NAME, CL_DEVICE_VERSION, 
        CL_DRIVER_VERSION, CL_DEVICE_OPENCL_C_VERSION, 
        CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_COMPUTE_UNITS
    };
    printf("Device information: \n");
    for (int i = 0; i < n_attr; i++)
    {
        size_t info_size;
        clGetDeviceInfo(device_id, attr_types[i], 0, NULL, &info_size);
        char *info_str = (char *) malloc(info_size);
        clGetDeviceInfo(device_id, attr_types[i], info_size, info_str, NULL);
        printf("    %-17s: %s\n", attr_names[i], info_str);
        free(info_str);
    }
}

// Read kernel file and create a program with the given cl_context and cl_device_id
cl_int cl_build_program_from_file(
    const char *file_name, cl_context context, 
    cl_device_id device_id, cl_program *program
)
{
    FILE *inf = fopen(file_name, "r");
    if (inf == NULL) 
    {
        fprintf(stderr, "%s, %d: Error opening kernel file %s\n", __FILE__, __LINE__, file_name);
        *program = NULL;
        return 255;
    }

    fseek(inf, 0, SEEK_END);
    long size = ftell(inf);
    size_t file_size = (size_t) (size + 1);
    rewind(inf);

    char *file_content = (char *) malloc(sizeof(char) * (size + 1));
    size_t read_source_size = fread(file_content, 1, sizeof(char) * size, inf);
    assert(file_size == read_source_size + 1);
    file_content[size] = '\0';

    cl_int status;
    const char **file_content_ = (const char **) &file_content;
    const size_t *file_size_ = &file_size;
    *program = clCreateProgramWithSource(context, 1, file_content_, file_size_, &status);
    CL_CHECK_RET(clCreateProgramWithSource, status);
    
    const char *options = NULL;
    void *user_data = NULL;
    CL_CHECK_CALL( status = clBuildProgram(*program, 1, &device_id, options, NULL, user_data) );
    free(file_content);
    return status;
}