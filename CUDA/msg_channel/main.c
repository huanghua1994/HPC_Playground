#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>

#include "msg_channel_host.h"
#include "test_kernel.h"

int main(int argc, char **argv)
{
    int grid_dim_x, block_dim_x;
    printf("grid_dim_x, block_dim_x = ");
    scanf("%d%d", &grid_dim_x, &block_dim_x);

    int n_thread = grid_dim_x * block_dim_x;

    const uint64_t ch_msg_buf_size = 128 * 1024;
    host_daemon_param[0] = n_thread;
    host_daemon_param[1] = 0;
    msg_channel_setup(ch_msg_buf_size);

    launch_test_kernel(grid_dim_x, block_dim_x);

    msg_channel_stop();

    double ref_result    = (double) (n_thread - 1) * (double) (n_thread) * 0.5;
    double daemon_result = *((double *) &host_daemon_param[1]);
    double result_relerr = fabs((ref_result - daemon_result) / ref_result);

    if (result_relerr < 1e-14) printf("host daemon result is correct\n");
    else printf("ref_result (%.15e) != host daemon result (%.15e)\n", ref_result, daemon_result);

    return 0;
}