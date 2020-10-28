#include "../common/sycl_utils.hpp"
#include "msg_channel_sycl.hpp"
#include "msg_channel_host.hpp"

int main(int argc, char **argv)
{
    uint64_t msgchl_buffer_size = 128 * 1024;
    size_t n_workitem = 1024 * 1024;

    try
    {
        sycl::queue q(sycl::default_selector{});

        uint64_t *msgchl_workbuf;
        msgchl_setup(q, msgchl_buffer_size, &msgchl_workbuf);

        msg_channel_host_daemon_start(msgchl_workbuf, n_workitem);

        test_msgchl_kernel(q, msgchl_workbuf, n_workitem);
        q.wait();

        msg_channel_host_daemon_wait_end();

        int refsum = 0;
        for (size_t i = 0; i < n_workitem; i++)
        {
            int ref_value0 = i % 1924;
            int ref_value1 = i % 1112;
            refsum += ref_value0 + ref_value1;
        }
        std::cout << "correct sum = " << refsum << std::endl;

        sycl::free(msgchl_workbuf, q);
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 255;
    }

    return 0;
}