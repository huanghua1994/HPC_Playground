#include "msg_channel_sycl.hpp"

inline void wait_until_ge_val(uint64_t *addr, const uint64_t val)
{
    sycl::atomic<uint64_t> atomic_op{ sycl::global_ptr<uint64_t>{addr} };
    uint64_t val_at_addr = atomic_op.load();
    while (val_at_addr < val) val_at_addr = atomic_op.load();
}

inline void msgchl_wait_buf_available(uint64_t *msgchl_workbuf, const uint64_t buffer_size, const uint64_t issue_idx)
{
    uint64_t wait_val = issue_idx - (buffer_size - 1);
    uint64_t *process_idx_p = msgchl_workbuf + MSGCHL_PROCESS_IDX_IDX;
    sycl::atomic<uint64_t> atomic_op{ sycl::global_ptr<uint64_t>{process_idx_p} };
    uint64_t process_idx = atomic_op.load();
    if (process_idx + buffer_size - 1 < issue_idx)
        wait_until_ge_val(process_idx_p, wait_val);
    // In the CU-PGAS code here should have a __threadfence_system(), it's copied from NVSHMEM
    // But I don't know why NVSHMEM has this and what is the counterpart in SYCL
}

void msgchl_issue_2int_request(uint64_t *msgchl_workbuf, const int i, const int value0, const int value1)
{
    
    // In the CU-PGAS code here should have a __threadfence(), it's copied from NVSHMEM
    // But I don't know why NVSHMEM has this and what is the counterpart in SYCL
    uint64_t buffer_size      = msgchl_workbuf[MSGCHL_BUFFER_SIZE_IDX];
    uint64_t buffer_size_log2 = msgchl_workbuf[MSGCHL_BUFFER_SIZE_LOG2_IDX];
    uint64_t *issue_idx_p = msgchl_workbuf + MSGCHL_ISSUE_IDX_IDX;
    sycl::atomic<uint64_t> atomic_op{ sycl::global_ptr<uint64_t>{issue_idx_p} };
    uint64_t issue_idx  = atomic_op.fetch_add(1);
    uint64_t buffer_idx = MSGCHL_WRAPPED_INDEX(buffer_size, issue_idx);
    uint64_t issue_flag = MSGCHL_CYCLIC_FLAG(buffer_size_log2, issue_idx);
    msgchl_wait_buf_available(msgchl_workbuf, buffer_size, issue_idx);

    // Assemble message payloads and write to buffer
    uint64_t *buffer_ptr = &msgchl_workbuf[MSGCHL_BUFFER_START_IDX] + buffer_idx * MSG_PAYLOAD_LEN;

    // msg_payload_unit_0
    // high ---------------> low
    //      32 |       24 |    8
    //  value0 | reserved | flag
    // Do we need atomic store here, or can we use *((volatile uint64_t *) buffer_ptr) = xxx?
    {
        uint64_t unit0 = static_cast<uint64_t>(value0);
        unit0 = unit0 << 32;
        unit0 = unit0 | issue_flag;
        sycl::atomic<uint64_t> atomic_unit0{ sycl::global_ptr<uint64_t>{buffer_ptr} };
        atomic_unit0.store(unit0);
    }

    // msg_payload_unit_1
    // high ---------------> low
    //      32 |       24 |    8
    //  value1 | reserved | flag
    // Do we need atomic store here, or can we use *((volatile uint64_t *) buffer_ptr) = xxx?
    buffer_ptr++;
    {
        uint64_t unit1 = static_cast<uint64_t>(value1);
        unit1 = unit1 << 32;
        unit1 = unit1 | issue_flag;
        sycl::atomic<uint64_t> atomic_unit1{ sycl::global_ptr<uint64_t>{buffer_ptr} };
        atomic_unit1.store(unit1);
    }
}

void test_msgchl_kernel(sycl::queue &q, uint64_t *msgchl_workbuf, const size_t n_workitem)
{
    size_t local_size = 64;
    sycl::range global_range {n_workitem};
    sycl::range local_range  {local_size};

    q.submit( [&](sycl::handler &h) {
        sycl::stream dbgout(1024, 128, h);

        h.parallel_for<class test_msgchl> (
        sycl::nd_range{global_range, local_range}, [=](sycl::nd_item<1> it)
        {
            int i = it.get_global_id(0);
            int value0 = i % 1924;
            int value1 = i % 1112;

            msgchl_issue_2int_request(msgchl_workbuf, i, value0, value1);
        });
    });
}

void msgchl_setup(sycl::queue &q, uint64_t msgchl_buffer_size, uint64_t **msgchl_workbuf_)
{
    uint64_t msgchl_buffer_size_log2 = 0, tmp = 1;
    while (tmp < msgchl_buffer_size)
    {
        tmp *= 2;
        msgchl_buffer_size_log2++;
    }
    if (tmp != msgchl_buffer_size)
    {
        fprintf(stderr, "[ERROR] msgchl_buffer_size must be a power of 2!\n");
        *msgchl_workbuf_ = NULL;
        return;
    }

    size_t msgchl_workbuf_size = static_cast<size_t>(MSGCHL_BUFFER_START_IDX + msgchl_buffer_size);
    uint64_t *msgchl_workbuf = static_cast<uint64_t *>( sycl::malloc_host(sizeof(msg_payload_t) * msgchl_workbuf_size, q) );
    msgchl_workbuf[MSGCHL_BUFFER_SIZE_IDX]      = msgchl_buffer_size;
    msgchl_workbuf[MSGCHL_BUFFER_SIZE_LOG2_IDX] = msgchl_buffer_size_log2;
    msgchl_workbuf[MSGCHL_ISSUE_IDX_IDX]        = 0;
    msgchl_workbuf[MSGCHL_PROCESS_IDX_IDX]      = 0;
    
    // Initially correct payload flag is 1, fill it with 0 to mark all payload units are not ready
    uint64_t *msgchl_buffer = msgchl_workbuf + MSGCHL_BUFFER_START_IDX;
    memset(msgchl_buffer, 0, sizeof(uint64_t) * msgchl_buffer_size);

    *msgchl_workbuf_ = msgchl_workbuf;
}