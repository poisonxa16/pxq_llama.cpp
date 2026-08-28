#include "common.cuh"

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// PXA_FUSE_DELTANET (R2): optional redirect of the new ssm-state write straight into the
// recurrent cache row (fuses away the CONCAT state copy). nullptr = the classic behavior.
void ggml_cuda_op_delta_net_ex(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst_override);
