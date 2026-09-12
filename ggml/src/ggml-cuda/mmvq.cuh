#include "common.cuh"

// TENSELERATE: raised from upstream's 8 so GGML_CUDA_MMVQ_MAX can keep batches up to 32
// on the dp4a vector path. Each width is a separate kernel instantiation (see the switch
// in mmvq.cu), so this costs compile time and binary size, and the per-thread accumulator
// float tmp[ncols_dst][rows_per_cuda_block] plus the shared tmp_shared[] both scale
// linearly in the width - expect register pressure above the widths upstream tuned for.
#define MMVQ_MAX_BATCH_SIZE 32 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// GGML_CUDA_NO_MMVQ=1: skip the dp4a vector kernels so small batches go to MMQ
bool ggml_cuda_no_mmvq();
// GGML_CUDA_MMVQ_MAX=N: largest batch (ne11) still sent to the dp4a vector kernels when
// MMQ could take the tensor; wider goes to the tensor-core MMQ path. Default is
// MMVQ_MAX_BATCH_SIZE (unchanged behaviour); GGML_CUDA_NO_MMVQ=1 is the same as 0.
// N=1 keeps single-token decode on MMVQ and routes speculative verification batches
// (2..8 columns) to MMQ, which is where the dp4a path loses on the CMP 170HX.
int ggml_cuda_mmvq_max_batch();

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
