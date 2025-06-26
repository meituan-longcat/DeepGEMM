import torch
from typing import Tuple

from .gemm import get_best_configs
from .tuner import jit_tuner
from .utils import ceil_div, get_col_major_tma_aligned_tensor, get_num_sms

# C++ code templates
includes = ('"deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};
constexpr auto kIsSwapAB = {IS_SWAP_AB};
constexpr auto kIsWithOffset = (GemmType::{GEMM_TYPE} == GemmType::GroupedWithOffset);
constexpr auto kIsPDL = {IS_PDL};

// Make a templated grouped GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, {NUM_GROUPS}, kNumStages, kNumTMAMulticast, GemmType::{GEMM_TYPE}>;

// Launch kernel
if constexpr (kIsSwapAB) {
    auto tma_a_desc = GemmType::make_2d_tma_a_desc_swap_ab(rhs, N);
    auto tma_b_desc = GemmType::make_2d_tma_b_desc_swap_ab(lhs, m);
    auto tma_scales_b_desc = kIsWithOffset ? GemmType::make_tma_scales_b_offset_desc_swap_ab(lhs_scales, m, {NUM_GROUPS})
                                           : GemmType::make_2d_tma_scales_b_desc_swap_ab(lhs_scales, m);
    auto tma_d_desc = GemmType::make_2d_tma_d_desc_swap_ab(out, m);
    GemmType::run_swap_ab<kIsPDL>(out, rhs_scales, grouped_layout,
                          m,
                          tma_a_desc, tma_b_desc, tma_scales_b_desc, tma_d_desc,
                          stream, num_sms, smem_size);
} else {
    auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
    auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
    auto tma_scales_a_desc = kIsWithOffset ? GemmType::make_tma_scales_a_offset_desc(lhs_scales, m, {NUM_GROUPS})
                                           : GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
    auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m);
    GemmType::run(out, rhs_scales, grouped_layout,
                  m,
                  tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                  stream, num_sms, smem_size);
}
"""


def m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs: Tuple[torch.Tensor, torch.Tensor],
                                              rhs: Tuple[torch.Tensor, torch.Tensor],
                                              out: torch.Tensor, m_indices: torch.Tensor, use_pdl: bool) -> None:
    """
    Do a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.
    On the M axis, inputs are grouped into several batches, of which batch sizes aligned to
        `get_m_alignment_for_contiguous_layout()` (128).

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_indices: a tensor of shape `[m_sum]` with type `torch.int`.
            `m_indices[i]` records the group which the j-th row of the LHS belong to,
            which means that the i-th row of the LHS matrix will be multiplied with `rhs[m_indices[i]]`.
            Values of `m_indices` in every-m-alignment-block must also be the same.
            `-1` in this tensor indicates no RHS matrix selected, the kernel will skip the computation for that aligned block.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms,
                                                                                  is_grouped_contiguous=True)
    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m_indices, m, num_groups,
            torch.cuda.current_stream(), num_sms, smem_size)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast,
              'IS_PDL': 'true' if use_pdl else 'false',
              'IS_SWAP_AB': 'false', 'GEMM_TYPE': 'GroupedContiguous'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('m', int), ('num_groups', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)


def m_grouped_gemm_fp8_fp8_bf16_nt_masked(lhs: Tuple[torch.Tensor, torch.Tensor],
                                          rhs: Tuple[torch.Tensor, torch.Tensor],
                                          out: torch.Tensor, masked_m: torch.Tensor, expected_m: int, use_pdl: bool) -> None:
    """
    Do a grouped GEMM (masked format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.
    Moreover, this alignment requirement is different with the contiguous-format kernel, as we require that each batch
        should be separately transposed.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, m_max, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[num_groups, m_max, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[num_groups, m_max, n]`, representing the result.
        masked_m: a tensor of shape `[num_groups]`, `masked_m[i]` records actual rows of the `lhs[i]` matrix to compute
            in the i-th group.
        expected_m: a value hint (which is a value on CPU) for the M expectation of each batch,
            correctly setting this value may lead to better performance.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    num_groups__, m_, n_ = out.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs_scales.shape == (num_groups, m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and masked_m.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()

    swap_ab_threshold = 64
    should_swap_ab = expected_m < swap_ab_threshold

    if should_swap_ab:
        config_m, config_n = n, expected_m
    else:
        config_m, config_n = expected_m, n

    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(
        config_m, config_n, k, num_groups, num_sms, False, should_swap_ab
    )

    # Extra checks for TMA store
    if num_groups > 1:
        if should_swap_ab:
            assert m >= block_n, f'For masked grouped GEMM with swap_ab, shape M should be >= block N (current M: {m}, block N: {block_n})'
        else:
            assert m <= block_m or m % block_m == 0, f'For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})'

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            masked_m, m,
            torch.cuda.current_stream(), num_sms, smem_size)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast,
              'IS_SWAP_AB': 'true' if should_swap_ab else 'false',
              'IS_PDL': 'true' if use_pdl else 'false',
              'GEMM_TYPE': 'GroupedMasked'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)


def m_grouped_gemm_fp8_fp8_bf16_nt_offset(lhs: Tuple[torch.Tensor, torch.Tensor],
                                          rhs: Tuple[torch.Tensor, torch.Tensor],
                                          out: torch.Tensor, m_offset: torch.Tensor, use_pdl: bool) -> None:
    """
    Do a grouped GEMM (offset format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.
    On the M axis, inputs are grouped using offset indices to specify the start and end positions of each group
        within the flattened LHS tensor.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_offset: a tensor of shape `[num_groups + 1] with type torch.int`.
            `m_offset[i]` records the starting row index of the i-th group in the flattened LHS tensor,
            and `m_offset[i+1] - m_offset[i]` gives the number of rows for the i-th group.
            The last element `m_offset[num_groups]` should equal `m_sum`.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_offset.numel()

    # Type and shape checks
    assert m == m_ and (m__ - 1) == num_groups and k == k_ and n == n_
    # assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_offset.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_offset.is_contiguous()

    # FIXME: check lhs_scales shape
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()

    max_shape_m_4_align = ceil_div(m, 4) * 4
    expected_m = ceil_div(max_shape_m_4_align, num_groups)

    swap_ab_threshold = 64
    should_swap_ab = expected_m < swap_ab_threshold

    if should_swap_ab:
        config_m, config_n = n, expected_m
    else:
        config_m, config_n = expected_m, n

    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(
        config_m, config_n, k, num_groups, num_sms, False, should_swap_ab
    )

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m_offset, m,
            torch.cuda.current_stream(), num_sms, smem_size)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast,
              'IS_SWAP_AB': 'true' if should_swap_ab else 'false',
              'IS_PDL': 'true' if use_pdl else 'false',
              'GEMM_TYPE': 'GroupedWithOffset'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
