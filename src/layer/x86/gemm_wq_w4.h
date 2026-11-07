// FP32-activation W4 N-vectorized dequant/FMA kernel.
// Same signed two's-complement nibbles and inverse block scales as Gemm.
#if NCNN_WEIGHT_QUANT && __AVX512F__ && __AVX512BW__

static int gemm_wq_w4_direct(const Mat& A, const Mat& packed_indices, const Mat& packed_descales,
                             const Mat& input_scales, const Mat& C, Mat& output,
                             int broadcast_type_C, int M, int N, int K, int block_size,
                             float alpha, float beta, const Option& opt)
{
    const int tiles = (int)(((size_t)N + 15) / 16);
    const int blocks = (int)(((size_t)K + block_size - 1) / block_size);
    const size_t a_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const size_t out_hstep = output.dims == 3 ? output.cstep : (size_t)output.w;

    // Apply optional per-input-feature scaling before processing output tiles.
    Mat scaled;
    if (!input_scales.empty())
    {
        if ((size_t)M > ((size_t)-1) / ((size_t)K * sizeof(float)))
            return -100;
        scaled.create(K, M, (size_t)4u, opt.workspace_allocator);
        if (scaled.empty())
            return -100;
        const float* scale = input_scales;
        for (int m = 0; m < M; m++)
        {
            const float* a = (const float*)A + (size_t)m * a_hstep;
            float* dst = scaled.row(m);
            for (int k = 0; k < K; k++)
                dst[k] = a[k] * scale[k];
        }
    }

    // Each job computes one input row and up to 16 output features.
    const int64_t jobs = (int64_t)M * tiles;
    int num_threads = (int)std::min((int64_t)std::max(opt.num_threads, 1), jobs);
    // Avoid a parallel region for truly tiny work; this is a conservative gate,
    // not a universal autotuned crossover or an OpenMP wait-policy workaround.
    if ((double)M * N * K < 32768.)
        num_threads = 1;

    #pragma omp parallel for num_threads(num_threads) if(num_threads > 1)
    for (int64_t job = 0; job < jobs; job++)
    {
        const int m = (int)(job / tiles);
        const int tile = (int)(job % tiles);

        // Each packed row holds all K weights for up to 16 output columns.
        const unsigned char* indices = packed_indices.row<const unsigned char>(tile);
        const float* scales = packed_descales.row(tile);
        const float* a = scaled.empty() ? (const float*)A + (size_t)m * a_hstep : scaled.row(m);

        // Each lane accumulates one output across all quantization blocks.
        __m512 result = _mm512_setzero_ps();
        for (int b = 0; b < blocks; b++)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            const int end = (int)std::min((size_t)K, ((size_t)b + 1) * block_size);
            int k = b * block_size;
            for (; k + 1 < end; k += 2)
            {
                // Load 16 bytes: two adjacent K weights for each output lane.
                const __m512i packed = _mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i*)(indices + (size_t)(k / 2) * 16)));

                // Extract the low and high nibbles, then restore their signs.
                __m512i q0 = _mm512_and_si512(packed, _mm512_set1_epi32(15));
                __m512i q1 = _mm512_srli_epi32(packed, 4);
                q0 = _mm512_sub_epi32(
                    _mm512_xor_si512(q0, _mm512_set1_epi32(8)), _mm512_set1_epi32(8));
                q1 = _mm512_sub_epi32(
                    _mm512_xor_si512(q1, _mm512_set1_epi32(8)), _mm512_set1_epi32(8));

                // Accumulate the two K positions separately in FP32.
                acc0 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(q0), _mm512_set1_ps(a[k]), acc0);
                acc1 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(q1), _mm512_set1_ps(a[k + 1]), acc1);
            }

            // Handle the final unpaired weight when K is odd.
            if (k < end)
            {
                __m512i q = _mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i*)(indices + (size_t)(k / 2) * 16)));
                q = _mm512_and_si512(q, _mm512_set1_epi32(15));
                q = _mm512_sub_epi32(_mm512_xor_si512(q, _mm512_set1_epi32(8)), _mm512_set1_epi32(8));
                acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q), _mm512_set1_ps(a[k]), acc0);
            }

            // Apply each output lane's block scale and add to the running result.
            result = _mm512_fmadd_ps(
                _mm512_add_ps(acc0, acc1),
                _mm512_loadu_ps(scales + (size_t)b * 16), result);
        }

        // Only valid output lanes are loaded or stored in the final tile.
        const int n = tile * 16;
        const int count = std::min(16, N - n);
        const __mmask16 active = (__mmask16)((1u << count) - 1u);

        // Preserve the GEMM epilogue: alpha * (dot + beta * C).
        if (!C.empty())
        {
            const float* c = C;
            __m512 bias = _mm512_setzero_ps();
            if (broadcast_type_C == 0)
                bias = _mm512_set1_ps(c[0]);
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                bias = _mm512_set1_ps(c[m]);
            if (broadcast_type_C == 3)
                bias = _mm512_maskz_loadu_ps(active, c + (size_t)m * N + n);
            if (broadcast_type_C == 4)
                bias = _mm512_maskz_loadu_ps(active, c + n);
            result = _mm512_add_ps(result, _mm512_mul_ps(_mm512_set1_ps(beta), bias));
        }
        result = _mm512_mul_ps(result, _mm512_set1_ps(alpha));
        _mm512_mask_storeu_ps((float*)output + (size_t)m * out_hstep + n, active, result);
    }
    return 0;
}

#endif // NCNN_WEIGHT_QUANT && __AVX512F__ && __AVX512BW__
