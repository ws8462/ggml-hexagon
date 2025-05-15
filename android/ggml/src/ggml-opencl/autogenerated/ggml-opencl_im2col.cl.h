R"(#ifdef cl_khr_fp16
)"
R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(#elif defined(cl_amd_fp16)
)"
R"(#pragma OPENCL EXTENSION cl_amd_fp16 : enable
)"
R"(#else
)"
R"(#error "Half precision floating point not supportedby OpenCL implementation on your device."
)"
R"(#endif
)"
R"(
)"
R"(#ifdef cl_khr_subgroups
)"
R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
R"(#elif defined(cl_intel_subgroups)
)"
R"(#pragma OPENCL EXTENSION cl_intel_subgroups : enable
)"
R"(#else
)"
R"(#error "Subgroup not supported on your device."
)"
R"(#endif
)"
R"(
)"
R"(#ifdef cl_intel_required_subgroup_size
)"
R"(// Always use subgroup size of 32 on Intel.
)"
R"(#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
)"
R"(#define INTEL_GPU 1
)"
R"(#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
)"
R"(#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
)"
R"(#elif defined(cl_qcom_reqd_sub_group_size)
)"
R"(// Always use subgroups size of 64 on Adreno.
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(#define ADRENO_GPU 1
)"
R"(#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
)"
R"(#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
)"
R"(#else
)"
R"(// TODO: do not know how to choose subgroup size on other GPUs.
)"
R"(#error "Selecting subgroup size is not supported on your device."
)"
R"(#endif
)"
R"(
)"
R"(kernel void kernel_im2col_f32(
)"
R"(        global float * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        ulong batch_offset,
)"
R"(        ulong delta_offset,
)"
R"(        long IW,
)"
R"(        long IH,
)"
R"(        long IC,
)"
R"(        long OW,
)"
R"(        long OH,
)"
R"(        long KW,
)"
R"(        long KH,
)"
R"(        long pelements,
)"
R"(        long CHW,
)"
R"(        int  s0,
)"
R"(        int  s1,
)"
R"(        int  p0,
)"
R"(        int  p1,
)"
R"(        int  d0,
)"
R"(        int  d1
)"
R"() {
)"
R"(    // threadIdx.x + blockIdx.x * blockDim.x
)"
R"(    long i = get_global_id(0);
)"
R"(    if (i >= pelements) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    src1 = (global float*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    long  ksize = OW * (KH > 1 ? KW : 1);
)"
R"(    long  kx = i / ksize;
)"
R"(    long  kd = kx * ksize;
)"
R"(    long  ky = (i - kd) / OW;
)"
R"(    long  ix = i % OW;
)"
R"(
)"
R"(    long  oh = get_group_id(1);
)"
R"(    long  batch = get_group_id(2) / IC;
)"
R"(    long  ic = get_group_id(2) % IC;
)"
R"(
)"
R"(    long iiw = ix * s0 + kx * d0 - p0;
)"
R"(    long iih = oh * s1 + ky * d1 - p1;
)"
R"(
)"
R"(    long offset_dst =
)"
R"(        ((batch * OH + oh) * OW + ix) * CHW +
)"
R"(        (ic * (KW * KH) + ky * KW + kx);
)"
R"(
)"
R"(    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
)"
R"(        dst[offset_dst] = 0.0f;
)"
R"(    } else {
)"
R"(        long offset_src = ic * delta_offset + batch * batch_offset;
)"
R"(        dst[offset_dst] = src1[offset_src + iih * IW + iiw];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_im2col_f16(
)"
R"(        global float * src1,
)"
R"(        ulong offset1,
)"
R"(        global half  * dst,
)"
R"(        ulong offsetd,
)"
R"(        ulong batch_offset,
)"
R"(        ulong delta_offset,
)"
R"(        long IW,
)"
R"(        long IH,
)"
R"(        long IC,
)"
R"(        long OW,
)"
R"(        long OH,
)"
R"(        long KW,
)"
R"(        long KH,
)"
R"(        long pelements,
)"
R"(        long CHW,
)"
R"(        int  s0,
)"
R"(        int  s1,
)"
R"(        int  p0,
)"
R"(        int  p1,
)"
R"(        int  d0,
)"
R"(        int  d1
)"
R"() {
)"
R"(    long i = get_global_id(0);
)"
R"(
)"
R"(    if (i >= pelements) {
)"
R"(        return;
)"
R"(    }
)"
R"(
)"
R"(    src1 = (global float*)((global char*)src1 + offset1);
)"
R"(    dst = (global half*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    long  ksize = OW * (KH > 1 ? KW : 1);
)"
R"(    long  kx = i / ksize;
)"
R"(    long  kd = kx * ksize;
)"
R"(    long  ky = (i - kd) / OW;
)"
R"(    long  ix = i % OW;
)"
R"(
)"
R"(    long  oh = get_group_id(1);
)"
R"(    long  batch = get_group_id(2) / IC;
)"
R"(    long  ic = get_group_id(2) % IC;
)"
R"(
)"
R"(    long iiw = ix * s0 + kx * d0 - p0;
)"
R"(    long iih = oh * s1 + ky * d1 - p1;
)"
R"(
)"
R"(    long offset_dst =
)"
R"(        ((batch * OH + oh) * OW + ix) * CHW +
)"
R"(        (ic * (KW * KH) + ky * KW + kx);
)"
R"(
)"
R"(    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
)"
R"(        dst[offset_dst] = 0.0f;
)"
R"(    } else {
)"
R"(        long offset_src = ic * delta_offset + batch * batch_offset;
)"
R"(        dst[offset_dst] = src1[offset_src + iih * IW + iiw];
)"
R"(    }
)"
R"(}
)"
