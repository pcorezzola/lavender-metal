/*
 * lava5: the A5xx (Adreno 512) compute dispatch as a PM4 stream, and the flat description of a compiled kernel it
 * needs. The order of the packets is Mesa's gallium freedreno driver (a5xx/fd5_compute.c, fd5_emit.c, ir3_const.h; MIT,
 * Copyright Rob Clark and the Mesa contributors); the register encoders are the headers Mesa generates (a5xx/).
 * tools/a5cs/a5cs_gen.c is the same stream for one fixed kernel, and was verified on the phone (256 of 256 words).
 */
#ifndef LAVA5_H
#define LAVA5_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LAV5_MAX_BUFFERS 31
#define LAV5_MAX_RANGES  32

struct lav5_range {
	uint32_t block;         /* Metal's [[buffer(n)]] the range is read from */
	uint32_t offset_vec4;   /* where in the constant file it lands */
	uint32_t start_dword;   /* first dword of the buffer that is pushed */
	uint32_t size_dwords;   /* a multiple of 4 */
};

/* what a compiled kernel needs to be dispatched (lavcc.cpp fills it from hakios's Kernel) */
struct lav5_kernel {
	uint32_t * words;       /* the machine code, nwords dwords */
	uint32_t nwords;
	uint32_t instrlen, constlen;
	int32_t max_reg, max_half_reg;
	uint32_t double_threadsize;
	uint32_t branchstack;
	uint32_t localid_regid, wgid_regid;
	uint32_t * imm;         /* constants the kernel carries */
	uint32_t nimm;          /* dwords */
	uint32_t imm_vec4;
	int32_t dp_vec4;        /* driver params (ir3_driver_params_cs), -1 when the kernel reads none */
	uint32_t num_ubos, num_ssbos, num_ibos;
	uint32_t ssbo_undeclared;
	uint32_t shared_size, pvtmem_size;
	uint32_t const_data_size;
	uint32_t nranges;
	struct lav5_range ranges[LAV5_MAX_RANGES];
	uint32_t buffer_mask;   /* bit n: the kernel has a resource at [[buffer(n)]] */
	uint32_t other_resources;   /* a texture, an image, a sampler or a threadgroup argument: not served yet */
	char name[64];
};

/* a buffer bound at one Metal index: where the GPU sees it, where the processor sees it (for the constants it pushes) */
struct lav5_binding {
	uint64_t gpu;
	const void * cpu;
	uint64_t bytes;         /* what is bound, past its offset */
};

struct lav5_dispatch {
	const struct lav5_kernel * k;
	uint64_t code_gpu;                              /* the code, in memory */
	struct lav5_binding buf[LAV5_MAX_BUFFERS];
	uint32_t groups[3], local[3];
	uint64_t flush_gpu;                             /* four bytes the cache flush writes its timestamp to */
	uint64_t hole_gpu;                              /* memory nobody names, for an SSBO slot nothing is bound at */
};

/* ---- the 3D side (lava5r.c): a vertex and a fragment stage, their link, and one pass of draws ---- */

#define LAV5_MAX_IO 34

struct lav5_io {
	uint8_t slot, regid, compmask, inloc, sysval, bary, flat, half;
};

struct lav5_stage {
	struct lav5_kernel k;       /* the code, constants, footprints (the compute-only fields are zero) */
	uint32_t is_fragment;
	uint32_t nin;
	struct lav5_io in[LAV5_MAX_IO];
	uint32_t nout;
	struct { uint8_t slot, regid, half, pad; } out[32];
	/* vertex */
	uint32_t nattr;             /* inputs that are not system values */
	uint32_t vertex_id_regid, instance_id_regid, position_regid, point_size_regid;
	uint32_t clip_mask, cull_mask;
	/* fragment */
	uint32_t total_in, frag_coord_regid, face_regid;
	uint32_t frag_coord_compmask;
	uint32_t ij_regid[7];       /* ir3's order: persp pixel, sample, centroid, centre rhw, linear pixel, centroid, sample */
	uint32_t color_regid[8];
	uint32_t color_half[8];
	uint32_t ncolor;
	uint32_t depth_regid;
	uint32_t need_pixlod, has_kill, writes_pos;
	uint8_t fb_target[32];      /* per texture unit: n + 1 when the stage reads colour attachment n there (framebuffer fetch), else 0 */
	uint32_t ntexslots;         /* texture units the stage uses (the fetch units included) */
};

struct lav5_link {
	struct { uint8_t slot, regid, compmask, loc; } var[32];
	uint32_t cnt, max_loc, varmask[4], position_loc, interp[8], psize_loc;
};

/* a vertex attribute: Metal's [[attribute(index)]] read from vertex buffer `buffer` at `offset` as `format` (an a5xx
 * a5xx_vtx_fmt) */
struct lav5_attr {
	uint32_t index, format, offset, buffer;
	uint32_t is_float, swap;
};

struct lav5_vbuf {
	uint64_t gpu, bytes;
	uint32_t stride;
};

/* a texture bound for the fragment stage: linear, 2D (or 1D); an a5xx TFMT5_* format with `nch` channels in memory */
struct lav5_tex {
	uint64_t gpu;
	uint32_t width, height, pitch;
	uint32_t bgra;                              /* 1: B,G,R,A in memory (bgra8Unorm), 0: R,G,B,A */
	uint32_t valid;
	uint32_t tfmt;                              /* TFMT5_*; 0 (TFMT5_NONE... see below) means 8888 as before */
	uint32_t nch;                               /* channels the format holds (1..4); the others read 0, and w reads 1 */
	uint32_t alpha_only;                        /* A8: the identity swizzle already reads (0, 0, 0, a) */
	uint32_t is1d;
	uint32_t srgb;
};

/* a sampler: filters 0 nearest / 1 linear; wrap 0 repeat / 1 clamp to edge / 2 mirror repeat */
struct lav5_samp {
	uint32_t mag, min, mip_linear, wrap_s, wrap_t;
	uint32_t valid;
};

struct lav5_blend {
	uint32_t enable;
	uint32_t rgb_src, rgb_dst, rgb_op, a_src, a_dst, a_op;    /* a5xx values (FACTOR_*, BLEND_*) */
	uint32_t writemask;                         /* RGBA bits, 0xf default */
};

/* one more colour target of the pass (MRT): linear 8888 like the first */
struct lav5_rt {
	uint64_t gpu;
	uint32_t pitch;
	uint32_t format;                            /* 0 bgra8, 1 rgba8 */
	uint32_t valid;
};

struct lav5_render {
	const struct lav5_stage * vs, * fs;
	const struct lav5_link * link;
	uint64_t vs_code, fs_code;
	uint64_t rt_gpu;
	uint32_t rt_width, rt_height, rt_pitch;     /* pixels, pixels, bytes */
	uint32_t rt_format;                         /* 0: BGRA8 unorm (Metal's bgra8Unorm), 1: RGBA8 unorm */
	uint32_t nattr;
	struct lav5_attr attr[16];
	struct lav5_vbuf vbuf[LAV5_MAX_BUFFERS];
	struct lav5_binding vs_buf[LAV5_MAX_BUFFERS], fs_buf[LAV5_MAX_BUFFERS];
	uint32_t prim;                              /* a5xx pc_di_primtype */
	uint32_t first, count, instances, base_instance;
	uint64_t vs_pvt_gpu, fs_pvt_gpu;            /* private memory of each stage (lav5_pvtmem_bytes of it), 0 when it has none */
	uint64_t index_gpu;                         /* 0: not indexed; else the first index */
	uint32_t index_size;                        /* 1: 16-bit, 2: 32-bit (a4xx_index_size) */
	uint32_t index_max;                         /* indices from index_gpu to the end of the buffer */
	float viewport[6];                          /* x, y, width, height, znear, zfar (Metal's, top-left origin) */
	uint32_t scissor[4];                        /* x, y, width, height */
	uint32_t cull;                              /* 0 none, 1 front, 2 back */
	uint32_t front_cw;
	uint64_t flush_gpu;
	struct lav5_tex tex[16];                    /* fragment stage, by Metal index */
	struct lav5_samp samp[16];
	struct lav5_blend blend;
	float blend_color[4];
	uint64_t hole_gpu;                          /* memory nobody names, for an SSBO slot nothing is bound at */
	uint32_t nrt;                               /* colour targets of the pass; 0: just the rt_* one. rt[0] is then rt_*, rt[1..] the others */
	struct lav5_rt rt[8];
	struct lav5_blend blend_x[8];               /* blend of targets 1.. ([0] is `blend`) */
};

/* the stream of one draw into one colour target, in sysmem (bypass) mode; 0 with *why when refused */
/* why one stage cannot be drawn with, or NULL (checked again by lav5_build_render) */
const char * lav5_refuse_stage(const struct lav5_stage * s);

/* bytes of private memory the stage needs for all its fibers (0: none); a5xx has no figure for fibers per SP, so this is generous */
uint32_t lav5_pvtmem_bytes(const struct lav5_stage * s);

size_t lav5_build_render(uint32_t * out, size_t cap, const struct lav5_render * r, const char ** why);

/* the scatter copy of a texture into the physical runs of an IOSurface, by the 2D engine; and the experiment mask (/var/tmp/lavexp) */
size_t lav5_build_copy(uint32_t * words, size_t cap, uint64_t src_gpu, const uint64_t * dst_pa, const uint64_t * src_off,
    const uint32_t * len, unsigned nruns, uint64_t flush_gpu);
unsigned lav5_exp(void);
size_t lav5_build_memwrite(uint32_t * words, size_t cap, uint64_t pa, uint32_t val);

/* the words of the stream; returns how many, 0 with *why set (a string literal) when the dispatch is refused */
size_t lav5_build(uint32_t * out, size_t cap, const struct lav5_dispatch * d, const char ** why);

/* a kernel this builder can run, or the reason it cannot (before a pipeline state exists) */
const char * lav5_refuse_kernel(const struct lav5_kernel * k);

#ifdef __cplusplus
}
#endif

#endif
