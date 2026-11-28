/*
 * lava5r: one render pass of one draw on the A5xx, in sysmem (bypass) mode, as a PM4 stream. The order and the values are
 * Mesa's gallium freedreno driver (a5xx/fd5_program.c, fd5_emit.c, fd5_gmem.c, fd5_draw.c; MIT, Copyright Rob Clark and the
 * Mesa contributors); the register encoders are the headers Mesa generates (a5xx/). The shaders are what ir3 compiled
 * (lavcc.cpp), taken as the plain data of lava5.h.
 */
#include <stdint.h>
#include <string.h>
#include "lava5.h"
#define A5XX_MAX_RENDER_TARGETS 8

/* performance experiments: bit mask in /var/tmp/lavexp (decimal), read once per process. 1: RB DP2 bit, 2: CCU cfg for A512 bypass */
#include <stdio.h>
unsigned
lav5_exp(void)
{
	static int v = -1;

	if (v < 0) {
		FILE * f = fopen("/var/tmp/lavexp", "r");
		int x = 0;

		if (f != NULL) {
			if (fscanf(f, "%d", &x) != 1) {
				x = 0;
			}
			fclose(f);
		}
		v = x;
	}
	return (unsigned)v;
}

static inline uint32_t
fui(float f)
{
	uint32_t u;

	memcpy(&u, &f, 4);
	return u;
}

static inline uint16_t
_mesa_float_to_half(float f)
{
	(void)f;
	return 0;
}

#include "a5xx/adreno_common.xml.h"
#include "a5xx/adreno_pm4.xml.h"
#include "a5xx/a3xx.xml.h"
#include "a5xx/a5xx.xml.h"

struct st {
	uint32_t * w;
	size_t n, cap;
	int over;
};

static unsigned
par(unsigned v)
{
	return (__builtin_popcount(v) & 1) ^ 1;
}

static void
out(struct st * s, uint32_t v)
{
	if (s->n < s->cap) {
		s->w[s->n++] = v;
	} else {
		s->over = 1;
	}
}

static void
pkt4(struct st * s, unsigned reg, unsigned cnt)
{
	out(s, 0x40000000 | cnt | par(cnt) << 7 | (reg & 0x3ffff) << 8 | par(reg) << 27);
}

static void
pkt7(struct st * s, unsigned op, unsigned cnt)
{
	out(s, 0x70000000 | cnt | par(cnt) << 15 | (op & 0x7f) << 16 | par(op) << 23);
}

static void
set(struct st * s, unsigned reg, uint32_t v)
{
	pkt4(s, reg, 1);
	out(s, v);
}

static void
zeros(struct st * s, unsigned reg, unsigned n)
{
	pkt4(s, reg, n);
	for (unsigned i = 0; i < n; i++) {
		out(s, 0);
	}
}

#define regid(num, comp) (((num) << 2) | (comp))
#define NOREG regid(63, 0)
#define VALIDREG(r) ((r) != NOREG)
#define COND(c, v) ((c) ? (v) : 0)

static void
wfi(struct st * s)
{
	pkt7(s, CP_WAIT_FOR_IDLE, 0);
}

/* fd5_emit_restore */
static void
restore(struct st * s)
{
	pkt7(s, CP_SET_RENDER_MODE, 5);
	out(s, CP_SET_RENDER_MODE_0_MODE(BYPASS));
	out(s, 0); out(s, 0); out(s, 0); out(s, 0);
	pkt4(s, REG_A5XX_UCHE_CACHE_INVALIDATE_MIN_LO, 5);
	out(s, 0); out(s, 0); out(s, 0); out(s, 0); out(s, 0x12);
	wfi(s);
	set(s, REG_A5XX_HLSQ_UPDATE_CNTL, 0xfffff);
	set(s, REG_A5XX_PC_RESTART_INDEX, 0xffffffff);
	set(s, REG_A5XX_PC_RASTER_CNTL, 0x12);
	pkt4(s, REG_A5XX_GRAS_SU_POINT_MINMAX, 2);
	out(s, A5XX_GRAS_SU_POINT_MINMAX_MIN(1.0f) | A5XX_GRAS_SU_POINT_MINMAX_MAX(4092.0f));
	out(s, A5XX_GRAS_SU_POINT_SIZE(0.5f));
	set(s, REG_A5XX_GRAS_SU_CONSERVATIVE_RAS_CNTL, 0);
	set(s, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_CNTL, 0);
	set(s, REG_A5XX_SP_VS_CONFIG_MAX_CONST, 0);
	set(s, REG_A5XX_SP_FS_CONFIG_MAX_CONST, 0);
	zeros(s, REG_A5XX_UNKNOWN_E292, 2);
	set(s, REG_A5XX_RB_MODE_CNTL, 0x44);
	set(s, REG_A5XX_RB_DBG_ECO_CNTL, 0x00100000 | ((lav5_exp() & 1) ? 0x200u : 0));      /* bit 9: DP2 workaround (kgsl a5xx_start) */
	set(s, REG_A5XX_VFD_MODE_CNTL, 0);
	set(s, REG_A5XX_PC_MODE_CNTL, 0x1f);
	set(s, REG_A5XX_SP_MODE_CNTL, 0x1e);
	set(s, REG_A5XX_SP_DBG_ECO_CNTL, 0x40000800);
	set(s, REG_A5XX_TPL1_MODE_CNTL, 0x544);
	pkt4(s, REG_A5XX_HLSQ_TIMEOUT_THRESHOLD_0, 2);
	out(s, 0x80); out(s, 0);
	set(s, REG_A5XX_VPC_DBG_ECO_CNTL, 0x400);
	set(s, REG_A5XX_HLSQ_MODE_CNTL, 1);
	set(s, REG_A5XX_VPC_MODE_CNTL, 0);
	pkt7(s, CP_SET_DRAW_STATE, 3);
	out(s, CP_SET_DRAW_STATE__0_COUNT(0) | CP_SET_DRAW_STATE__0_DISABLE_ALL_GROUPS | CP_SET_DRAW_STATE__0_GROUP_ID(0));
	out(s, CP_SET_DRAW_STATE__1_ADDR_LO(0));
	out(s, CP_SET_DRAW_STATE__2_ADDR_HI(0));
	set(s, REG_A5XX_GRAS_SU_CONSERVATIVE_RAS_CNTL, 0);
	set(s, REG_A5XX_GRAS_SC_BIN_CNTL, 0);
	set(s, REG_A5XX_VPC_FS_PRIMITIVEID_CNTL, 0xff);
	set(s, REG_A5XX_VPC_SO_OVERRIDE, A5XX_VPC_SO_OVERRIDE_SO_DISABLE);
	zeros(s, REG_A5XX_VPC_SO_BUFFER_BASE_LO(0), 3);
	zeros(s, REG_A5XX_VPC_SO_FLUSH_BASE_LO(0), 2);
	set(s, REG_A5XX_PC_GS_PARAM, 0);
	set(s, REG_A5XX_PC_HS_PARAM, 0);
	set(s, REG_A5XX_TPL1_TP_FS_ROTATION_CNTL, 0);
	set(s, REG_A5XX_UNKNOWN_E004, 0);
	set(s, REG_A5XX_GRAS_SU_LAYERED, 0);
	set(s, REG_A5XX_VPC_SO_BUF_CNTL, 0);
	set(s, REG_A5XX_VPC_SO_BUFFER_OFFSET(0), 0);
	set(s, REG_A5XX_PC_GS_LAYERED, 0);
	set(s, REG_A5XX_UNKNOWN_E5AB, 0);
	set(s, REG_A5XX_UNKNOWN_E5C2, 0);
	zeros(s, REG_A5XX_VPC_SO_BUFFER_BASE_LO(1), 3);
	zeros(s, REG_A5XX_VPC_SO_BUFFER_OFFSET(1), 6);
	zeros(s, REG_A5XX_VPC_SO_BUFFER_OFFSET(2), 6);
	zeros(s, REG_A5XX_VPC_SO_BUFFER_OFFSET(3), 3);
	set(s, REG_A5XX_UNKNOWN_E5DB, 0);
	set(s, REG_A5XX_SP_HS_CTRL_REG0, 0);
	set(s, REG_A5XX_SP_GS_CTRL_REG0, 0);
	zeros(s, REG_A5XX_TPL1_VS_TEX_COUNT, 4);
	zeros(s, REG_A5XX_TPL1_FS_TEX_COUNT, 2);
	zeros(s, REG_A5XX_UNKNOWN_E7C0, 3);
	zeros(s, REG_A5XX_UNKNOWN_E7C5, 3);
	zeros(s, REG_A5XX_UNKNOWN_E7CA, 3);
	zeros(s, REG_A5XX_UNKNOWN_E7CF, 3);
	zeros(s, REG_A5XX_UNKNOWN_E7D4, 3);
	zeros(s, REG_A5XX_UNKNOWN_E7D9, 3);
	set(s, REG_A5XX_RB_CLEAR_CNTL, 0);
}

/* fd5_emit_shader: the code into the stage's shader block, from memory */
static void
emit_shader(struct st * s, enum a4xx_state_block sb, uint64_t code, unsigned instrlen)
{
	pkt7(s, CP_LOAD_STATE4, 3);
	out(s, CP_LOAD_STATE4_0_DST_OFF(0) | CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) | CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
	    CP_LOAD_STATE4_0_NUM_UNIT(instrlen));
	out(s, (uint32_t)code | ST4_SHADER);
	out(s, (uint32_t)(code >> 32));
}

/* fd5_emit_shader_obj: SP_xS_OBJ_START_LO/HI, PVT_MEM_PARAM, PVT_MEM_ADDR (2), PVT_MEM_SIZE */
#define PVT_FIBERS_PER_SP 4096          /* a5xx has no figure in freedreno_devices.py (TODO there); a6xx's, to be safe */
#define PVT_NUM_SP 2

static uint32_t
pvt_per_fiber(const struct lav5_stage * st)
{
	return (st->k.pvtmem_size + 511) & ~511u;
}

static uint32_t
pvt_per_sp(const struct lav5_stage * st)
{
	return (pvt_per_fiber(st) * PVT_FIBERS_PER_SP + 4095) & ~4095u;
}

uint32_t
lav5_pvtmem_bytes(const struct lav5_stage * st)
{
	return st->k.pvtmem_size == 0 ? 0 : pvt_per_sp(st) * PVT_NUM_SP;
}

/* fd5_emit_shader_obj + ir3_get_private_mem: per fiber size, HWSTACKOFFSET and TOTALPVTMEMSIZE both the per-SP size */
static void
emit_shader_obj(struct st * s, unsigned reg, uint64_t code, const struct lav5_stage * st, uint64_t pvt)
{
	uint32_t per_sp = st->k.pvtmem_size ? pvt_per_sp(st) : 0;

	pkt4(s, reg, 6);
	out(s, (uint32_t)code);
	out(s, (uint32_t)(code >> 32));
	out(s, A5XX_SP_VS_PVT_MEM_PARAM_MEMSIZEPERITEM(pvt_per_fiber(st)) | A5XX_SP_VS_PVT_MEM_PARAM_HWSTACKOFFSET(per_sp));
	out(s, (uint32_t)pvt);
	out(s, (uint32_t)(pvt >> 32));
	out(s, A5XX_SP_VS_PVT_MEM_SIZE_TOTALPVTMEMSIZE(per_sp));
}

static void
load_consts(struct st * s, enum a4xx_state_block sb, unsigned vec4, const uint32_t * d, unsigned dwords)
{
	while (dwords > 0) {
		unsigned k = dwords > 256 ? 256 : dwords;

		pkt7(s, CP_LOAD_STATE4, 3 + k);
		out(s, CP_LOAD_STATE4_0_DST_OFF(vec4) | CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) | CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
		    CP_LOAD_STATE4_0_NUM_UNIT(k / 4));
		out(s, CP_LOAD_STATE4_1_EXT_SRC_ADDR(0) | CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
		out(s, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
		for (unsigned i = 0; i < k; i++) {
			out(s, d[i]);
		}
		d += k;
		dwords -= k;
		vec4 += k / 4;
	}
}

/* a stage's constants: the pushed ranges of the buffers bound for it, the immediates, then its driver params */
static const char *
emit_stage_consts(struct st * s, enum a4xx_state_block sb, const struct lav5_kernel * k, const struct lav5_binding * buf,
    const uint32_t * dp, unsigned dpn)
{
	for (unsigned i = 0; i < k->nranges; i++) {
		const struct lav5_range * r = &k->ranges[i];
		const struct lav5_binding * b = r->block < LAV5_MAX_BUFFERS ? &buf[r->block] : NULL;
		uint32_t tmp[256];

		if (r->size_dwords == 0 || r->size_dwords % 4 != 0 || r->size_dwords > 256) {
			return "a constant range that is not a whole number of vec4 or longer than 256 dwords";
		}
		if (r->offset_vec4 + r->size_dwords / 4 > k->constlen) {
			return "a constant range runs past the constant file";
		}
		if (b == NULL || b->cpu == NULL) {
			return "a constant range reads a buffer nothing is bound at";
		}
		memset(tmp, 0, r->size_dwords * 4);
		for (unsigned j = 0; j < r->size_dwords; j++) {
			uint64_t byte = (uint64_t)(r->start_dword + j) * 4;

			if (byte + 4 <= b->bytes) {
				memcpy(&tmp[j], (const uint8_t *)b->cpu + byte, 4);
			}
		}
		load_consts(s, sb, r->offset_vec4, tmp, r->size_dwords);
	}
	if (k->nimm != 0) {
		unsigned n = (k->nimm + 3) & ~3u;
		uint32_t tmp[1024];

		if (k->imm_vec4 + n / 4 > k->constlen || n > 1024) {
			return "the immediates run past the constant file";
		}
		memset(tmp, 0, n * 4);
		memcpy(tmp, k->imm, k->nimm * 4);
		load_consts(s, sb, k->imm_vec4, tmp, n);
	}
	if (dp != NULL && k->dp_vec4 >= 0) {
		unsigned avail = (k->constlen - (uint32_t)k->dp_vec4) * 4;
		unsigned n = dpn < avail ? dpn : avail;

		n &= ~3u;
		if (n != 0) {
			load_consts(s, sb, (uint32_t)k->dp_vec4, dp, n);
		}
	}
	return NULL;
}

/* gl_varying_slot numbers the compiler reports the vertex outputs with */
#define SLOT_CLIP0 17
#define SLOT_CLIP1 18

static int
find_out(const struct lav5_stage * s, unsigned slot)
{
	for (unsigned i = 0; i < s->nout; i++) {
		if (s->out[i].slot == slot) {
			return (int)i;
		}
	}
	return -1;
}

const char *
lav5_refuse_stage(const struct lav5_stage * s)
{
	if (!s->is_fragment) {
		if (((s->clip_mask | s->cull_mask) && (find_out(s, SLOT_CLIP0) < 0 && find_out(s, SLOT_CLIP1) < 0)) ||
		    ((s->clip_mask | s->cull_mask) & ~0xffu)) {
			return "the vertex stage's clip or cull distances are not among its outputs";
		}
		if (!VALIDREG(s->position_regid)) {
			return "the vertex stage writes no position";
		}
	} else if (s->ncolor > 8) {
		return "the fragment stage writes more than 8 colours";
	}
	if (s->k.num_ssbos > 16) {
		return "the stage uses more than 16 device buffers";
	}
	if (s->k.shared_size) {
		return "the stage uses threadgroup memory";
	}
	if (s->k.const_data_size) {
		return "the stage has a constant data table";
	}
	return NULL;
}

size_t
lav5_build_render(uint32_t * words, size_t cap, const struct lav5_render * r, const char ** why)
{
	struct st st = { words, 0, cap, 0 }, * s = &st;
	struct lav5_render rmod;
	struct lav5_stage fsmod;
	const struct lav5_stage * vs = r->vs, * fs;
	if (r->nrt == 2 && r->blend.writemask == 0 && r->blend_x[1].writemask != 0) {
		/* colour target 0 is masked and target 1 is the one written: draw to target 1 alone, as if it were target 0 */
		rmod = *r;
		fsmod = *r->fs;
		rmod.rt_gpu = r->rt[1].gpu;
		rmod.rt_pitch = r->rt[1].pitch;
		rmod.rt_format = r->rt[1].format;
		rmod.blend = r->blend_x[1];
		rmod.nrt = 1;
		rmod.rt[0] = r->rt[1];
		fsmod.color_regid[0] = r->fs->color_regid[1];
		fsmod.color_half[0] = r->fs->color_half[1];
		fsmod.ncolor = r->fs->ncolor > 1 ? r->fs->ncolor - 1 : 1;
		r = &rmod;
	}
	if (r->nrt == 2 && r->blend.writemask != 0 && r->blend_x[1].writemask == 0) {
		/* target 1 is masked (nothing would be written) and nothing reads it back: draw to target 0 alone, saves its memory traffic */
		int reads1 = 0;

		for (unsigned q = 0; q < 32; q++) {
			reads1 |= r->fs->fb_target[q] == 2;
		}
		if (!reads1) {
			rmod = *r;
			fsmod = *r->fs;
			rmod.nrt = 1;
			fsmod.ncolor = 1;
			r = &rmod;
		}
	}
	fs = r->fs;
	if (r == &rmod) {
		fs = &fsmod;
	}
	struct lav5_link lk = *r->link;
	const struct lav5_link * l = &lk;
	uint32_t clip_cull = vs->clip_mask | vs->cull_mask, clip0_loc = 0xff, clip1_loc = 0xff;
	unsigned vs_instrlen = vs->k.instrlen, fs_instrlen = fs->k.instrlen;
	unsigned vs_constlen = vs->k.constlen / 4, fs_constlen = fs->k.constlen / 4;
	unsigned vs_constoff = 0, fs_constoff = vs_constlen;
	unsigned fs_instroff, vs_instroff = 0;
	enum a3xx_threadsize fssz = fs->k.double_threadsize ? FOUR_QUADS : TWO_QUADS;
	uint32_t pos_regid = vs->position_regid, psize_regid = vs->point_size_regid;
	unsigned psize_loc = l->psize_loc;
	const char * e;
	unsigned i, j, nvtx = 0;
	uint32_t color_fmt = r->rt_format == 1 ? RB5_R8G8B8A8_UNORM : RB5_R8G8B8A8_UNORM;
	enum a3xx_color_swap swap = r->rt_format == 1 ? WZYX : WXYZ;
	int32_t vp_x, vp_y;
	unsigned nrt = r->nrt ? r->nrt : 1;

	*why = NULL;
	if (vs->nattr != r->nattr) {
		*why = "the vertex stage's attributes and the vertex descriptor's differ in number";
		return 0;
	}
	if (!VALIDREG(pos_regid)) {
		*why = "the vertex stage writes no position";
		return 0;
	}
	if ((e = lav5_refuse_stage(vs)) != NULL || (e = lav5_refuse_stage(fs)) != NULL) {
		*why = e;
		return 0;
	}
	/* clip and cull distances go after the position and the point size in the VPC (fd5_program.c) */
	if (clip_cull) {
		int k0 = find_out(vs, SLOT_CLIP0), k1 = find_out(vs, SLOT_CLIP1);

		if ((clip_cull & 0xf) && k0 >= 0 && lk.cnt < 32 && lk.max_loc + 4 <= 32) {
			clip0_loc = lk.max_loc;
			lk.var[lk.cnt].slot = SLOT_CLIP0;
			lk.var[lk.cnt].regid = vs->out[k0].regid;
			lk.var[lk.cnt].compmask = clip_cull & 0xf;
			lk.var[lk.cnt].loc = (uint8_t)lk.max_loc;
			lk.cnt++;
			lk.max_loc += 4;
		}
		if ((clip_cull >> 4) && k1 >= 0 && lk.cnt < 32 && lk.max_loc + 4 <= 32) {
			clip1_loc = lk.max_loc;
			lk.var[lk.cnt].slot = SLOT_CLIP1;
			lk.var[lk.cnt].regid = vs->out[k1].regid;
			lk.var[lk.cnt].compmask = clip_cull >> 4;
			lk.var[lk.cnt].loc = (uint8_t)lk.max_loc;
			lk.cnt++;
			lk.max_loc += 4;
		}
		if (clip0_loc == 0xff && clip1_loc == 0xff) {
			*why = "the vertex stage's clip distances have no room in the VPC";
			return 0;
		}
	}

	/* setup_stages: when both do not fit the instruction memory, one or both run from external memory */
	if (vs_instrlen + fs_instrlen > 64) {
		if (fs_instrlen < 64) {
			vs_instrlen = 0;
		} else if (vs_instrlen < 64) {
			fs_instrlen = 0;
		} else {
			vs_instrlen = fs_instrlen = 0;
		}
	}
	fs_instroff = 64 - fs_instrlen;

	/* ---- fd5_emit_sysmem_prep ---- */
	restore(s);
	pkt7(s, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
	out(s, 0);
	pkt7(s, CP_EVENT_WRITE, 1);
	out(s, PC_CCU_INVALIDATE_COLOR);
	set(s, REG_A5XX_PC_POWER_CNTL, 1);
	set(s, REG_A5XX_VFD_POWER_CNTL, 1);
	wfi(s);
	set(s, REG_A5XX_RB_CCU_CNTL, (lav5_exp() & 2) ? 0x08000000 : 0x10000000);
	set(s, REG_A5XX_RB_CNTL, A5XX_RB_CNTL_WIDTH(0) | A5XX_RB_CNTL_HEIGHT(0) | A5XX_RB_CNTL_BYPASS);
	pkt4(s, REG_A5XX_GRAS_SC_WINDOW_SCISSOR_TL, 2);
	out(s, A5XX_GRAS_SC_WINDOW_SCISSOR_TL_X(0) | A5XX_GRAS_SC_WINDOW_SCISSOR_TL_Y(0));
	out(s, A5XX_GRAS_SC_WINDOW_SCISSOR_BR_X(r->rt_width - 1) | A5XX_GRAS_SC_WINDOW_SCISSOR_BR_Y(r->rt_height - 1));
	pkt4(s, REG_A5XX_RB_RESOLVE_CNTL_1, 2);
	out(s, A5XX_RB_RESOLVE_CNTL_1_X(0) | A5XX_RB_RESOLVE_CNTL_1_Y(0));
	out(s, A5XX_RB_RESOLVE_CNTL_2_X(r->rt_width - 1) | A5XX_RB_RESOLVE_CNTL_2_Y(r->rt_height - 1));
	set(s, REG_A5XX_RB_WINDOW_OFFSET, A5XX_RB_WINDOW_OFFSET_X(0) | A5XX_RB_WINDOW_OFFSET_Y(0));
	set(s, REG_A5XX_VPC_SO_OVERRIDE, 0);
	pkt7(s, CP_SET_VISIBILITY_OVERRIDE, 1);
	out(s, 0x1);

	/* emit_zs with no depth buffer */
	pkt4(s, REG_A5XX_RB_DEPTH_BUFFER_INFO, 5);
	out(s, A5XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT(DEPTH5_NONE));
	out(s, 0); out(s, 0); out(s, 0); out(s, 0);
	set(s, REG_A5XX_GRAS_SU_DEPTH_BUFFER_INFO, A5XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT(DEPTH5_NONE));
	zeros(s, REG_A5XX_RB_DEPTH_FLAG_BUFFER_BASE_LO, 3);
	set(s, REG_A5XX_RB_STENCIL_INFO, 0);

	/* emit_mrt: the colour targets of the pass, the others empty */
	for (i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
		pkt4(s, REG_A5XX_RB_MRT_BUF_INFO(i), 5);
		if (i < nrt) {
			uint64_t gpu = i == 0 ? r->rt_gpu : r->rt[i].gpu;
			uint32_t pitch = i == 0 ? r->rt_pitch : r->rt[i].pitch;
			enum a3xx_color_swap sw = (i == 0 ? r->rt_format : r->rt[i].format) == 1 ? WZYX : WXYZ;

			out(s, A5XX_RB_MRT_BUF_INFO_COLOR_FORMAT(color_fmt) | A5XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE(TILE5_LINEAR) |
			    A5XX_RB_MRT_BUF_INFO_COLOR_SWAP(sw));
			out(s, A5XX_RB_MRT_PITCH(pitch));
			out(s, A5XX_RB_MRT_ARRAY_PITCH(pitch * r->rt_height));
			out(s, (uint32_t)gpu);
			out(s, (uint32_t)(gpu >> 32));
		} else {
			out(s, A5XX_RB_MRT_BUF_INFO_COLOR_FORMAT(0) | A5XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE(TILE5_LINEAR) |
			    A5XX_RB_MRT_BUF_INFO_COLOR_SWAP(WZYX));
			out(s, A5XX_RB_MRT_PITCH(0));
			out(s, A5XX_RB_MRT_ARRAY_PITCH(0));
			out(s, 0);
			out(s, 0);
		}
		set(s, REG_A5XX_SP_FS_MRT_REG(i), A5XX_SP_FS_MRT_REG_COLOR_FORMAT(i < nrt ? color_fmt : 0));
		zeros(s, REG_A5XX_RB_MRT_FLAG_BUFFER(i), 4);
	}
	/* emit_msaa: one sample */
	pkt4(s, REG_A5XX_TPL1_TP_RAS_MSAA_CNTL, 2);
	out(s, A5XX_TPL1_TP_RAS_MSAA_CNTL_SAMPLES(MSAA_ONE));
	out(s, A5XX_TPL1_TP_DEST_MSAA_CNTL_SAMPLES(MSAA_ONE) | A5XX_TPL1_TP_DEST_MSAA_CNTL_MSAA_DISABLE);
	pkt4(s, REG_A5XX_RB_RAS_MSAA_CNTL, 2);
	out(s, A5XX_RB_RAS_MSAA_CNTL_SAMPLES(MSAA_ONE));
	out(s, A5XX_RB_DEST_MSAA_CNTL_SAMPLES(MSAA_ONE) | A5XX_RB_DEST_MSAA_CNTL_MSAA_DISABLE);
	pkt4(s, REG_A5XX_GRAS_SC_RAS_MSAA_CNTL, 2);
	out(s, A5XX_GRAS_SC_RAS_MSAA_CNTL_SAMPLES(MSAA_ONE));
	out(s, A5XX_GRAS_SC_DEST_MSAA_CNTL_SAMPLES(MSAA_ONE) | A5XX_GRAS_SC_DEST_MSAA_CNTL_MSAA_DISABLE);

	/* ---- fd5_emit_state ---- */
	set(s, REG_A5XX_RB_RENDER_COMPONENTS, A5XX_RB_RENDER_COMPONENTS_RT0(0xf) |
	    COND(nrt > 1, A5XX_RB_RENDER_COMPONENTS_RT1(0xf)) | COND(nrt > 2, A5XX_RB_RENDER_COMPONENTS_RT2(0xf)) |
	    COND(nrt > 3, A5XX_RB_RENDER_COMPONENTS_RT3(0xf)) | COND(nrt > 4, A5XX_RB_RENDER_COMPONENTS_RT4(0xf)) |
	    COND(nrt > 5, A5XX_RB_RENDER_COMPONENTS_RT5(0xf)) | COND(nrt > 6, A5XX_RB_RENDER_COMPONENTS_RT6(0xf)) |
	    COND(nrt > 7, A5XX_RB_RENDER_COMPONENTS_RT7(0xf)));
	set(s, REG_A5XX_RB_ALPHA_CONTROL, 0);
	set(s, REG_A5XX_RB_STENCIL_CONTROL, 0);
	set(s, REG_A5XX_GRAS_LRZ_CNTL, 0);
	pkt4(s, REG_A5XX_RB_STENCILREFMASK, 2);
	out(s, 0); out(s, 0);
	set(s, REG_A5XX_RB_DEPTH_CNTL, 0);
	set(s, REG_A5XX_RB_DEPTH_PLANE_CNTL, 0);
	set(s, REG_A5XX_GRAS_SU_DEPTH_PLANE_CNTL, 0);

	/* scissor (Metal's, top-left origin: the same numbers); first band if band_step > 0 */
	uint32_t first_h = (r->band_step != 0 && r->scissor[3] > r->band_step) ? r->band_step : r->scissor[3];
	pkt4(s, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0, 2);
	out(s, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(r->scissor[0]) | A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(r->scissor[1]));
	out(s, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(r->scissor[0] + r->scissor[2] - 1) |
	    A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(r->scissor[1] + first_h - 1));
	pkt4(s, REG_A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0, 2);
	out(s, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(r->scissor[0]) | A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(r->scissor[1]));
	out(s, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(r->scissor[0] + r->scissor[2] - 1) |
	    A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(r->scissor[1] + first_h - 1));

	/* viewport: NDC y up to a window y down (Metal's), z in [0, 1] */
	(void)vp_x; (void)vp_y;
	wfi(s);
	pkt4(s, REG_A5XX_GRAS_CL_VPORT_XOFFSET_0, 6);
	out(s, A5XX_GRAS_CL_VPORT_XOFFSET_0(r->viewport[0] + r->viewport[2] * 0.5f));
	out(s, A5XX_GRAS_CL_VPORT_XSCALE_0(r->viewport[2] * 0.5f));
	out(s, A5XX_GRAS_CL_VPORT_YOFFSET_0(r->viewport[1] + r->viewport[3] * 0.5f));
	out(s, A5XX_GRAS_CL_VPORT_YSCALE_0(-r->viewport[3] * 0.5f));
	out(s, A5XX_GRAS_CL_VPORT_ZOFFSET_0(r->viewport[4]));
	out(s, A5XX_GRAS_CL_VPORT_ZSCALE_0(r->viewport[5] - r->viewport[4]));

	/* ---- fd5_program_emit ---- */
	pkt4(s, REG_A5XX_HLSQ_VS_CONFIG, 5);
	out(s, A5XX_HLSQ_VS_CONFIG_CONSTOBJECTOFFSET(vs_constoff) | A5XX_HLSQ_VS_CONFIG_SHADEROBJOFFSET(vs_instroff) | A5XX_HLSQ_VS_CONFIG_ENABLED);
	out(s, A5XX_HLSQ_FS_CONFIG_CONSTOBJECTOFFSET(fs_constoff) | A5XX_HLSQ_FS_CONFIG_SHADEROBJOFFSET(fs_instroff) | A5XX_HLSQ_FS_CONFIG_ENABLED);
	out(s, A5XX_HLSQ_HS_CONFIG_CONSTOBJECTOFFSET(fs_constoff + fs_constlen) | A5XX_HLSQ_HS_CONFIG_SHADEROBJOFFSET(fs_instroff));
	out(s, A5XX_HLSQ_DS_CONFIG_CONSTOBJECTOFFSET(fs_constoff + fs_constlen) | A5XX_HLSQ_DS_CONFIG_SHADEROBJOFFSET(fs_instroff));
	out(s, A5XX_HLSQ_GS_CONFIG_CONSTOBJECTOFFSET(fs_constoff + fs_constlen) | A5XX_HLSQ_GS_CONFIG_SHADEROBJOFFSET(fs_instroff));
	set(s, REG_A5XX_HLSQ_CS_CONFIG, 0);
	pkt4(s, REG_A5XX_HLSQ_VS_CNTL, 5);
	out(s, A5XX_HLSQ_VS_CNTL_INSTRLEN(vs_instrlen) | COND(vs->k.num_ssbos, A5XX_HLSQ_VS_CNTL_SSBO_ENABLE));
	out(s, A5XX_HLSQ_FS_CNTL_INSTRLEN(fs_instrlen) | COND(fs->k.num_ssbos, A5XX_HLSQ_FS_CNTL_SSBO_ENABLE));
	out(s, A5XX_HLSQ_HS_CNTL_INSTRLEN(0));
	out(s, A5XX_HLSQ_DS_CNTL_INSTRLEN(0));
	out(s, A5XX_HLSQ_GS_CNTL_INSTRLEN(0));
	pkt4(s, REG_A5XX_SP_VS_CONFIG, 5);
	out(s, A5XX_SP_VS_CONFIG_CONSTOBJECTOFFSET(vs_constoff) | A5XX_SP_VS_CONFIG_SHADEROBJOFFSET(vs_instroff) | A5XX_SP_VS_CONFIG_ENABLED);
	out(s, A5XX_SP_FS_CONFIG_CONSTOBJECTOFFSET(fs_constoff) | A5XX_SP_FS_CONFIG_SHADEROBJOFFSET(fs_instroff) | A5XX_SP_FS_CONFIG_ENABLED);
	out(s, A5XX_SP_HS_CONFIG_CONSTOBJECTOFFSET(fs_constoff + fs_constlen) | A5XX_SP_HS_CONFIG_SHADEROBJOFFSET(fs_instroff));
	out(s, A5XX_SP_DS_CONFIG_CONSTOBJECTOFFSET(fs_constoff + fs_constlen) | A5XX_SP_DS_CONFIG_SHADEROBJOFFSET(fs_instroff));
	out(s, A5XX_SP_GS_CONFIG_CONSTOBJECTOFFSET(fs_constoff + fs_constlen) | A5XX_SP_GS_CONFIG_SHADEROBJOFFSET(fs_instroff));
	set(s, REG_A5XX_SP_CS_CONFIG, 0);
	pkt4(s, REG_A5XX_HLSQ_VS_CONSTLEN, 2);
	out(s, vs_constlen); out(s, vs_instrlen);
	pkt4(s, REG_A5XX_HLSQ_FS_CONSTLEN, 2);
	out(s, fs_constlen); out(s, fs_instrlen);
	zeros(s, REG_A5XX_HLSQ_HS_CONSTLEN, 2);
	zeros(s, REG_A5XX_HLSQ_DS_CONSTLEN, 2);
	zeros(s, REG_A5XX_HLSQ_GS_CONSTLEN, 2);
	zeros(s, REG_A5XX_HLSQ_CS_CONSTLEN, 2);

	set(s, REG_A5XX_SP_VS_CTRL_REG0,
	    A5XX_SP_VS_CTRL_REG0_HALFREGFOOTPRINT((unsigned)(vs->k.max_half_reg + 1)) |
	    A5XX_SP_VS_CTRL_REG0_FULLREGFOOTPRINT((unsigned)(vs->k.max_reg + 1)) |
	    COND(vs_instrlen != 0, A5XX_SP_VS_CTRL_REG0_BUFFER) | 0x2 | A5XX_SP_VS_CTRL_REG0_BRANCHSTACK(vs->k.branchstack) |
	    COND(vs->need_pixlod, A5XX_SP_VS_CTRL_REG0_PIXLODENABLE));

	pkt4(s, REG_A5XX_VPC_VAR_DISABLE(0), 4);
	for (i = 0; i < 4; i++) {
		out(s, ~l->varmask[i]);
	}
	for (i = 0, j = 0; i < 16 && j < l->cnt; i++) {
		uint32_t reg = 0;

		pkt4(s, REG_A5XX_SP_VS_OUT_REG(i), 1);
		reg |= A5XX_SP_VS_OUT_REG_A_REGID(l->var[j].regid) | A5XX_SP_VS_OUT_REG_A_COMPMASK(l->var[j].compmask);
		j++;
		if (j < 32) {
			reg |= A5XX_SP_VS_OUT_REG_B_REGID(l->var[j].regid) | A5XX_SP_VS_OUT_REG_B_COMPMASK(l->var[j].compmask);
		}
		j++;
		out(s, reg);
	}
	for (i = 0, j = 0; i < 8 && j < l->cnt; i++) {
		uint32_t reg = 0;

		pkt4(s, REG_A5XX_SP_VS_VPC_DST_REG(i), 1);
		reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC0(j < 32 ? l->var[j].loc : 0); j++;
		reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC1(j < 32 ? l->var[j].loc : 0); j++;
		reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC2(j < 32 ? l->var[j].loc : 0); j++;
		reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC3(j < 32 ? l->var[j].loc : 0); j++;
		out(s, reg);
	}
	emit_shader_obj(s, REG_A5XX_SP_VS_OBJ_START_LO, r->vs_code, r->vs, r->vs_pvt_gpu);
	if (vs_instrlen) {
		emit_shader(s, SB4_VS_SHADER, r->vs_code, vs_instrlen);
	}
	set(s, REG_A5XX_PC_PRIM_VTX_CNTL, COND(VALIDREG(psize_regid), A5XX_PC_PRIM_VTX_CNTL_PSIZE));
	set(s, REG_A5XX_SP_PRIMITIVE_CNTL, A5XX_SP_PRIMITIVE_CNTL_VSOUT(l->cnt));
	set(s, REG_A5XX_VPC_CNTL_0, A5XX_VPC_CNTL_0_STRIDE_IN_VPC(l->max_loc) | COND(fs->total_in > 0, A5XX_VPC_CNTL_0_VARYING) | 0x10000);
	emit_shader_obj(s, REG_A5XX_SP_FS_OBJ_START_LO, r->fs_code, r->fs, r->fs_pvt_gpu);

	pkt4(s, REG_A5XX_HLSQ_CONTROL_0_REG, 5);
	out(s, A5XX_HLSQ_CONTROL_0_REG_FSTHREADSIZE(fssz) | A5XX_HLSQ_CONTROL_0_REG_CSTHREADSIZE(TWO_QUADS) | 0x00000880);
	out(s, A5XX_HLSQ_CONTROL_1_REG_PRIMALLOCTHRESHOLD(63));
	out(s, A5XX_HLSQ_CONTROL_2_REG_FACEREGID(fs->face_regid) | A5XX_HLSQ_CONTROL_2_REG_SAMPLEID(NOREG) |
	    A5XX_HLSQ_CONTROL_2_REG_SAMPLEMASK(NOREG) | A5XX_HLSQ_CONTROL_2_REG_CENTERRHW(fs->ij_regid[3]));
	out(s, A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_PIXEL(fs->ij_regid[0]) | A5XX_HLSQ_CONTROL_3_REG_IJ_LINEAR_PIXEL(fs->ij_regid[4]) |
	    A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_CENTROID(fs->ij_regid[2]) | A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_CENTROID(fs->ij_regid[5]));
	out(s, A5XX_HLSQ_CONTROL_4_REG_XYCOORDREGID(fs->frag_coord_regid) |
	    A5XX_HLSQ_CONTROL_4_REG_ZWCOORDREGID(VALIDREG(fs->frag_coord_regid) ? fs->frag_coord_regid + 2 : NOREG) |
	    A5XX_HLSQ_CONTROL_4_REG_IJ_PERSP_SAMPLE(fs->ij_regid[1]) | A5XX_HLSQ_CONTROL_4_REG_IJ_LINEAR_SAMPLE(fs->ij_regid[6]));

	set(s, REG_A5XX_SP_FS_CTRL_REG0,
	    COND(fs->total_in > 0, A5XX_SP_FS_CTRL_REG0_VARYING) | 0x40002 | COND(fs_instrlen != 0, A5XX_SP_FS_CTRL_REG0_BUFFER) |
	    A5XX_SP_FS_CTRL_REG0_THREADSIZE(fssz) | A5XX_SP_FS_CTRL_REG0_HALFREGFOOTPRINT((unsigned)(fs->k.max_half_reg + 1)) |
	    A5XX_SP_FS_CTRL_REG0_FULLREGFOOTPRINT((unsigned)(fs->k.max_reg < 0 ? 1 : fs->k.max_reg + 1)) | A5XX_SP_FS_CTRL_REG0_BRANCHSTACK(fs->k.branchstack) |
	    COND(fs->need_pixlod, A5XX_SP_FS_CTRL_REG0_PIXLODENABLE));
	set(s, REG_A5XX_HLSQ_UPDATE_CNTL, 0x020fffff);
	set(s, REG_A5XX_VPC_GS_SIV_CNTL, 0x0000ffff);
	set(s, REG_A5XX_SP_SP_CNTL, 0x00000010);

	{
		uint32_t ij = 0, rc0 = 0;

		if (VALIDREG(fs->ij_regid[0])) { ij |= A5XX_GRAS_CNTL_IJ_PERSP_PIXEL; rc0 |= A5XX_RB_RENDER_CONTROL0_IJ_PERSP_PIXEL; }
		if (VALIDREG(fs->ij_regid[2])) { ij |= A5XX_GRAS_CNTL_IJ_PERSP_CENTROID; rc0 |= A5XX_RB_RENDER_CONTROL0_IJ_PERSP_CENTROID; }
		if (VALIDREG(fs->ij_regid[1])) { ij |= A5XX_GRAS_CNTL_IJ_PERSP_SAMPLE; rc0 |= A5XX_RB_RENDER_CONTROL0_IJ_PERSP_SAMPLE; }
		if (VALIDREG(fs->ij_regid[4])) { ij |= A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL; rc0 |= A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL; }
		if (VALIDREG(fs->ij_regid[5])) { ij |= A5XX_GRAS_CNTL_IJ_LINEAR_CENTROID; rc0 |= A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_CENTROID; }
		if (VALIDREG(fs->ij_regid[6])) { ij |= A5XX_GRAS_CNTL_IJ_LINEAR_SAMPLE; rc0 |= A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_SAMPLE; }
		if (fs->frag_coord_compmask != 0) {
			ij |= A5XX_GRAS_CNTL_COORD_MASK(fs->frag_coord_compmask) | A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL;
			rc0 |= A5XX_RB_RENDER_CONTROL0_COORD_MASK(fs->frag_coord_compmask) | A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL;
		}
		if (VALIDREG(fs->face_regid)) {
			ij |= A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL;
			rc0 |= A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL;
		}
		set(s, REG_A5XX_GRAS_CNTL, ij);
		pkt4(s, REG_A5XX_RB_RENDER_CONTROL0, 2);
		out(s, rc0);
		out(s, COND(VALIDREG(fs->face_regid), A5XX_RB_RENDER_CONTROL1_FACENESS));
	}

	pkt4(s, REG_A5XX_SP_FS_OUTPUT_REG(0), 8);
	for (i = 0; i < 8; i++) {
		uint32_t rid = i < fs->ncolor && i < nrt ? fs->color_regid[i] : NOREG;

		out(s, A5XX_SP_FS_OUTPUT_REG_REGID(rid) | COND(i < fs->ncolor && i < nrt && fs->color_half[i], A5XX_SP_FS_OUTPUT_REG_HALF_PRECISION));
	}
	set(s, REG_A5XX_VPC_PACK, A5XX_VPC_PACK_NUMNONPOSVAR(fs->total_in) | A5XX_VPC_PACK_PSIZELOC(psize_loc));
	pkt4(s, REG_A5XX_VPC_VARYING_INTERP_MODE(0), 8);
	for (i = 0; i < 8; i++) {
		out(s, l->interp[i]);
	}
	zeros(s, REG_A5XX_VPC_VARYING_PS_REPL_MODE(0), 8);
	set(s, REG_A5XX_GRAS_VS_CL_CNTL, A5XX_GRAS_VS_CL_CNTL_CLIP_MASK(vs->clip_mask) | A5XX_GRAS_VS_CL_CNTL_CULL_MASK(vs->cull_mask));
	set(s, REG_A5XX_VPC_CLIP_CNTL, A5XX_VPC_CLIP_CNTL_CLIP_MASK(clip_cull) | A5XX_VPC_CLIP_CNTL_CLIP_DIST_03_LOC(clip0_loc) |
	    A5XX_VPC_CLIP_CNTL_CLIP_DIST_47_LOC(clip1_loc));
	set(s, REG_A5XX_PC_CLIP_CNTL, A5XX_PC_CLIP_CNTL_CLIP_MASK(vs->clip_mask));
	if (fs_instrlen) {
		emit_shader(s, SB4_FS_SHADER, r->fs_code, fs_instrlen);
	}
	pkt4(s, REG_A5XX_VFD_CONTROL_1, 5);
	out(s, A5XX_VFD_CONTROL_1_REGID4VTX(vs->vertex_id_regid) | A5XX_VFD_CONTROL_1_REGID4INST(vs->instance_id_regid) | 0xfc0000);
	out(s, 0x0000fcfc);
	out(s, 0x0000fcfc);
	out(s, 0x000000fc);
	out(s, 0);

	/* rasterizer: the cull and the winding; Metal's front face is clockwise unless the encoder says otherwise */
	set(s, REG_A5XX_GRAS_SU_CNTL, A5XX_GRAS_SU_CNTL_LINEHALFWIDTH(0.5f) | A5XX_GRAS_SU_CNTL_LINE_MODE(BRESENHAM) |
	    COND(r->cull & 1, A5XX_GRAS_SU_CNTL_CULL_FRONT) | COND(r->cull & 2, A5XX_GRAS_SU_CNTL_CULL_BACK) |
	    COND(r->front_cw, A5XX_GRAS_SU_CNTL_FRONT_CW));
	pkt4(s, REG_A5XX_GRAS_SU_POINT_MINMAX, 2);
	out(s, A5XX_GRAS_SU_POINT_MINMAX_MIN(1.0f) | A5XX_GRAS_SU_POINT_MINMAX_MAX(VALIDREG(psize_regid) ? 4092.0f : 1.0f));
	out(s, A5XX_GRAS_SU_POINT_SIZE(1.0f));
	zeros(s, REG_A5XX_GRAS_SU_POLY_OFFSET_SCALE, 3);
	set(s, REG_A5XX_PC_RASTER_CNTL, 0);
	set(s, REG_A5XX_GRAS_CL_CNTL, A5XX_GRAS_CL_CNTL_ZERO_GB_SCALE_Z);
	set(s, REG_A5XX_PC_PRIMITIVE_CNTL, A5XX_PC_PRIMITIVE_CNTL_PROVOKING_VTX_LAST | A5XX_PC_PRIMITIVE_CNTL_STRIDE_IN_VPC(l->max_loc));
	set(s, REG_A5XX_RB_FS_OUTPUT_CNTL, A5XX_RB_FS_OUTPUT_CNTL_MRT(nrt));
	set(s, REG_A5XX_SP_FS_OUTPUT_CNTL, A5XX_SP_FS_OUTPUT_CNTL_MRT(nrt) | A5XX_SP_FS_OUTPUT_CNTL_DEPTH_REGID(NOREG) |
	    A5XX_SP_FS_OUTPUT_CNTL_SAMPLEMASK_REGID(NOREG));

	/* constants of both stages; the vertex stage's driver params are {draw id, vertex offset, first instance, 0, indexed} */
	{
		uint32_t dp[8] = { 0, r->first, r->base_instance, 0, 0, 0, 0, 0 };

		if ((e = emit_stage_consts(s, SB4_VS_SHADER, &vs->k, r->vs_buf, dp, 8)) != NULL ||
		    (e = emit_stage_consts(s, SB4_FS_SHADER, &fs->k, r->fs_buf, NULL, 0)) != NULL) {
			*why = e;
			return 0;
		}
	}

	/* device buffers (emit_ssbos): one table shared by the stages, sizes then addresses; slot i is Metal's buffer index i */
	{
		unsigned n = vs->k.num_ssbos > fs->k.num_ssbos ? vs->k.num_ssbos : fs->k.num_ssbos;

		if (n > 0) {
			pkt7(s, CP_LOAD_STATE4, 3 + 2 * n);
			out(s, CP_LOAD_STATE4_0_DST_OFF(0) | CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) | CP_LOAD_STATE4_0_STATE_BLOCK(SB4_SSBO) |
			    CP_LOAD_STATE4_0_NUM_UNIT(n));
			out(s, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) | CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
			out(s, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
			for (i = 0; i < n; i++) {
				const struct lav5_binding * b = i < vs->k.num_ssbos && r->vs_buf[i].gpu ? &r->vs_buf[i] : &r->fs_buf[i];
				uint64_t sz = b->gpu != 0 ? b->bytes : 4;

				out(s, A5XX_SSBO_1_0_WIDTH((uint32_t)sz & 0xffff));
				out(s, A5XX_SSBO_1_1_HEIGHT((uint32_t)(sz >> 16)));
			}
			pkt7(s, CP_LOAD_STATE4, 3 + 2 * n);
			out(s, CP_LOAD_STATE4_0_DST_OFF(0) | CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) | CP_LOAD_STATE4_0_STATE_BLOCK(SB4_SSBO) |
			    CP_LOAD_STATE4_0_NUM_UNIT(n));
			out(s, CP_LOAD_STATE4_1_STATE_TYPE(ST4_UBO) | CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
			out(s, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
			for (i = 0; i < n; i++) {
				const struct lav5_binding * b = i < vs->k.num_ssbos && r->vs_buf[i].gpu ? &r->vs_buf[i] : &r->fs_buf[i];
				uint64_t a = b->gpu != 0 ? b->gpu : r->hole_gpu;

				out(s, (uint32_t)a);
				out(s, (uint32_t)(a >> 32));
			}
		}
	}

	/* blend (fd5_blend_state_create): target 0 as the pipeline says, the others off */
	{
		const struct lav5_blend * b0 = &r->blend;
		uint32_t mask = 0;

		for (i = 0; i < nrt; i++) {
			if ((i == 0 ? r->blend : r->blend_x[i]).enable) {
				mask |= 1u << i;
			}
		}
		for (i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
			const struct lav5_blend * b = i == 0 ? b0 : &r->blend_x[i];
			uint32_t ctl = A5XX_RB_MRT_CONTROL_ROP_CODE(ROP_COPY) | A5XX_RB_MRT_CONTROL_COMPONENT_ENABLE(i < nrt ? b->writemask : 0);
			uint32_t bc = A5XX_RB_MRT_BLEND_CONTROL_RGB_SRC_FACTOR(FACTOR_ONE) | A5XX_RB_MRT_BLEND_CONTROL_RGB_DEST_FACTOR(FACTOR_ZERO) |
			    A5XX_RB_MRT_BLEND_CONTROL_ALPHA_SRC_FACTOR(FACTOR_ONE) | A5XX_RB_MRT_BLEND_CONTROL_ALPHA_DEST_FACTOR(FACTOR_ZERO);

			if (i < nrt && b->enable) {
				ctl |= A5XX_RB_MRT_CONTROL_BLEND | A5XX_RB_MRT_CONTROL_BLEND2;
				bc = A5XX_RB_MRT_BLEND_CONTROL_RGB_SRC_FACTOR(b->rgb_src) | A5XX_RB_MRT_BLEND_CONTROL_RGB_BLEND_OPCODE(b->rgb_op) |
				    A5XX_RB_MRT_BLEND_CONTROL_RGB_DEST_FACTOR(b->rgb_dst) | A5XX_RB_MRT_BLEND_CONTROL_ALPHA_SRC_FACTOR(b->a_src) |
				    A5XX_RB_MRT_BLEND_CONTROL_ALPHA_BLEND_OPCODE(b->a_op) | A5XX_RB_MRT_BLEND_CONTROL_ALPHA_DEST_FACTOR(b->a_dst);
			}
			set(s, REG_A5XX_RB_MRT_CONTROL(i), ctl);
			set(s, REG_A5XX_RB_MRT_BLEND_CONTROL(i), bc);
		}
		set(s, REG_A5XX_SP_BLEND_CNTL, A5XX_SP_BLEND_CNTL_ENABLE_BLEND(mask) | A5XX_SP_BLEND_CNTL_UNK8);
		set(s, REG_A5XX_RB_BLEND_CNTL, A5XX_RB_BLEND_CNTL_ENABLE_BLEND(mask) | A5XX_RB_BLEND_CNTL_SAMPLE_MASK(0xffff) |
		    COND(nrt > 1, A5XX_RB_BLEND_CNTL_INDEPENDENT_BLEND));
		if (mask) {
			/* the constant blend colour (fd5_emit_state, FD_DIRTY_BLEND_COLOR) */
			pkt4(s, REG_A5XX_RB_BLEND_RED, 8);
			for (i = 0; i < 4; i++) {
				float c = r->blend_color[i], cl = c < 0 ? 0 : c > 1 ? 1 : c, cs = c < -1 ? -1 : c > 1 ? 1 : c;
				uint32_t f16 = _mesa_float_to_half(c) & 0xffff, u8 = (uint32_t)(cl * 0xff) & 0xff, s8 = (uint32_t)(int32_t)(cs * 0x7f) & 0xff;

				out(s, f16 | (u8 << 16) | (s8 << 24));
				out(s, fui(c));
			}
		}
	}

	/* textures and samplers of the fragment stage (emit_textures): units 0..n-1, n the highest index used + 1 */
	{
		uint32_t nt = 0, ns = 0;

		for (i = 0; i < 16; i++) {
			if (r->tex[i].valid) {
				nt = i + 1;
			}
			if (r->samp[i].valid) {
				ns = i + 1;
			}
		}
		/* a stage may sample a unit nothing is bound at (a texture that was refused is nil): give every unit up to the highest
		 * a valid descriptor and sampler, or the TP reads a width-0 image and the CP never finishes the draw */
		/* the uber shader samples whichever unit its (unspecialised) path names, bound or not: all 16 get a descriptor */
		nt = fs->ntexslots > 16 && fs->ntexslots <= 32 ? fs->ntexslots : 16;
		ns = nt;
		if (nt > ns) {
			ns = nt;
		}
		if (ns > nt) {
			nt = ns;
		}
		if (ns > 0) {
			pkt7(s, CP_LOAD_STATE4, 3 + 4 * ns);
			out(s, CP_LOAD_STATE4_0_DST_OFF(0) | CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) | CP_LOAD_STATE4_0_STATE_BLOCK(SB4_FS_TEX) |
			    CP_LOAD_STATE4_0_NUM_UNIT(ns));
			out(s, CP_LOAD_STATE4_1_STATE_TYPE(ST4_SHADER) | CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
			out(s, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
			for (i = 0; i < ns; i++) {
				const struct lav5_samp * sp = &r->samp[i];
				enum a5xx_tex_filter mag = sp->mag ? A5XX_TEX_LINEAR : A5XX_TEX_NEAREST, min = sp->min ? A5XX_TEX_LINEAR : A5XX_TEX_NEAREST;
				enum a5xx_tex_clamp ws = sp->wrap_s == 0 ? A5XX_TEX_REPEAT : sp->wrap_s == 2 ? A5XX_TEX_MIRROR_REPEAT : A5XX_TEX_CLAMP_TO_EDGE;
				enum a5xx_tex_clamp wt = sp->wrap_t == 0 ? A5XX_TEX_REPEAT : sp->wrap_t == 2 ? A5XX_TEX_MIRROR_REPEAT : A5XX_TEX_CLAMP_TO_EDGE;

				out(s, COND(sp->mip_linear, A5XX_TEX_SAMP_0_MIPFILTER_LINEAR_NEAR) | A5XX_TEX_SAMP_0_XY_MAG(mag) |
				    A5XX_TEX_SAMP_0_XY_MIN(min) | A5XX_TEX_SAMP_0_WRAP_S(ws) | A5XX_TEX_SAMP_0_WRAP_T(wt) |
				    A5XX_TEX_SAMP_0_WRAP_R(A5XX_TEX_CLAMP_TO_EDGE));
				out(s, A5XX_TEX_SAMP_1_MIN_LOD(0.0f) | A5XX_TEX_SAMP_1_MAX_LOD(0.125f) | A5XX_TEX_SAMP_1_CUBEMAPSEAMLESSFILTOFF);
				out(s, A5XX_TEX_SAMP_2_BCOLOR_OFFSET(i));
				out(s, 0);
			}
		}
		if (nt > 0) {
			pkt7(s, CP_LOAD_STATE4, 3 + 12 * nt);
			out(s, CP_LOAD_STATE4_0_DST_OFF(0) | CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) | CP_LOAD_STATE4_0_STATE_BLOCK(SB4_FS_TEX) |
			    CP_LOAD_STATE4_0_NUM_UNIT(nt));
			out(s, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) | CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
			out(s, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
			for (i = 0; i < nt; i++) {
				struct lav5_tex dummy;

				memset(&dummy, 0, sizeof(dummy));
				const struct lav5_tex * t = i < 16 ? &r->tex[i] : &dummy;
				uint32_t sx = A5XX_TEX_X, sy = A5XX_TEX_Y, sz = A5XX_TEX_Z, sw = A5XX_TEX_W;

				if (i < 32 && fs->fb_target[i] != 0 && fs->fb_target[i] <= 8) {
					/* a framebuffer fetch: the colour attachment the shader reads, as a texture at this unit */
					const struct lav5_rt * rt = &r->rt[fs->fb_target[i] - 1];

					memset(&dummy, 0, sizeof(dummy));
					dummy.gpu = rt->gpu;
					dummy.width = r->rt_width;
					dummy.height = r->rt_height;
					dummy.pitch = rt->pitch;
					dummy.bgra = rt->format == 0;
					dummy.nch = 4;
					dummy.valid = 1;
					t = &dummy;
				} else if (!t->valid) {
					memset(&dummy, 0, sizeof(dummy));
					dummy.gpu = r->hole_gpu;
					dummy.width = dummy.height = 1;
					dummy.pitch = 64;
					dummy.nch = 4;
					t = &dummy;
				}
				uint32_t tfmt = t->tfmt ? t->tfmt : (uint32_t)TFMT5_8_8_8_8_UNORM;

				if (t->valid && t->bgra) {
					sx = A5XX_TEX_Z;
					sz = A5XX_TEX_X;
				}
				if (t->valid && t->nch != 0 && t->nch < 4 && !t->alpha_only) {
					/* fdl6_format_swiz: the channels a format lacks read 0, and w reads 1 */
					if (t->nch < 2) {
						sy = A5XX_TEX_ZERO;
					}
					if (t->nch < 3) {
						sz = A5XX_TEX_ZERO;
					}
					sw = A5XX_TEX_ONE;
				}
				out(s, A5XX_TEX_CONST_0_FMT(tfmt) | A5XX_TEX_CONST_0_TILE_MODE(TILE5_LINEAR) | A5XX_TEX_CONST_0_SWIZ_X(sx) |
				    A5XX_TEX_CONST_0_SWIZ_Y(sy) | A5XX_TEX_CONST_0_SWIZ_Z(sz) | A5XX_TEX_CONST_0_SWIZ_W(sw) | COND(t->srgb, A5XX_TEX_CONST_0_SRGB));
				out(s, A5XX_TEX_CONST_1_WIDTH(t->width) | A5XX_TEX_CONST_1_HEIGHT(t->height));
				out(s, A5XX_TEX_CONST_2_PITCHALIGN(0) | A5XX_TEX_CONST_2_PITCH(t->pitch) | A5XX_TEX_CONST_2_TYPE(t->is1d ? A5XX_TEX_1D : A5XX_TEX_2D));
				out(s, A5XX_TEX_CONST_3_ARRAY_PITCH((t->pitch * t->height + 4095) & ~4095u));
				out(s, (uint32_t)t->gpu);
				out(s, (uint32_t)(t->gpu >> 32) | A5XX_TEX_CONST_5_DEPTH(1));
				for (j = 0; j < 6; j++) {
					out(s, 0);
				}
			}
		}
		set(s, REG_A5XX_TPL1_FS_TEX_COUNT, nt);
	}
	set(s, REG_A5XX_TPL1_CS_TEX_COUNT, 0);

	/* vertex buffers: one fetch and one decode per attribute (fd5_emit_vertex_bufs) */
	for (i = 0; i < r->nattr; i++) {
		const struct lav5_attr * a = &r->attr[i];
		const struct lav5_vbuf * vb = &r->vbuf[a->buffer];
		uint64_t base = vb->gpu + a->offset;
		const struct lav5_io * in = NULL;

		for (j = 0; j < vs->nin; j++) {
			if (!vs->in[j].sysval && vs->in[j].compmask && vs->in[j].slot == a->index + 15) { /* VERT_ATTRIB_GENERIC0 is 15 */
				in = &vs->in[j];
			}
		}
		if (in == NULL) {
			continue;               /* an attribute the vertex stage does not read */
		}
		pkt4(s, REG_A5XX_VFD_FETCH(nvtx), 4);
		out(s, (uint32_t)base);
		out(s, (uint32_t)(base >> 32));
		out(s, (uint32_t)(vb->bytes - a->offset));
		out(s, vb->stride);
		pkt4(s, REG_A5XX_VFD_DECODE(nvtx), 2);
		out(s, A5XX_VFD_DECODE_INSTR_IDX(nvtx) | A5XX_VFD_DECODE_INSTR_FORMAT(a->format) | A5XX_VFD_DECODE_INSTR_SWAP(a->swap) |
		    A5XX_VFD_DECODE_INSTR_UNK30 | COND(a->is_float, A5XX_VFD_DECODE_INSTR_FLOAT));
		out(s, 1);
		set(s, REG_A5XX_VFD_DEST_CNTL(nvtx), A5XX_VFD_DEST_CNTL_INSTR_WRITEMASK(in->compmask) | A5XX_VFD_DEST_CNTL_INSTR_REGID(in->regid));
		nvtx++;
	}
	set(s, REG_A5XX_VFD_CONTROL_0, A5XX_VFD_CONTROL_0_VTXCNT(nvtx));

	pkt4(s, REG_A5XX_VFD_INDEX_OFFSET, 2);
	out(s, r->first);
	out(s, r->base_instance);
	set(s, REG_A5XX_PC_RESTART_INDEX, 0xffffffff);
	set(s, REG_A5XX_RB_RENDER_CNTL, 0x8);
	set(s, REG_A5XX_GRAS_SC_CNTL, 0x8);

	/* the draw (and any remaining scissor bands in the same stream) */
	uint32_t band_step = (r->band_step != 0 && r->scissor[3] > r->band_step) ? r->band_step : r->scissor[3];
	for (uint32_t off = 0; off < r->scissor[3]; off += band_step) {
		if (off > 0) {
			uint32_t by = r->scissor[1] + off;
			uint32_t bh = r->scissor[3] - off < band_step ? r->scissor[3] - off : band_step;

			wfi(s);
			pkt4(s, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0, 2);
			out(s, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(r->scissor[0]) | A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(by));
			out(s, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(r->scissor[0] + r->scissor[2] - 1) |
			    A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(by + bh - 1));
			pkt4(s, REG_A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0, 2);
			out(s, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(r->scissor[0]) | A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(by));
			out(s, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(r->scissor[0] + r->scissor[2] - 1) |
			    A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(by + bh - 1));
			wfi(s);
		}
		if (r->index_gpu != 0) {
			/* fd5_draw.c: instances, count, first index 0, the index buffer's address and the indices it holds */
			pkt7(s, CP_DRAW_INDX_OFFSET, 7);
			out(s, CP_DRAW_INDX_OFFSET_0_PRIM_TYPE(r->prim) | CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT(DI_SRC_SEL_DMA) |
			    CP_DRAW_INDX_OFFSET_0_INDEX_SIZE(r->index_size == 1 ? INDEX4_SIZE_16_BIT : INDEX4_SIZE_32_BIT) |
			    CP_DRAW_INDX_OFFSET_0_VIS_CULL(IGNORE_VISIBILITY));
			out(s, r->instances);
			out(s, r->count);
			out(s, 0);
			out(s, (uint32_t)r->index_gpu);
			out(s, (uint32_t)(r->index_gpu >> 32));
			out(s, r->index_max);
		} else {
			pkt7(s, CP_DRAW_INDX_OFFSET, 3);
			out(s, CP_DRAW_INDX_OFFSET_0_PRIM_TYPE(r->prim) | CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT(DI_SRC_SEL_AUTO_INDEX) |
			    CP_DRAW_INDX_OFFSET_0_INDEX_SIZE(INDEX4_SIZE_32_BIT) | CP_DRAW_INDX_OFFSET_0_VIS_CULL(IGNORE_VISIBILITY));
			out(s, r->instances);
			out(s, r->count);
		}
	}

	/* fd5_emit_sysmem_fini, then the flush with a timestamp so that memory holds the pixels */
	pkt7(s, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
	out(s, 0);
	wfi(s);
	pkt7(s, CP_EVENT_WRITE, 4);
	out(s, PC_CCU_FLUSH_COLOR_TS);
	out(s, (uint32_t)r->flush_gpu);
	out(s, (uint32_t)(r->flush_gpu >> 32));
	out(s, 0xc0c0c0c0);
	wfi(s);
	pkt7(s, CP_EVENT_WRITE, 1);
	out(s, CACHE_FLUSH);
	wfi(s);
	pkt7(s, CP_EVENT_WRITE, 4);
	out(s, CACHE_FLUSH_TS);
	out(s, (uint32_t)(r->flush_gpu + 4));
	out(s, (uint32_t)((r->flush_gpu + 4) >> 32));
	out(s, 0xa4a4a4a4);
	pkt7(s, CP_WAIT_MEM_WRITES, 0);
	wfi(s);

	if (s->over) {
		*why = "the stream is longer than the service takes";
		return 0;
	}
	return s->n;
}

/*
 * A scatter copy by the 2D engine: `src_gpu` (linear, the texture) is copied, in units of `runlen` bytes (a multiple of 8192), to the
 * physical addresses dst_pa[i] taken from the surface's runs, by CP_BLIT as R8 images 8192 wide (the sequence of step 18 of the
 * kernel, which copied 64 bytes right). src_off[i] is where run i starts in the source. Returns the words used, or 0 when it
 * does not fit.
 */
size_t
lav5_build_copy(uint32_t * words, size_t cap, uint64_t src_gpu, const uint64_t * dst_pa, const uint64_t * src_off,
    const uint32_t * len, unsigned nruns, uint64_t flush_gpu)
{
	struct st st = { words, 0, cap, 0 }, * s = &st;

	pkt4(s, REG_A5XX_UCHE_CACHE_INVALIDATE_MIN_LO, 5);
	out(s, 0);
	out(s, 0);
	out(s, 0);
	out(s, 0);
	out(s, 0x12);
	wfi(s);
	set(s, REG_A5XX_RB_RENDER_CNTL, 0x8);
	set(s, 0x2100, 0x86000000u);                    /* RB_2D_BLIT_CNTL: ROP_COPY, UNORM8, bit 31 */
	set(s, 0x2180, 0x86000000u);                    /* GRAS_2D_BLIT_CNTL */
	set(s, 0x2184, 0x9);
	set(s, REG_A5XX_RB_CNTL, 1u << 17);             /* BYPASS */
	set(s, REG_A5XX_RB_MODE_CNTL, 1u << 1);
	set(s, REG_A5XX_SP_MODE_CNTL, 0xc);
	set(s, REG_A5XX_TPL1_MODE_CNTL, 0x244u | (1u << 7));
	set(s, REG_A5XX_HLSQ_MODE_CNTL, 0x2);
	set(s, REG_A5XX_GRAS_CL_CNTL, 0x181);
	pkt7(s, CP_SET_RENDER_MODE, 1);
	out(s, 5);                                      /* BLIT2D */
	set(s, 0x2040, 0x3u << 3);                      /* SP_A2D_OUTPUT_INFO: float, R8_UNORM */
	set(s, 0x2181, 0x3);
	set(s, 0x2182, 0x3);
	for (unsigned i = 0; i < nruns; i++) {
		uint64_t sp = src_gpu + src_off[i], dp = dst_pa[i];
		unsigned rows = len[i] / 8192, pitch = 8192 >> 6, apitch = (8192 * rows) >> 6;

		pkt4(s, 0x2107, 9);                     /* RB_2D_SRC_INFO .. */
		out(s, 0x3);
		out(s, (uint32_t)sp);
		out(s, (uint32_t)(sp >> 32));
		out(s, pitch | (apitch << 16));
		for (int k = 0; k < 5; k++) {
			out(s, 0);
		}
		pkt4(s, 0x2110, 9);                     /* RB_2D_DST_INFO .. */
		out(s, 0x3);
		out(s, (uint32_t)dp);
		out(s, (uint32_t)(dp >> 32));
		out(s, pitch | (apitch << 16));
		for (int k = 0; k < 5; k++) {
			out(s, 0);
		}
		pkt4(s, 0x2000, 5);                     /* TPL1_A2D_SRC_TEXTURE_INFO .. */
		out(s, 0x3);
		out(s, 8192u | (rows << 15));
		out(s, (uint32_t)sp);
		out(s, (uint32_t)(sp >> 32));
		out(s, pitch << 9);
		pkt7(s, CP_BLIT, 5);
		out(s, 3);                              /* BLIT_OP_SCALE */
		out(s, 0);
		out(s, 8191u | ((rows - 1) << 16));
		out(s, 0);
		out(s, 8191u | ((rows - 1) << 16));
	}
	pkt7(s, CP_SET_RENDER_MODE, 1);
	out(s, 8);                                      /* END2D */
	wfi(s);
	pkt7(s, CP_EVENT_WRITE, 4);
	out(s, PC_CCU_FLUSH_COLOR_TS);
	out(s, (uint32_t)flush_gpu);
	out(s, (uint32_t)(flush_gpu >> 32));
	out(s, 0xc0c0c0c1);
	wfi(s);
	return s->over ? 0 : s->n;
}

/* debug: one dword written by CP_MEM_WRITE (no 2D engine) to a physical address, for the test above 4 GB */
size_t
lav5_build_memwrite(uint32_t * words, size_t cap, uint64_t pa, uint32_t val)
{
	struct st st = { words, 0, cap, 0 }, * s = &st;

	pkt7(s, 0x3d, 3);
	out(s, (uint32_t)pa);
	out(s, (uint32_t)(pa >> 32));
	out(s, val);
	wfi(s);
	return s->over ? 0 : s->n;
}
