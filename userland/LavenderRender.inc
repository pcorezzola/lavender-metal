#include <libkern/OSCacheControl.h>
/* LavenderRender.inc: M5.4 - textures, render pipeline states and the render encoder. Included by LavenderMetal.m (one
 * translation unit), after the compute classes and before the device. One draw is one stream (lava5r.c), sysmem mode; the
 * colour target is a texture the encoder cleared (by the CPU) or loaded. No depth, blending or sampling yet; what is not
 * served is refused by name in the log. */

typedef struct { double x, y, w, h, znear, zfar; } LavViewport;
typedef struct { uint64_t x, y, w, h; } LavScissor;
typedef struct { uint64_t x, y, z; } LavOrigin;
typedef struct { LavOrigin origin; LavSize size; } LavRegion;
typedef struct { double r, g, b, a; } LavClear;

static id
msg0(id o, const char * sel)
{
	return o != nil ? ((id (*)(id, SEL))objc_msgSend)(o, sel_registerName(sel)) : nil;
}

static uint64_t
msgU(id o, const char * sel)
{
	return o != nil ? ((uint64_t (*)(id, SEL))objc_msgSend)(o, sel_registerName(sel)) : 0;
}

static id
msgAt(id o, const char * sel, uint64_t i)
{
	return o != nil ? ((id (*)(id, SEL, uint64_t))objc_msgSend)(o, sel_registerName(sel), i) : nil;
}

#define PF_RGBA8 70
#define PF_BGRA8 80

/* Metal pixel formats a texture can be made in, for sampling (and getBytes/replaceRegion): bytes a pixel, the a5xx TFMT5_*, the
 * channels it holds, and the swizzle rule (lava5r.c). Only RGBA8 and BGRA8 are also drawn into. */
struct pfmt { uint32_t mtl, bpp, tfmt, nch, bgra, alpha_only, srgb; };
static const struct pfmt kPfmt[] = {
	{ 1,   1, 2,            1, 0, 1, 0 },    /* A8Unorm */
	{ 10,  1, 3,             1, 0, 0, 0 },    /* R8Unorm */
	{ 20,  2, 21,            1, 0, 0, 0 },    /* R16Unorm */
	{ 23,  2, 24,             1, 0, 0, 0 },    /* R16Uint */
	{ 25,  2, 23,            1, 0, 0, 0 },    /* R16Float */
	{ 30,  2, 15,           2, 0, 0, 0 },    /* RG8Unorm */
	{ 40,  2, 14,         3, 0, 0, 0 },    /* B5G6R5Unorm */
	{ 43,  2, 10,       4, 1, 0, 0 },    /* BGR5A1Unorm */
	{ 53,  4, 75,             1, 0, 0, 0 },    /* R32Uint */
	{ 55,  4, 74,            1, 0, 0, 0 },    /* R32Float */
	{ 60,  4, 67,         2, 0, 0, 0 },    /* RG16Unorm */
	{ 65,  4, 69,         2, 0, 0, 0 },    /* RG16Float */
	{ 70,  4, 48,       4, 0, 0, 0 },    /* RGBA8Unorm */
	{ 71,  4, 48,       4, 0, 0, 1 },    /* RGBA8Unorm_sRGB */
	{ 80,  4, 48,       4, 1, 0, 0 },    /* BGRA8Unorm */
	{ 81,  4, 48,       4, 1, 0, 1 },    /* BGRA8Unorm_sRGB */
	{ 90,  4, 54,    4, 0, 0, 0 },    /* RGB10A2Unorm */
	{ 105, 8, 103,         2, 0, 0, 0 },    /* RG32Float */
	{ 115, 8, 98,   4, 0, 0, 0 },    /* RGBA16Float */
};

static const struct pfmt *
pfmtOf(uint64_t mtl)
{
	for (size_t i = 0; i < sizeof(kPfmt) / sizeof(kPfmt[0]); i++) {
		if (kPfmt[i].mtl == mtl) {
			return &kPfmt[i];
		}
	}
	return NULL;
}

static BOOL
pfIsDraw(uint32_t fmt)
{
	return fmt == PF_RGBA8 || fmt == PF_BGRA8;
}

struct texstate {
	IOSurfaceRef surface;                   /* retained: the texture is a CAMetalLayer drawable; present copies into it */
	struct lbuf b;
	uint32_t width, height, format, pitch;
	uint32_t bpp, type;                     /* bytes a pixel; Metal's textureType (0: 1D, 2: 2D) */
	uint64_t usage, storage;
	id label;
	uint64_t * segPa;                       /* the physical runs of the surface (16 KB pages mostly), for the 2D-engine copy */
	uint64_t * segOff;                      /* where each starts in the texture */
	uint32_t * segLen;
	uint32_t nseg, segsTried;
	uint32_t segsLogged;                    /* the physical runs of the surface were listed once (debug) */
	uint32_t synced;                        /* the texture memory equals the surface (written back whole once): a Load need not read the slow surface again */
};

/* a big texture may live in write-back memory (lavexp bit 32): fast for the CPU, so every CPU access is followed by a clean+invalidate of the
 * bytes it touched (and a page of read-ahead), and the cache never holds a line of it while the GPU works */
static void
texCpuDone(struct texstate * t, size_t off, size_t len)
{
	if (t->b.cached && len != 0 && off < t->b.len) {
		size_t end = off + len + 4096;

		if (end > t->b.len) {
			end = t->b.len;
		}
		sys_dcache_flush((uint8_t *)t->b.cpu + off, end - off);
	}
}

static void
texCpuDoneRows(struct texstate * t, size_t ya, size_t yb)
{
	texCpuDone(t, ya * t->pitch, (yb - ya) * (size_t)t->pitch);
}

static BOOL
lavBufNewTex(size_t len, struct lbuf * b)
{
	if ((lav5_exp() & 32) && len >= POOL_SLAB_BYTES / 2 && gConn != 0 && gStream.handle != 0) {
		memset(b, 0, sizeof(*b));
		if (lavBufNewDirectF(len, b, 1)) {
			return YES;
		}
	}
	return lavBufNew(len, b);
}

static id
deviceNewTexture(__unused id self, __unused SEL _cmd, id desc)
{
	uint64_t w = msgU(desc, "width"), h = msgU(desc, "height"), fmt = msgU(desc, "pixelFormat"), type = msgU(desc, "textureType");
	uint64_t depth = msgU(desc, "depth"), mips = msgU(desc, "mipmapLevelCount"), samples = msgU(desc, "sampleCount");
	uint64_t layers = msgU(desc, "arrayLength");
	struct texstate * st;
	const struct pfmt * pf;
	id o;
	size_t pitch, len;
	BOOL ok;

	if (!lavOpen() || w == 0 || h == 0 || w > 4096 || h > 4096) {
		lavlog("newTexture %llux%llu refused", (unsigned long long)w, (unsigned long long)h);
		return nil;
	}
	pf = pfmtOf(fmt);
	if (pf == NULL || (type != 2 && type != 0) || depth > 1 || samples > 1 || layers > 1) {
		lavlog("newTexture: pixel format %llu, type %llu, %llu mips, %llu samples, %llu layers: not served (1D/2D, one sample, one layer, a format of kPfmt)",
		    (unsigned long long)fmt, (unsigned long long)type, (unsigned long long)mips, (unsigned long long)samples,
		    (unsigned long long)layers);
		return nil;
	}
	pitch = ((size_t)w * pf->bpp + 63) & ~(size_t)63;
	len = pitch * (size_t)h;
	if (mips > 1) {
		/* room for the whole chain (under 4/3 of the base level); only level 0 is drawn to and sampled for now */
		len = len + len / 2;
	}
	if (len > (16u << 20)) {
		return nil;
	}
	o = makeObject(gTextureClass, sizeof(struct texstate));
	if (o == nil) {
		return nil;
	}
	st = STATE(o, struct texstate);
	memset(st, 0, sizeof(*st));
	pthread_mutex_lock(&gGPULock);
	ok = lavBufNewTex((len + 4095) & ~(size_t)4095, &st->b);
	pthread_mutex_unlock(&gGPULock);
	if (!ok) {
		return nil;
	}
	memset(st->b.cpu, 0, st->b.len);
	texCpuDone(st, 0, st->b.len);
	st->width = (uint32_t)w;
	st->height = (uint32_t)h;
	st->format = (uint32_t)fmt;
	st->bpp = pf->bpp;
	st->type = (uint32_t)type;
	st->pitch = (uint32_t)pitch;
	st->usage = msgU(desc, "usage");
	st->storage = msgU(desc, "storageMode");
	lavlog("texture %llux%llu format %llu, pitch %zu, at GPU %#llx", (unsigned long long)w, (unsigned long long)h,
	    (unsigned long long)fmt, pitch, (unsigned long long)st->b.gpu);
	return o;
}

/* IOSurface <-> the texture's own memory (the GPU cannot reach an IOSurface's scattered pages): rows copied at the edges */
static void
surfaceCopyRows(struct texstate * t, BOOL toSurface, size_t ya, size_t yb, BOOL profile)
{
	size_t sp, rows = t->height, row = (size_t)t->width * t->bpp;
	uint8_t * base;
	uint64_t lockStart = profile ? clock_gettime_nsec_np(CLOCK_MONOTONIC) : 0;
	uint64_t lockNs = 0, copyNs = 0, cacheNs = 0, unlockStart = 0;

	if (IOSurfaceLock(t->surface, toSurface ? 0 : kIOSurfaceLockReadOnly, NULL) != 0) {
		lavlog("IOSurfaceLock failed");
		return;
	}
	if (profile) {
		lockNs = clock_gettime_nsec_np(CLOCK_MONOTONIC) - lockStart;
	}
	sp = IOSurfaceGetBytesPerRow(t->surface);
	base = IOSurfaceGetBaseAddress(t->surface);
	if (IOSurfaceGetHeight(t->surface) < rows) {
		rows = IOSurfaceGetHeight(t->surface);
	}
	if (yb > rows) {
		yb = rows;
	}
	if (toSurface && !t->segsLogged && sp * rows >= (1u << 20)) {
		/* debugging: how many physically contiguous runs make up this surface (could the GPU render into it directly?) */
		uint64_t in[2] = { (uint64_t)(uintptr_t)base, (uint64_t)sp * rows }, out[3 + 2 * 24];
		size_t osz = sizeof(out);
		kern_return_t kr;

		t->segsLogged = 1;
		memset(out, 0, sizeof(out));
		kr = IOConnectCallStructMethod(gConn, kLavSegs, in, sizeof(in), out, &osz);
		lavlog("surface %ux%u pitch %zu: segments kr %#x: %llu runs, longest %llu KB, total %llu KB", t->width, t->height, sp, kr,
		    (unsigned long long)out[0], (unsigned long long)(out[1] >> 10), (unsigned long long)(out[2] >> 10));
		for (unsigned i = 0; i < 24 && 3 + 2 * i + 1 < osz / 8; i++) {
			lavlog("  run %u: pa %#llx, %llu KB", i, (unsigned long long)out[3 + 2 * i], (unsigned long long)(out[4 + 2 * i] >> 10));
		}
	}
	{
		/* the surface is slow memory and one core only gets ~150 MB/s out of it: four slices at once */
		size_t n = yb > ya ? yb - ya : 0, nsl = n >= 64 ? 4 : 1;
		uint8_t * cpu = (uint8_t *)t->b.cpu;
		size_t pitch = t->pitch;

		uint64_t copyStart = profile ? clock_gettime_nsec_np(CLOCK_MONOTONIC) : 0;

		dispatch_apply(nsl, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
			size_t y0 = ya + n * k / nsl, y1 = ya + n * (k + 1) / nsl;

			for (size_t y = y0; y < y1; y++) {
				if (toSurface) {
					memcpy(base + y * sp, cpu + y * pitch, row);
				} else {
					memcpy(cpu + y * pitch, base + y * sp, row);
				}
			}
		});
		if (profile) {
			copyNs = clock_gettime_nsec_np(CLOCK_MONOTONIC) - copyStart;
		}
		if (yb > ya) {
			uint64_t cacheStart = profile ? clock_gettime_nsec_np(CLOCK_MONOTONIC) : 0;
			texCpuDoneRows(t, ya, yb);
			if (profile) {
				cacheNs = clock_gettime_nsec_np(CLOCK_MONOTONIC) - cacheStart;
			}
		}
	}
	if (toSurface && ya == 0 && yb == rows) {
		t->synced = 1;
	} else if (!toSurface) {
		t->synced = 1;
	}
	if (toSurface && t->bpp == 4 && access("/var/tmp/lavtrace", F_OK) == 0) {
		/* what the display is about to scan out: a 16x16 grid of the surface, how many samples are not black, and three of them */
		unsigned lit = 0, tot = 0;
		const uint32_t * c = (const uint32_t *)(base + (rows / 2) * sp + (t->width / 2) * 4);
		const uint32_t * tl = (const uint32_t *)(base + 8 * sp + 8 * 4);

		for (unsigned gy = 0; gy < 16; gy++) {
			for (unsigned gx = 0; gx < 16; gx++) {
				const uint32_t * p = (const uint32_t *)(base + ((size_t)gy * rows / 16) * sp + ((size_t)gx * t->width / 16) * 4);

				tot++;
				if ((*p & 0xffffff) != 0) {
					lit++;
				}
			}
		}
		lavlog("surface %ux%u after the draws: %u of %u samples lit, centre %08x, top-left %08x", t->width, t->height, lit, tot, *c, *tl);
		if (access("/var/tmp/lavshot", F_OK) == 0) {
			/* debugging: the surface as raw BGRA rows, once */
			FILE * sf = fopen("/var/tmp/lavsurf.bgra", "wb");

			unlink("/var/tmp/lavshot");
			if (sf != NULL) {
				for (size_t y = 0; y < rows; y++) {
					fwrite(base + y * sp, 4, t->width, sf);
				}
				fclose(sf);
				lavlog("surface dumped to /var/tmp/lavsurf.bgra (%ux%zu)", t->width, rows);
			}
		}
	}
	if (profile) {
		unlockStart = clock_gettime_nsec_np(CLOCK_MONOTONIC);
	}
	IOSurfaceUnlock(t->surface, toSurface ? 0 : kIOSurfaceLockReadOnly, NULL);
	if (profile) {
		uint64_t unlockNs = clock_gettime_nsec_np(CLOCK_MONOTONIC) - unlockStart;
		lavlog("surface sync: id %u CPU bytes %zu lock %llu us copy %llu us cache %llu us unlock %llu us",
		    (unsigned)IOSurfaceGetID(t->surface), (yb > ya ? yb - ya : 0) * row,
		    (unsigned long long)(lockNs / 1000), (unsigned long long)(copyNs / 1000),
		    (unsigned long long)(cacheNs / 1000), (unsigned long long)(unlockNs / 1000));
	}
}

static void
surfaceCopy(struct texstate * t, BOOL toSurface)
{
	surfaceCopyRows(t, toSurface, 0, t->height, NO);
}

/* blit: a region of one texture copied into another (level 0, one slice). Both are in memory the CPU can reach. */
static BOOL
runTexCopy(struct op * op)
{
	struct texstate * s = STATE(op->src, struct texstate), * d = STATE(op->dst, struct texstate);

	if (s->bpp != d->bpp || op->cx + op->cw > s->width || op->cy + op->ch > s->height || op->cdx + op->cw > d->width ||
	    op->cdy + op->ch > d->height) {
		lavlog("copyFromTexture skipped: %ux%u/%u bytes to %ux%u/%u bytes, region %llux%llu", s->width, s->height, s->bpp, d->width,
		    d->height, d->bpp, (unsigned long long)op->cw, (unsigned long long)op->ch);
		return YES;
	}
	if (s->surface != NULL) {
		surfaceCopy(s, NO);
	}
	for (uint64_t y = 0; y < op->ch; y++) {
		memmove((uint8_t *)d->b.cpu + (op->cdy + y) * d->pitch + op->cdx * d->bpp,
		    (uint8_t *)s->b.cpu + (op->cy + y) * s->pitch + op->cx * s->bpp, op->cw * s->bpp);
	}
	if (op->ch != 0) {
		texCpuDoneRows(s, op->cy, op->cy + op->ch);
		texCpuDoneRows(d, op->cdy, op->cdy + op->ch);
	}
	return YES;
}

static void
encCopyTextureImpl(id self, id src, uint64_t sLevel, LavOrigin so, LavSize ss, id dst, uint64_t dLevel, LavOrigin dorg)
{
	struct cbstate * st = STATE(STATE(self, struct encstate)->commandBuffer, struct cbstate);
	struct op * op;

	if (src == nil || dst == nil || object_getClass(src) != gTextureClass || object_getClass(dst) != gTextureClass) {
		lavlog("copyFromTexture: not a texture of this driver");
		return;
	}
	if (sLevel != 0 || dLevel != 0) {
		lavlog("copyFromTexture of level %llu to %llu: only level 0 is served", (unsigned long long)sLevel, (unsigned long long)dLevel);
		return;
	}
	op = cbAddOp(st);
	op->kind = OP_TEXCOPY;
	op->src = [src retain];
	op->dst = [dst retain];
	op->cx = so.x;
	op->cy = so.y;
	op->cw = ss.width;
	op->ch = ss.height;
	op->cdx = dorg.x;
	op->cdy = dorg.y;
}

static void
encCopyTexture(id self, __unused SEL _cmd, id src, __unused uint64_t sSlice, uint64_t sLevel, LavOrigin so, LavSize ss, id dst,
    __unused uint64_t dSlice, uint64_t dLevel, LavOrigin dorg)
{
	encCopyTextureImpl(self, src, sLevel, so, ss, dst, dLevel, dorg);
}

static void
encCopyTextureOpt(id self, __unused SEL _cmd, id src, __unused uint64_t sSlice, uint64_t sLevel, LavOrigin so, LavSize ss, id dst,
    __unused uint64_t dSlice, uint64_t dLevel, LavOrigin dorg, __unused uint64_t options)
{
	encCopyTextureImpl(self, src, sLevel, so, ss, dst, dLevel, dorg);
}

static void
encGenMips(__unused id self, __unused SEL _cmd, __unused id tex)
{
	/* only level 0 is drawn from, so the chain is not built */
}

static int gBlitTested, gBlitTested2;

/* debug (lavexp bit 8): the 2D scatter copy between two buffers of ours, checked byte by byte, with guard bands round every destination run */
static void
blitSelfTest(void)
{
	enum { NRUN = 16, RUN = 16384, SLOT = 32768, GUARD = 8192 };
	struct lbuf src, dst;
	uint32_t words[STREAM_WORDS];
	uint64_t dpa[NRUN], soff[NRUN];
	uint32_t slen[NRUN];
	size_t w, bad = 0, badguard = 0, first = (size_t)-1;
	BOOL ok;

	/* the commit holds gGPULock already */
	ok = lavBufNew(NRUN * RUN, &src) && lavBufNew(NRUN * SLOT, &dst);
	if (!ok) {
		lavlog("blit self-test: no memory");
		return;
	}
	for (size_t i = 0; i < NRUN * RUN / 4; i++) {
		((uint32_t *)src.cpu)[i] = (uint32_t)(i * 2654435761u) ^ 0x5a5a0000u;
	}
	memset(dst.cpu, 0xa5, NRUN * SLOT);
	for (unsigned k = 0; k < NRUN; k++) {
		/* runs go to their slots in reverse order, like pages of a surface in no order */
		dpa[k] = dst.gpu + (uint64_t)(NRUN - 1 - k) * SLOT + GUARD;
		soff[k] = (uint64_t)k * RUN;
		slen[k] = RUN;
	}
	w = lav5_build_copy(words, STREAM_WORDS, src.gpu, dpa, soff, slen, NRUN, gStream.gpu + SCRATCH_OFFSET + 64);
	lavlog("blit self-test: %zu words for %d runs; submit %s", w, NRUN, (w != 0 && lavSubmit(words, w)) ? "ok" : "FAILED");
	for (unsigned k = 0; k < NRUN; k++) {
		const uint8_t * slot = (const uint8_t *)dst.cpu + (size_t)(NRUN - 1 - k) * SLOT;
		const uint8_t * exp = (const uint8_t *)src.cpu + (size_t)k * RUN;

		for (size_t i = 0; i < SLOT; i++) {
			uint8_t want = (i >= GUARD && i < GUARD + RUN) ? exp[i - GUARD] : 0xa5;

			if (slot[i] != want) {
				if (i >= GUARD && i < GUARD + RUN) {
					bad++;
				} else {
					badguard++;
				}
				if (first == (size_t)-1) {
					first = (size_t)(NRUN - 1 - k) * SLOT + i;
					lavlog("blit self-test: first difference at slot %u offset %zu: %02x, want %02x", NRUN - 1 - k, i, slot[i], want);
				}
			}
		}
	}
	lavlog("blit self-test: %zu data bytes wrong, %zu guard bytes touched (of %d data, %d guard)", bad, badguard, NRUN * RUN, NRUN * (SLOT - RUN));
	lavBufFree(&src);
	lavBufFree(&dst);
}

/* The GPU SMMU currently identity-maps the first 4 GiB. */
static BOOL
gpuSurfaceRunMapped(uint64_t pa, uint64_t len)
{
	const uint64_t limit = 4ull << 30;

	return len != 0 && pa < limit && len <= limit - pa;
}

/* Debug (lavexp bit 16): copy into mapped pages of a test IOSurface and verify the guards. */
static void
blitSelfTestSurface(void)
{
	enum { NR = 16, RUN = 16384, W = 1024, H = 2048, WIN = 2 << 20 };
	struct lbuf src;
	IOSurfaceRef surf;
	CFMutableDictionaryRef d = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	int vw = W, vh = H, vb = 4, vf = 'BGRA';
	CFNumberRef nw = CFNumberCreate(NULL, kCFNumberIntType, &vw), nh = CFNumberCreate(NULL, kCFNumberIntType, &vh);
	CFNumberRef nb = CFNumberCreate(NULL, kCFNumberIntType, &vb), nf = CFNumberCreate(NULL, kCFNumberIntType, &vf);
	uint8_t * base;
	size_t alloc, nrun = 0, nhigh = 0, bad = 0, stray = 0;
	uint64_t * rpa = malloc(2048 * 8), * roff = malloc(2048 * 8);
	uint32_t rlen[2048];
	int pick[NR], np = 0;
	uint64_t dpa[NR], soff[NR];
	uint32_t slen[NR], words[STREAM_WORDS];
	size_t w;

	lavlog("blit surface test: step 1 create");
	CFDictionarySetValue(d, kIOSurfaceWidth, nw);
	CFDictionarySetValue(d, kIOSurfaceHeight, nh);
	CFDictionarySetValue(d, kIOSurfaceBytesPerElement, nb);
	CFDictionarySetValue(d, kIOSurfacePixelFormat, nf);
	surf = IOSurfaceCreate(d);
	CFRelease(nw);
	CFRelease(nh);
	CFRelease(nb);
	CFRelease(nf);
	CFRelease(d);
	if (surf == NULL || rpa == NULL || roff == NULL || !lavBufNew(NR * RUN, &src)) {
		lavlog("blit surface test: no surface or buffer");
		if (surf != NULL) {
			CFRelease(surf);
		}
		free(rpa);
		free(roff);
		return;
	}
	lavlog("blit surface test: step 2 lock");
	IOSurfaceLock(surf, 0, NULL);
	base = IOSurfaceGetBaseAddress(surf);
	alloc = IOSurfaceGetAllocSize(surf);
	lavlog("blit surface test: step 3 memset %p alloc %zu", base, alloc);
	memset(base, 0xa5, alloc);
	sys_dcache_flush(base, alloc);
	lavlog("blit surface test: step 4 flushed");
	for (uint64_t w0 = 0; w0 < alloc; w0 += WIN) {
		uint64_t o[3 + 2 * 250], wl = alloc - w0 < WIN ? alloc - w0 : WIN, in[2] = { (uint64_t)(uintptr_t)base + w0, wl }, off = w0;
		size_t osz = sizeof(o);

		memset(o, 0, sizeof(o));
		lavlog("blit surface test: step 5 kLavSegs at %llu", (unsigned long long)w0);
		if (IOConnectCallStructMethod(gConn, kLavSegs, in, sizeof(in), o, &osz) != KERN_SUCCESS || o[0] == 0 || nrun + o[0] > 2048) {
			lavlog("blit surface test: segs failed at %llu", (unsigned long long)w0);
			goto cleanup;
		}
		for (uint64_t i = 0; i < o[0]; i++, nrun++) {
			rpa[nrun] = o[3 + 2 * i];
			rlen[nrun] = (uint32_t)o[4 + 2 * i];
			roff[nrun] = off;
			off += o[4 + 2 * i];
			nhigh += rpa[nrun] >= (4ull << 30);
		}
	}
	lavlog("blit surface test: %zu runs, %zu above 4 GB, alloc %zu", nrun, nhigh, alloc);
	for (size_t i = 0; i < nrun && np < NR; i++) {
		if (rlen[i] >= RUN && gpuSurfaceRunMapped(rpa[i], RUN)) {
			pick[np++] = (int)i;
		}
	}
	if (np == 0) {
		lavlog("blit surface test: no mapped destination runs");
		goto cleanup;
	}
	for (size_t i = 0; i < NR * RUN / 4; i++) {
		((uint32_t *)src.cpu)[i] = (uint32_t)(i * 2654435761u) ^ 0x3c3c0000u;
	}
	for (int k = 0; k < np; k++) {
		dpa[k] = rpa[pick[k]];
		soff[k] = (uint64_t)k * RUN;
		slen[k] = RUN;
		lavlog("blit surface test: run %d pa %#llx", pick[k], (unsigned long long)dpa[k]);
	}
	w = lav5_build_copy(words, STREAM_WORDS, src.gpu, dpa, soff, slen, np,
	    gStream.gpu + SCRATCH_OFFSET + 64);
	lavlog("blit surface test: %d mapped runs; submit %s", np,
	    (w != 0 && lavSubmit(words, w)) ? "ok" : "FAILED");
	sys_dcache_flush(base, alloc);
	for (size_t y = 0; y < alloc; y++) {
		size_t r = y / RUN * 0, k;
		int hit = -1;

		(void)r;
		for (k = 0; k < (size_t)np; k++) {
			if (y >= roff[pick[k]] && y < roff[pick[k]] + RUN) {
				hit = (int)k;
				break;
			}
		}
		if (hit >= 0) {
			bad += base[y] != ((const uint8_t *)src.cpu)[hit * RUN + (y - roff[pick[hit]])];
		} else {
			stray += base[y] != 0xa5;
		}
	}
	lavlog("blit surface test: %zu picked bytes wrong, %zu other bytes changed", bad, stray);
cleanup:
	IOSurfaceUnlock(surf, 0, NULL);
	CFRelease(surf);
	lavBufFree(&src);
	free(rpa);
	free(roff);
}

/* the texture to the surface by the GPU's 2D engine (the CPU would read write-combined memory, slowly): YES if done */
enum { LAV_COPY_MAX_RUN = 2 << 20 };

static BOOL
gpuSurfaceCopy(struct texstate * t, size_t ya, size_t yb, uint32_t maxRun)
{
	size_t sp = IOSurfaceGetBytesPerRow(t->surface), alloc = IOSurfaceGetAllocSize(t->surface);
	uint8_t * base = IOSurfaceGetBaseAddress(t->surface);
	uint32_t words[STREAM_WORDS];
	uint64_t flush = gStream.gpu + SCRATCH_OFFSET + 64;
	size_t a, b;

	if (sp != t->pitch || base == NULL || yb <= ya) {
		return NO;
	}
	if (t->nseg == 0 && !t->segsTried) {
		/* the kernel answers at most 4 KB of runs per call: ask for the surface in windows of 2 MB */
		enum { WIN = LAV_COPY_MAX_RUN, MAXRUNS = 1024 };
		uint64_t o[3 + 2 * 250];
		uint64_t off = 0, n = 0;
		BOOL ok = YES;

		t->segsTried = 1;
		t->segPa = malloc(MAXRUNS * 8);
		t->segOff = malloc(MAXRUNS * 8);
		t->segLen = malloc(MAXRUNS * 4);
		ok = t->segPa != NULL && t->segOff != NULL && t->segLen != NULL && (alloc % 8192) == 0;
		for (uint64_t w0 = 0; ok && w0 < alloc; w0 += WIN) {
			uint64_t wl = alloc - w0 < WIN ? alloc - w0 : WIN, in[2] = { (uint64_t)(uintptr_t)base + w0, wl };
			size_t osz = sizeof(o);
			kern_return_t kr;

			memset(o, 0, sizeof(o));
			kr = IOConnectCallStructMethod(gConn, kLavSegs, in, sizeof(in), o, &osz);
			ok = kr == KERN_SUCCESS && o[0] != 0 && o[0] == (osz / 8 - 3) / 2 && o[2] == wl && n + o[0] <= MAXRUNS;
			if (!ok) {
				lavlog("GPU copy: window at %llu: kr %#x, %llu runs (%llu listed), total %llu of %llu", (unsigned long long)w0, kr,
				    (unsigned long long)o[0], (unsigned long long)((osz / 8 - 3) / 2), (unsigned long long)o[2], (unsigned long long)wl);
			}
			for (uint64_t i = 0; ok && i < o[0]; i++) {
				t->segPa[n] = o[3 + 2 * i];
				t->segOff[n] = off;
				t->segLen[n] = (uint32_t)o[4 + 2 * i];
				ok = (o[4 + 2 * i] % 8192) == 0 && gpuSurfaceRunMapped(o[3 + 2 * i], o[4 + 2 * i]);
				if (!ok) {
					lavlog("GPU copy: run %llu: pa %#llx len %llu: refused", (unsigned long long)i, (unsigned long long)o[3 + 2 * i], (unsigned long long)o[4 + 2 * i]);
				}
				off += o[4 + 2 * i];
				n++;
			}
		}
		if (ok) {
			t->nseg = (uint32_t)n;
		}
		lavlog("surface %ux%u: GPU copy %s (%llu runs, alloc %zu)", t->width, t->height, ok ? "ready" : "not possible", (unsigned long long)n, alloc);
	}
	if (t->nseg == 0) {
		return NO;
	}
	a = ya * sp;
	b = yb * sp;
	uint32_t runOffset = 0;
	for (uint32_t i = 0; i < t->nseg;) {
		uint64_t dpa[96], soff[96];
		uint32_t slen[96];
		unsigned n = 0;
		size_t w;

		while (i < t->nseg && n < 96) {
			if (t->segOff[i] + t->segLen[i] <= a || t->segOff[i] >= b) {
				i++;
				continue;
			}
			uint32_t length = MIN(t->segLen[i] - runOffset, maxRun);

			dpa[n] = t->segPa[i] + runOffset;
			soff[n] = t->segOff[i] + runOffset;
			slen[n++] = length;
			runOffset += length;
			if (runOffset == t->segLen[i]) {
				i++;
				runOffset = 0;
			}
		}
		if (n == 0) {
			break;
		}
		w = lav5_build_copy(words, STREAM_WORDS, t->b.gpu, dpa, soff, slen, n, flush);
		if (w == 0 || !lavSubmit(words, w)) {
			return NO;
		}
	}
	/* the GPU wrote the pages behind the CPU's cache: drop the stale lines so the display path reads the new pixels */
	if (IOSurfaceLock(t->surface, 0, NULL) == 0) {
		sys_dcache_flush(base + a, b - a);
		IOSurfaceUnlock(t->surface, 0, NULL);
	}
	if (ya == 0 && yb == t->height) {
		t->synced = 1;
	}
	return YES;
}

/* Compare both copy paths on the same GPU-mapped buffers without timing verification. */
static void
blitBenchmark(void)
{
	enum { PITCH = 4352, HEIGHT = 2340, UNIT = 8192, CHUNK = 64 * UNIT,
	    BYTES = (PITCH * HEIGHT + UNIT - 1) & ~(UNIT - 1),
	    RUNS = (BYTES + CHUNK - 1) / CHUNK, TRIALS = 8 };
	struct lbuf src, dst;
	uint32_t words[STREAM_WORDS], len[RUNS];
	uint64_t addresses[RUNS], offsets[RUNS];
	size_t count;

	if (!lavBufNewTex(BYTES, &src)) {
		lavlog("blit benchmark: no source buffer");
		return;
	}
	if (!lavBufNew(BYTES, &dst)) {
		lavlog("blit benchmark: no destination buffer");
		lavBufFree(&src);
		return;
	}
	for (unsigned i = 0; i < RUNS; i++) {
		size_t offset = (size_t)i * CHUNK;

		addresses[i] = dst.gpu + offset;
		offsets[i] = offset;
		len[i] = BYTES - offset < CHUNK ? BYTES - offset : CHUNK;
	}
	count = gpuSurfaceRunMapped(src.gpu, BYTES) && gpuSurfaceRunMapped(dst.gpu, BYTES)
	    ? lav5_build_copy(words, STREAM_WORDS, src.gpu, addresses, offsets, len, RUNS,
	          gStream.gpu + SCRATCH_OFFSET + 64)
	    : 0;
	if (count == 0) {
		lavlog("blit benchmark: could not encode a mapped copy");
		goto out;
	}
	lavlog("blit benchmark: %u bytes, %u runs, source cached %u, destination cached %u",
	    BYTES, RUNS, (unsigned)src.cached, (unsigned)dst.cached);
	for (unsigned trial = 0; trial < TRIALS; trial++) {
		const uint8_t * source = src.cpu;
		uint8_t * target = dst.cpu;

		for (size_t i = 0; i < BYTES / 4; i++) {
			((uint32_t *)src.cpu)[i] = (uint32_t)i * 2654435761u ^ trial;
		}
		if (src.cached) {
			sys_dcache_flush(src.cpu, BYTES);
		}
		for (unsigned pass = 0; pass < 2; pass++) {
			BOOL gpu = ((trial + pass) & 1) != 0;
			BOOL ok = YES;
			uint64_t start, elapsed;

			memset(target, 0xa5, BYTES);
			if (dst.cached) {
				sys_dcache_flush(target, BYTES);
			}
			start = clock_gettime_nsec_np(CLOCK_MONOTONIC);
			if (gpu) {
				ok = lavSubmit(words, count);
			} else {
				dispatch_apply(4, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t slice) {
					size_t first = slice * BYTES / 4, end = (slice + 1) * BYTES / 4;

					memcpy(target + first, source + first, end - first);
				});
				if (src.cached) {
					sys_dcache_flush(src.cpu, BYTES);
				}
			}
			if (dst.cached) {
				sys_dcache_flush(target, BYTES);
			}
			elapsed = clock_gettime_nsec_np(CLOCK_MONOTONIC) - start;
			ok = ok && memcmp(source, target, BYTES) == 0;
			if (src.cached) {
				sys_dcache_flush(src.cpu, BYTES);
			}
			lavlog("blit benchmark: trial %u %s %llu us %s", trial,
			    gpu ? "GPU" : "CPU", (unsigned long long)(elapsed / 1000), ok ? "PASS" : "FAIL");
			if (!ok) {
				goto out;
			}
		}
	}
out:
	lavBufFree(&dst);
	lavBufFree(&src);
}

/* Probe real display-size allocations while keeping three surfaces alive. */
static void
scanoutCopyTest(void)
{
	enum { WIDTH = 1080, HEIGHT = 2340, PITCH = 4352, WINDOW = 2 << 20, SURFACES = 3 };
	NSDictionary * properties = @{
	    (NSString *)kIOSurfaceWidth: @(WIDTH), (NSString *)kIOSurfaceHeight: @(HEIGHT),
	    (NSString *)kIOSurfaceBytesPerElement: @4, (NSString *)kIOSurfaceBytesPerRow: @(PITCH),
	    (NSString *)kIOSurfacePixelFormat: @('BGRA'),
	};
	IOSurfaceRef surfaces[SURFACES] = {};

	for (unsigned i = 0; i < SURFACES; i++) {
		surfaces[i] = IOSurfaceCreate((CFDictionaryRef)properties);
	}
	for (unsigned i = 0; i < SURFACES; i++) {
		IOSurfaceRef surface = surfaces[i];
		if (surface == NULL || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != 0) {
			lavlog("scanout test: slot %u could not create or lock surface", i);
			continue;
		}
		uint8_t * base = IOSurfaceGetBaseAddress(surface);
		size_t alloc = IOSurfaceGetAllocSize(surface), mapped = 0, runs = 0;
		unsigned sid = IOSurfaceGetID(surface);
		uint64_t previousEnd = 0;
		BOOL contiguous = YES, complete = base != NULL;

		for (size_t offset = 0; complete && offset < alloc; offset += WINDOW) {
			size_t length = MIN(alloc - offset, WINDOW);
			uint64_t in[] = { (uint64_t)(uintptr_t)(base + offset), length };
			uint64_t segments[3 + 2 * 250] = {};
			size_t outputSize = sizeof(segments);
			kern_return_t result = IOConnectCallStructMethod(gConn, kLavSegs, in, sizeof(in),
			    segments, &outputSize);

			if (result != KERN_SUCCESS || outputSize < 3 * sizeof(uint64_t) ||
			    segments[0] == 0 || segments[0] > 250 ||
			    segments[0] != (outputSize / sizeof(uint64_t) - 3) / 2 || segments[2] != length) {
				complete = NO;
				break;
			}
			for (size_t j = 0; j < segments[0]; j++, runs++) {
				uint64_t pa = segments[3 + 2 * j], size = segments[4 + 2 * j];

				if (runs != 0 && pa != previousEnd) {
					contiguous = NO;
				}
				previousEnd = pa + size;
				if (gpuSurfaceRunMapped(pa, size)) {
					mapped += size;
				}
			}
		}
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
		lavlog("scanout test: slot %u id %u bytes %zu mapped %zu runs %zu contiguous %u complete %u",
		    i, sid, alloc, mapped, runs, (unsigned)contiguous, (unsigned)complete);
		if (!complete || mapped != alloc || IOSurfaceGetBytesPerRow(surface) != PITCH ||
		    alloc < (size_t)HEIGHT * PITCH) {
			continue;
		}
		struct texstate texture = { .width = WIDTH, .height = HEIGHT, .pitch = PITCH,
		    .bpp = 4, .surface = surface };

		if (!lavBufNewTex(alloc, &texture.b)) {
			lavlog("scanout test: id %u could not allocate source", sid);
			continue;
		}
		for (size_t j = 0; j < alloc / 4; j++) {
			((uint32_t *)texture.b.cpu)[j] = 0xff000000u | ((uint32_t)j * 2654435761u >> 8);
		}
		texCpuDone(&texture, 0, alloc);
		uint64_t start = clock_gettime_nsec_np(CLOCK_MONOTONIC);
		BOOL ok = gpuSurfaceCopy(&texture, 0, HEIGHT, LAV_COPY_MAX_RUN);
		uint64_t elapsed = clock_gettime_nsec_np(CLOCK_MONOTONIC) - start;

		if (ok) {
			ok = IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) == 0;
			if (ok) {
				ok = memcmp(texture.b.cpu, IOSurfaceGetBaseAddress(surface), (size_t)HEIGHT * PITCH) == 0;
				IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
			}
		}
		lavlog("scanout test: id %u GPU copy plus first mapping %llu us %s",
		    sid, (unsigned long long)(elapsed / 1000), ok ? "PASS" : "FAIL");
		free(texture.segPa);
		free(texture.segOff);
		free(texture.segLen);
		lavBufFree(&texture.b);
	}
	for (unsigned i = 0; i < SURFACES; i++) {
		if (surfaces[i] != NULL) {
			CFRelease(surfaces[i]);
		}
	}
}

/* Replay the current frame into the same surface, comparing visible BGRA bytes. */
static void
compareSurfaceCopies(struct texstate * t)
{
	static const uint32_t maxRun[] = { 0, 512 << 10, LAV_COPY_MAX_RUN };
	size_t stride = IOSurfaceGetBytesPerRow(t->surface), rowBytes = (size_t)t->width * t->bpp;
	unsigned sid = IOSurfaceGetID(t->surface);

	for (unsigned trial = 0; trial < 4; trial++) {
		for (unsigned pass = 0; pass < 3; pass++) {
			uint32_t chunk = maxRun[(trial + pass + 1) % 3];
			uint64_t start = clock_gettime_nsec_np(CLOCK_MONOTONIC);
			BOOL ok = YES;

			if (chunk == 0) {
				surfaceCopyRows(t, YES, 0, t->height, NO);
			} else {
				ok = gpuSurfaceCopy(t, 0, t->height, chunk);
			}
			uint64_t elapsed = clock_gettime_nsec_np(CLOCK_MONOTONIC) - start;
			if (ok && IOSurfaceLock(t->surface, kIOSurfaceLockReadOnly, NULL) == 0) {
				const uint8_t * target = IOSurfaceGetBaseAddress(t->surface);
				const uint8_t * source = t->b.cpu;

				for (size_t y = 0; ok && y < t->height; y++) {
					ok = memcmp(source + y * t->pitch, target + y * stride, rowBytes) == 0;
				}
				IOSurfaceUnlock(t->surface, kIOSurfaceLockReadOnly, NULL);
				texCpuDoneRows(t, 0, t->height);
			} else {
				ok = NO;
			}
			lavlog("surface compare: id %u trial %u chunk %u %llu us %s", sid, trial, chunk,
			    (unsigned long long)(elapsed / 1000), ok ? "PASS" : "FAIL");
			if (!ok) {
				surfaceCopyRows(t, YES, 0, t->height, NO);
				return;
			}
		}
	}
}

static void
cbSyncSurfaces(struct cbstate * st)
{
	BOOL profile = access("/var/tmp/lavsurfsync", F_OK) == 0;
	static BOOL didBenchmark, didScanout, didCompare;

	if (!didScanout && access("/var/tmp/lavscanouttest", F_OK) == 0) {
		didScanout = YES;
		scanoutCopyTest();
	}

	if (!didBenchmark && access("/var/tmp/lavblitbench", F_OK) == 0) {
		didBenchmark = YES;
		blitBenchmark();
	}
	if ((lav5_exp() & 8) && !gBlitTested) {
		gBlitTested = 1;
		blitSelfTest();
	}
	if ((lav5_exp() & 16) && !gBlitTested2) {
		gBlitTested2 = 1;
		blitSelfTestSurface();
	}
	for (size_t i = 0; i < st->nops; i++) {
		id tex = st->ops[i].kind == OP_DRAW ? st->ops[i].draw->target : (st->ops[i].kind == OP_CLEAR || st->ops[i].kind == OP_TEXCOPY) ? st->ops[i].dst : nil;

		BOOL seen = NO;

		if (tex == nil || STATE(tex, struct texstate)->surface == NULL) {
			continue;
		}
		/* every draw of the commit that targets the same surface would copy all of it again: once is enough, the texture is final */
		for (size_t j = 0; j < i && !seen; j++) {
			id t2 = st->ops[j].kind == OP_DRAW ? st->ops[j].draw->target : (st->ops[j].kind == OP_CLEAR || st->ops[j].kind == OP_TEXCOPY) ? st->ops[j].dst : nil;

			seen = t2 == tex;
		}
		if (!seen) {
			struct texstate * ts = STATE(tex, struct texstate);
			uint64_t ya = ts->height, yb = 0;

			/* only the rows the commit touched go back to the surface: a clear or full-height draw is all of it, a band is a band */
			for (size_t j = i; j < st->nops; j++) {
				struct op * o = &st->ops[j];
				uint64_t a = 0, b = ts->height;

				if (o->kind == OP_DRAW && o->draw->target == tex) {
					a = o->draw->scissor[1];
					b = a + o->draw->scissor[3];
				} else if (o->kind == OP_TEXCOPY && o->dst == tex) {
					a = o->cdy;
					b = a + o->ch;
				} else if (!(o->kind == OP_CLEAR && o->dst == tex)) {
					continue;
				}
				if (a < ya) {
					ya = a;
				}
				if (b > yb) {
					yb = b;
				}
			}
			if (ya < yb) {
				BOOL gpu = NO;

				if (lav5_exp() & 4) {
					uint64_t start = profile ? clock_gettime_nsec_np(CLOCK_MONOTONIC) : 0;

					gpu = gpuSurfaceCopy(ts, ya, yb, LAV_COPY_MAX_RUN);
					if (profile) {
						uint64_t elapsed = clock_gettime_nsec_np(CLOCK_MONOTONIC) - start;

						lavlog("surface sync: id %u GPU rows %llu-%llu %llu us %s",
						    (unsigned)IOSurfaceGetID(ts->surface), (unsigned long long)ya,
						    (unsigned long long)yb, (unsigned long long)(elapsed / 1000),
						    gpu ? "OK" : "fallback");
					}
				}
				if (!gpu) {
					surfaceCopyRows(ts, YES, ya, yb, profile);
				} else if (!didCompare && ya == 0 && yb == ts->height &&
				    ts->width == 1080 && ts->height == 2340 &&
				    access("/var/tmp/lavsurfacecompare", F_OK) == 0) {
					didCompare = YES;
					compareSurfaceCopies(ts);
				}
			}
		}
	}
}

static id
deviceNewTextureSurface(id self, SEL cmd, id desc, IOSurfaceRef surface, uint64_t plane)
{
	id o;
	struct texstate * st;

	if (surface == NULL || plane != 0) {
		lavlog("newTexture from an IOSurface: plane %llu, pixel format %#x: only plane 0 is served", (unsigned long long)plane,
		    surface != NULL ? IOSurfaceGetPixelFormat(surface) : 0);
		return nil;
	}
	o = deviceNewTexture(self, cmd, desc);
	if (o == nil) {
		return nil;
	}
	st = STATE(o, struct texstate);
	if (IOSurfaceGetBytesPerElement(surface) != st->bpp || IOSurfaceGetWidth(surface) < st->width ||
	    IOSurfaceGetHeight(surface) < st->height) {
		lavlog("newTexture from an IOSurface: the surface (%zux%zu, format %#x, %zu bytes per pixel) does not hold the texture (%ux%u, format %u, %u bytes per pixel)",
		    IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface), IOSurfaceGetPixelFormat(surface), IOSurfaceGetBytesPerElement(surface),
		    st->width, st->height, st->format, st->bpp);
		[o release];
		return nil;
	}
	st->surface = (IOSurfaceRef)CFRetain(surface);
	lavlog("texture over an IOSurface %zux%zu, row %zu bytes", IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface),
	    IOSurfaceGetBytesPerRow(surface));
	return o;
}

static void
cbPresent(id self, __unused SEL _cmd, id drawable)
{
	struct cbstate * st = STATE(self, struct cbstate);

	if (st->drawables == nil) {
		st->drawables = [NSMutableArray new];
	}
	[st->drawables addObject:drawable];
}

static void
cbPresentAt(id self, SEL cmd, id drawable, __unused double t)
{
	cbPresent(self, cmd, drawable);
}

static void
texDealloc(id self, __unused SEL _cmd)
{
	struct texstate * st = STATE(self, struct texstate);
	struct objc_super sup = { self, class_getSuperclass(gTextureClass) };

	pthread_mutex_lock(&gGPULock);
	lavBufFree(&st->b);
	pthread_mutex_unlock(&gGPULock);
	if (st->surface != NULL) {
		CFRelease(st->surface);
	}
	free(st->segPa);
	free(st->segOff);
	free(st->segLen);
	[st->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static IOSurfaceRef texIOSurface(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->surface; }
static uint64_t texWidth(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->width; }
static uint64_t texHeight(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->height; }
static uint64_t texOne(__unused id self, __unused SEL _cmd) { return 1; }
static uint64_t texType(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->type; }
static uint64_t texFormat(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->format; }
static uint64_t texUsage(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->usage; }
static uint64_t texStorage(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->storage; }
static uint64_t texSize(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->b.len; }
static id texLabel(id self, __unused SEL _cmd) { return STATE(self, struct texstate)->label; }

static void
texSetLabel(id self, __unused SEL _cmd, id label)
{
	struct texstate * st = STATE(self, struct texstate);

	[st->label release];
	st->label = [label copy];
}

static BOOL
texRegionOk(struct texstate * st, LavRegion r, uint64_t level)
{
	return level == 0 && r.origin.x + r.size.width <= st->width && r.origin.y + r.size.height <= st->height;
}

static void
texGetBytes(id self, __unused SEL _cmd, void * dst, uint64_t pitch, LavRegion r, uint64_t level)
{
	struct texstate * st = STATE(self, struct texstate);

	if (!texRegionOk(st, r, level)) {
		lavlog("getBytes: region outside the texture");
		return;
	}
	for (uint64_t y = 0; y < r.size.height; y++) {
		memcpy((uint8_t *)dst + y * pitch, (uint8_t *)st->b.cpu + (r.origin.y + y) * st->pitch + r.origin.x * st->bpp, r.size.width * st->bpp);
	}
	texCpuDoneRows(st, r.origin.y, r.origin.y + r.size.height);
}

static void
texGetBytesSlice(id self, SEL cmd, void * dst, uint64_t pitch, __unused uint64_t image, LavRegion r, uint64_t level, __unused uint64_t slice)
{
	texGetBytes(self, cmd, dst, pitch, r, level);
}

static void
texReplace(id self, __unused SEL _cmd, LavRegion r, uint64_t level, const void * src, uint64_t pitch)
{
	struct texstate * st = STATE(self, struct texstate);

	if (!texRegionOk(st, r, level)) {
		lavlog("replaceRegion: region outside the texture");
		return;
	}
	for (uint64_t y = 0; y < r.size.height; y++) {
		memcpy((uint8_t *)st->b.cpu + (r.origin.y + y) * st->pitch + r.origin.x * st->bpp, (const uint8_t *)src + y * pitch, r.size.width * st->bpp);
	}
	texCpuDoneRows(st, r.origin.y, r.origin.y + r.size.height);
}

static void
texReplaceSlice(id self, SEL cmd, LavRegion r, uint64_t level, __unused uint64_t slice, const void * src, uint64_t pitch, __unused uint64_t image)
{
	texReplace(self, cmd, r, level, src, pitch);
}

/* ---- the render pipeline state ---- */

/* the pipeline compiled for one configuration of its fragment constant buffer: the stages, the link and the code */
struct rspec {
	uint64_t key;
	BOOL bad;
	struct lav5_stage vs, fs;
	struct lav5_link link;
	struct lbuf vcode, fcode;
};

#define RSPEC_MAX 64
#define RSPEC_BUFMAX 1024

struct rpipe {
	struct rspec * spec[RSPEC_MAX];         /* specialisations by the bytes that steer the fragment function (rpipeSpecialise) */
	unsigned nspec;
	int fbind;                              /* the fragment constant buffer whose bytes steer it, -1 none, -2 not looked at yet */
	uint32_t fdone;                         /* the buffers already looked at (bit I), so each is analysed once */
	uint8_t * fmask;                        /* 1 where a byte of it steers the control flow (fmasklen bytes) */
	uint32_t fmasklen;
	struct lav5_blend blend[8];
	id vfn, ffn;                            /* retained */
	struct lav5_stage vs, fs;
	struct lav5_link link;
	struct lbuf vcode, fcode;
	uint32_t nattr;
	struct lav5_attr attr[16];
	uint32_t stride[LAV5_MAX_BUFFERS];
	id label;
};

static BOOL
codeBuf(struct lbuf * b, const struct lav5_kernel * k, BOOL locked)
{
	size_t len = (size_t)k->instrlen * 128;
	BOOL ok;

	if (len < (size_t)k->nwords * 4) {
		len = (size_t)k->nwords * 4;
	}
	if (!locked) {
		pthread_mutex_lock(&gGPULock);
	}
	ok = lavBufNew((len + 4095) & ~(size_t)4095, b);
	if (!locked) {
		pthread_mutex_unlock(&gGPULock);
	}
	if (ok) {
		memset(b->cpu, 0, b->len);
		memcpy(b->cpu, k->words, (size_t)k->nwords * 4);
	}
	return ok;
}

/* Metal's MTLVertexFormat to an a5xx one: float, float2, float3, float4, uchar4Normalized */
static BOOL
vertexFormat(uint64_t f, uint32_t * fmt, uint32_t * isfloat)
{
	*isfloat = 1;
	switch (f) {
	case 28: *fmt = 74; return YES;
	case 29: *fmt = 103; return YES;
	case 30: *fmt = 116; return YES;
	case 31: *fmt = 130; return YES;
	case 9: *fmt = 48; return YES;
	case 25: *fmt = 69; return YES;
	case 26: *fmt = 90; return YES;
	case 27: *fmt = 98; return YES;
	}
	return NO;
}

static BOOL
readBlend(id desc, uint64_t index, struct lav5_blend * out)
{
	id ca = msgAt(msg0(desc, "colorAttachments"), "objectAtIndexedSubscript:", index);
	static const uint8_t factor[15] = { 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 16, 12, 13, 14, 15 };
	uint64_t wm = msgU(ca, "writeMask");

	memset(out, 0, sizeof(*out));
	out->writemask = ((wm >> 3) & 1) | (((wm >> 2) & 1) << 1) | (((wm >> 1) & 1) << 2) | ((wm & 1) << 3);
	out->enable = msgU(ca, "isBlendingEnabled") & 0xff ? 1 : 0;
	if (out->enable) {
		uint64_t f[4] = { msgU(ca, "sourceRGBBlendFactor"), msgU(ca, "destinationRGBBlendFactor"),
		    msgU(ca, "sourceAlphaBlendFactor"), msgU(ca, "destinationAlphaBlendFactor") };
		uint64_t ops[2] = { msgU(ca, "rgbBlendOperation"), msgU(ca, "alphaBlendOperation") };

		for (int i = 0; i < 4; i++) {
			if (f[i] >= 15) {
				lavlog("render pipeline: blend factor %llu is not served", (unsigned long long)f[i]);
				return NO;
			}
		}
		for (int i = 0; i < 2; i++) {
			if (ops[i] > 4) {
				lavlog("render pipeline: blend operation %llu is not served", (unsigned long long)ops[i]);
				return NO;
			}
		}
		out->rgb_src = factor[f[0]];
		out->rgb_dst = factor[f[1]];
		out->a_src = factor[f[2]];
		out->a_dst = factor[f[3]];
		out->rgb_op = (uint32_t)ops[0];
		out->a_op = (uint32_t)ops[1];
	}
	return YES;
}

/*
 * Compiling a pipeline takes seconds (QuartzCore's fixed_frag_lph_cph is 26000 words), and backboardd needs dozens at start. The
 * compiled stages are kept in /var/tmp/lavmetal-cache, one file per vertex/fragment pair, keyed by the two metallibs and the
 * function names and by this plug-in build (the compiler changes with it).
 */
#define PIPE_CACHE_DIR "/var/tmp/lavmetal-cache"
#define PIPE_CACHE_MAGIC 0x4c564331u

static void
pipeCachePath(struct fnstate * vs, struct fnstate * fs, char * path, size_t n)
{
	uint64_t h = 0xcbf29ce484222325ull;
	const char * stamp = __DATE__ __TIME__;

	h = lavHash64(h, stamp, strlen(stamp));
	h = lavHash64(h, &STATE(vs->library, struct libstate)->hash, 8);
	h = lavHash64(h, &STATE(fs->library, struct libstate)->hash, 8);
	h = lavHash64(h, [vs->name UTF8String], strlen([vs->name UTF8String]) + 1);
	h = lavHash64(h, [fs->name UTF8String], strlen([fs->name UTF8String]) + 1);
	h = lavHash64(h, &vs->nconsts, sizeof(vs->nconsts));
	if (vs->nconsts != 0) {
		h = lavHash64(h, vs->consts, vs->nconsts * sizeof(*vs->consts));
	}
	h = lavHash64(h, &fs->nconsts, sizeof(fs->nconsts));
	if (fs->nconsts != 0) {
		h = lavHash64(h, fs->consts, fs->nconsts * sizeof(*fs->consts));
	}
	snprintf(path, n, PIPE_CACHE_DIR "/%016llx.pipe", (unsigned long long)h);
}

static BOOL
pipeReadStage(FILE * f, struct lav5_stage * st)
{
	uint32_t * w, * imm;
	uint32_t nw, ni;

	if (fread(st, sizeof(*st), 1, f) != 1) {
		return NO;
	}
	nw = st->k.nwords;
	ni = st->k.nimm;
	if (nw == 0 || nw > (1u << 20) || ni > (1u << 16)) {
		return NO;
	}
	w = malloc(nw * 4);
	imm = ni != 0 ? malloc(ni * 4) : NULL;
	if (fread(w, 4, nw, f) != nw || (ni != 0 && fread(imm, 4, ni, f) != ni)) {
		free(w);
		free(imm);
		return NO;
	}
	st->k.words = w;
	st->k.imm = imm;
	return YES;
}

static BOOL
pipeCacheLoad(struct fnstate * vs, struct fnstate * fs, struct rpipe * rp)
{
	char path[160];
	uint32_t magic = 0;
	FILE * f;
	BOOL ok;

	pipeCachePath(vs, fs, path, sizeof(path));
	f = fopen(path, "rb");
	if (f == NULL) {
		return NO;
	}
	ok = fread(&magic, 4, 1, f) == 1 && magic == PIPE_CACHE_MAGIC && fread(&rp->link, sizeof(rp->link), 1, f) == 1 &&
	    pipeReadStage(f, &rp->vs) && pipeReadStage(f, &rp->fs);
	fclose(f);
	if (!ok) {
		memset(&rp->vs, 0, sizeof(rp->vs));
		memset(&rp->fs, 0, sizeof(rp->fs));
		return NO;
	}
	lavlog("pipeline cache: %s + %s", [vs->name UTF8String], [fs->name UTF8String]);
	return YES;
}

static void
pipeWriteStage(FILE * f, const struct lav5_stage * st)
{
	struct lav5_stage c = *st;

	c.k.words = NULL;
	c.k.imm = NULL;
	fwrite(&c, sizeof(c), 1, f);
	fwrite(st->k.words, 4, st->k.nwords, f);
	if (st->k.nimm != 0) {
		fwrite(st->k.imm, 4, st->k.nimm, f);
	}
}

static void
pipeCacheStore(struct fnstate * vs, struct fnstate * fs, struct rpipe * rp)
{
	char path[160], tmp[176];
	uint32_t magic = PIPE_CACHE_MAGIC;
	FILE * f;

	mkdir(PIPE_CACHE_DIR, 01777);
	chmod(PIPE_CACHE_DIR, 01777);
	pipeCachePath(vs, fs, path, sizeof(path));
	snprintf(tmp, sizeof(tmp), "%s.%d", path, (int)getpid());
	f = fopen(tmp, "wb");
	if (f == NULL) {
		return;
	}
	fwrite(&magic, 4, 1, f);
	fwrite(&rp->link, sizeof(rp->link), 1, f);
	pipeWriteStage(f, &rp->vs);
	pipeWriteStage(f, &rp->fs);
	if (fclose(f) == 0) {
		rename(tmp, path);
	} else {
		unlink(tmp);
	}
}

/* The bytes of buffer I of draw D's fragment stage: CPU address and how many (0, NULL: not bound). */
static const uint8_t *
fragBytes(struct draw * d, int i, uint32_t * len)
{
	*len = 0;
	if (d->fbuf[i] != nil) {
		struct bufstate * b = STATE(d->fbuf[i], struct bufstate);

		if (d->foff[i] >= b->b.len) {
			return NULL;
		}
		*len = (uint32_t)(b->b.len - d->foff[i] > RSPEC_BUFMAX ? RSPEC_BUFMAX : b->b.len - d->foff[i]);
		return (const uint8_t *)b->b.cpu + d->foff[i];
	}
	if (d->fbytes[i] != NULL) {
		*len = d->fblen[i] > RSPEC_BUFMAX ? RSPEC_BUFMAX : d->fblen[i];
		return d->fbytes[i];
	}
	return NULL;
}

/* The analysis of which bytes of a constant buffer steer a fragment function takes ~0.5 s and gives the same answer for the same function,
 * library, constants and buffer: kept for the whole process, so a second pipeline of the same function does not repeat it. */
#define CTL_MAX 256
static struct ctlent {
	uint64_t key;
	uint32_t len;
	unsigned cnt;
	uint8_t * mask;
} gCtl[CTL_MAX];
static unsigned gNCtl;

static uint64_t
ctlKey(struct fnstate * ff, int buf, uint32_t len)
{
	uint64_t h = 0xcbf29ce484222325ull;
	const char * stamp = __DATE__ __TIME__;

	h = lavHash64(h, stamp, strlen(stamp));
	h = lavHash64(h, &STATE(ff->library, struct libstate)->hash, 8);
	h = lavHash64(h, [ff->name UTF8String], strlen([ff->name UTF8String]) + 1);
	h = lavHash64(h, &ff->nconsts, sizeof(ff->nconsts));
	if (ff->nconsts != 0) {
		h = lavHash64(h, ff->consts, ff->nconsts * sizeof(*ff->consts));
	}
	h = lavHash64(h, &buf, sizeof(buf));
	return lavHash64(h, &len, sizeof(len));
}

static BOOL
ctlCacheGet(struct fnstate * ff, int buf, uint32_t len, uint8_t * mask, unsigned * cnt)
{
	uint64_t k = ctlKey(ff, buf, len);

	char path[160];
	FILE * f;
	uint32_t hdr[3] = { 0, 0, 0 };
	BOOL ok;

	for (unsigned i = 0; i < gNCtl; i++) {
		if (gCtl[i].key == k && gCtl[i].len == len) {
			memcpy(mask, gCtl[i].mask, len);
			*cnt = gCtl[i].cnt;
			return YES;
		}
	}
	/* a previous run of backboardd may have done the analysis */
	snprintf(path, sizeof(path), PIPE_CACHE_DIR "/%016llx.ctl", (unsigned long long)k);
	f = fopen(path, "rb");
	if (f == NULL) {
		return NO;
	}
	ok = fread(hdr, 4, 3, f) == 3 && hdr[0] == PIPE_CACHE_MAGIC && hdr[1] == len && fread(mask, 1, len, f) == len;
	fclose(f);
	if (!ok) {
		return NO;
	}
	*cnt = hdr[2];
	if (gNCtl < CTL_MAX) {
		gCtl[gNCtl].key = k;
		gCtl[gNCtl].len = len;
		gCtl[gNCtl].cnt = hdr[2];
		gCtl[gNCtl].mask = malloc(len);
		memcpy(gCtl[gNCtl].mask, mask, len);
		gNCtl++;
	}
	lavlog("control bytes from disk cache: %s buffer %d (%u bytes, %u steer)", [ff->name UTF8String], buf, len, hdr[2]);
	return YES;
}

static void
ctlCachePut(struct fnstate * ff, int buf, uint32_t len, const uint8_t * mask, unsigned cnt)
{
	uint64_t k = ctlKey(ff, buf, len);
	char path[160], tmp[176];
	FILE * f;

	mkdir(PIPE_CACHE_DIR, 01777);
	chmod(PIPE_CACHE_DIR, 01777);
	snprintf(path, sizeof(path), PIPE_CACHE_DIR "/%016llx.ctl", (unsigned long long)k);
	snprintf(tmp, sizeof(tmp), "%s.%d", path, (int)getpid());
	f = fopen(tmp, "wb");
	if (f != NULL) {
		uint32_t hdr[3] = { PIPE_CACHE_MAGIC, len, cnt };

		fwrite(hdr, 4, 3, f);
		fwrite(mask, 1, len, f);
		if (fclose(f) == 0) {
			rename(tmp, path);
		} else {
			unlink(tmp);
		}
	}
	if (gNCtl >= CTL_MAX) {
		return;
	}
	gCtl[gNCtl].key = k;
	gCtl[gNCtl].len = len;
	gCtl[gNCtl].cnt = cnt;
	gCtl[gNCtl].mask = malloc(len);
	memcpy(gCtl[gNCtl].mask, mask, len);
	gNCtl++;
}

/* A specialisation is also kept on disk, by pipeline functions, library hashes, constants, steering buffer and the steering bytes. */
static void
specCachePath(struct fnstate * vf, struct fnstate * ff, int fbind, uint64_t key, char * path, size_t n)
{
	uint64_t h = 0xcbf29ce484222325ull;
	const char * stamp = __DATE__ __TIME__;

	h = lavHash64(h, stamp, strlen(stamp));
	h = lavHash64(h, &STATE(vf->library, struct libstate)->hash, 8);
	h = lavHash64(h, &STATE(ff->library, struct libstate)->hash, 8);
	h = lavHash64(h, [vf->name UTF8String], strlen([vf->name UTF8String]) + 1);
	h = lavHash64(h, [ff->name UTF8String], strlen([ff->name UTF8String]) + 1);
	h = lavHash64(h, &vf->nconsts, sizeof(vf->nconsts));
	if (vf->nconsts != 0) {
		h = lavHash64(h, vf->consts, vf->nconsts * sizeof(*vf->consts));
	}
	h = lavHash64(h, &ff->nconsts, sizeof(ff->nconsts));
	if (ff->nconsts != 0) {
		h = lavHash64(h, ff->consts, ff->nconsts * sizeof(*ff->consts));
	}
	h = lavHash64(h, &fbind, sizeof(fbind));
	h = lavHash64(h, &key, sizeof(key));
	snprintf(path, n, PIPE_CACHE_DIR "/%016llx.spec", (unsigned long long)h);
}

/* YES: sp is filled from disk (its bad flag too), the code buffers are not made yet. */
static BOOL
specCacheLoad(const char * path, struct rspec * sp)
{
	FILE * f = fopen(path, "rb");
	uint32_t hdr[2] = { 0, 0 };
	BOOL ok;

	if (f == NULL) {
		return NO;
	}
	ok = fread(hdr, 4, 2, f) == 2 && hdr[0] == PIPE_CACHE_MAGIC + 1;
	if (ok && hdr[1] == 0) {
		ok = fread(&sp->link, sizeof(sp->link), 1, f) == 1 && pipeReadStage(f, &sp->vs) && pipeReadStage(f, &sp->fs);
	}
	fclose(f);
	if (ok) {
		sp->bad = hdr[1] != 0;
	} else {
		memset(&sp->vs, 0, sizeof(sp->vs));
		memset(&sp->fs, 0, sizeof(sp->fs));
	}
	return ok;
}

static void
specCacheStore(const char * path, const struct rspec * sp)
{
	char tmp[200];
	uint32_t hdr[2] = { PIPE_CACHE_MAGIC + 1, sp->bad ? 1u : 0u };
	FILE * f;

	mkdir(PIPE_CACHE_DIR, 01777);
	chmod(PIPE_CACHE_DIR, 01777);
	snprintf(tmp, sizeof(tmp), "%s.%d", path, (int)getpid());
	f = fopen(tmp, "wb");
	if (f == NULL) {
		return;
	}
	fwrite(hdr, 4, 2, f);
	if (!sp->bad) {
		fwrite(&sp->link, sizeof(sp->link), 1, f);
		pipeWriteStage(f, &sp->vs);
		pipeWriteStage(f, &sp->fs);
	}
	if (fclose(f) == 0) {
		rename(tmp, path);
	} else {
		unlink(tmp);
	}
}

/* CoreAnimation's uber fragment shader chooses its path through ~15 000 instructions by the few bytes of its configuration buffer, and
 * run for every pixel that is the slow part (3 us a pixel here). Compiled for the configuration of a draw it is a few dozen: so the
 * stages are compiled once per distinct configuration (the bytes the control flow depends on) and kept with the pipeline. NULL: the
 * pipeline's own stages are to be used. */
static const struct rspec *
rpipeSpecialise(struct rpipe * rp, struct draw * d)
{
	struct fnstate * vf, * ff;
	const uint8_t * data = NULL;
	uint32_t len = 0;
	uint64_t key = 0xcbf29ce484222325ull;
	struct rspec * sp = NULL;
	char why[256] = "";

	if (rp->vfn == nil || rp->ffn == nil || rp->fbind == -1) {
		return NULL;
	}
	vf = STATE(rp->vfn, struct fnstate);
	ff = STATE(rp->ffn, struct fnstate);
	pthread_mutex_lock(&gCCLock);
	if (rp->fbind == -2) {
		/* which buffer steers the function, and which of its bytes: asked once, at the first draw that binds it */
		rp->fbind = -2;
		for (int i = 0; i < LAV5_MAX_BUFFERS && rp->fbind == -2; i++) {
			uint8_t * mask;
			uint64_t t0;
			unsigned cnt = 0;
			BOOL got = NO;

			if (!(rp->fs.k.buffer_mask & (1u << i)) || (rp->fdone & (1u << i))) {
				continue;
			}
			data = fragBytes(d, i, &len);
			if (data == NULL || len == 0) {
				continue;
			}
			mask = calloc(1, len);
			got = ctlCacheGet(ff, i, len, mask, &cnt);
			if (!got) {
				t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
				if (lavcc_control_bytes(STATE(ff->library, struct libstate)->lib, [ff->name UTF8String], ff->consts, ff->nconsts,
				    (unsigned)i, len, mask, why, sizeof(why))) {
					for (uint32_t k = 0; k < len; k++) {
						cnt += mask[k] != 0;
					}
					got = YES;
					ctlCachePut(ff, i, len, mask, cnt);
					lavlog("pipeline %s: buffer %d (%u bytes) has %u bytes that steer the fragment function (%llu ms)",
					    [ff->name UTF8String], i, len, cnt, (unsigned long long)((clock_gettime_nsec_np(CLOCK_MONOTONIC) - t0) / 1000000u));
				} else {
					lavlog("pipeline %s: buffer %d: control bytes refused: %s", [ff->name UTF8String], i, why);
				}
			}
			rp->fdone |= 1u << i;
			if (got && cnt != 0) {
				rp->fbind = i;
				rp->fmask = mask;
				rp->fmasklen = len;
				break;
			}
			free(mask);
		}
		if (rp->fbind == -2) {
			if ((rp->fdone & rp->fs.k.buffer_mask) == rp->fs.k.buffer_mask) {
				rp->fbind = -1;         /* every buffer it reads was looked at and none steers it: never ask again */
			}
			pthread_mutex_unlock(&gCCLock);
			return NULL;            /* otherwise asked again at the next draw: a buffer was not bound yet */
		}
	}
	data = fragBytes(d, rp->fbind, &len);
	if (data == NULL || len < rp->fmasklen) {
		pthread_mutex_unlock(&gCCLock);
		return NULL;
	}
	for (uint32_t k = 0; k < rp->fmasklen; k++) {
		uint8_t b = rp->fmask[k] ? data[k] : 0;

		key = lavHash64(key, &b, 1);
	}
	for (unsigned i = 0; i < rp->nspec; i++) {
		if (rp->spec[i]->key == key) {
			sp = rp->spec[i];
			break;
		}
	}
	if (sp == NULL) {
		uint64_t t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
		uint8_t * fold;
		char spath[160];

		if (rp->nspec >= RSPEC_MAX) {
			pthread_mutex_unlock(&gCCLock);
			return NULL;
		}
		sp = calloc(1, sizeof(*sp));
		sp->key = key;
		specCachePath(vf, ff, rp->fbind, key, spath, sizeof(spath));
		if (specCacheLoad(spath, sp)) {
			if (!sp->bad && (!codeBuf(&sp->vcode, &sp->vs.k, YES) || !codeBuf(&sp->fcode, &sp->fs.k, YES))) {
				sp->bad = YES;
			}
			lavlog("pipeline %s: specialisation %016llx from disk cache%s", [ff->name UTF8String], (unsigned long long)key,
			    sp->bad ? " (refused)" : "");
			rp->spec[rp->nspec++] = sp;
			pthread_mutex_unlock(&gCCLock);
			return sp->bad ? NULL : sp;
		}
		fold = malloc(rp->fmasklen);
		for (uint32_t k = 0; k < rp->fmasklen; k++) {
			fold[k] = rp->fmask[k] != 0;
		}
		if (!lavcc_compile_program_known(STATE(vf->library, struct libstate)->lib, [vf->name UTF8String], vf->consts, vf->nconsts,
		    STATE(ff->library, struct libstate)->lib, [ff->name UTF8String], ff->consts, ff->nconsts, (unsigned)rp->fbind, data, fold,
		    rp->fmasklen, &sp->vs, &sp->fs, &sp->link, why, sizeof(why)) ||
		    !codeBuf(&sp->vcode, &sp->vs.k, YES) || !codeBuf(&sp->fcode, &sp->fs.k, YES)) {
			lavlog("pipeline %s: no specialisation for configuration %016llx: %s", [ff->name UTF8String], (unsigned long long)key, why);
			sp->bad = YES;
			specCacheStore(spath, sp);
		} else {
			specCacheStore(spath, sp);
			lavlog("pipeline %s: specialised for configuration %016llx: fragment %u words (was %u), %u ms", [ff->name UTF8String],
			    (unsigned long long)key, sp->fs.k.nwords, rp->fs.k.nwords, (unsigned)((clock_gettime_nsec_np(CLOCK_MONOTONIC) - t0) / 1000000u));
		}
		free(fold);
		rp->spec[rp->nspec++] = sp;
	}
	pthread_mutex_unlock(&gCCLock);
	return sp->bad ? NULL : sp;
}

static id
makeRenderPipeline(id desc, NSError ** err)
{
	id vf = msg0(desc, "vertexFunction"), ff = msg0(desc, "fragmentFunction");
	id vd = msg0(desc, "vertexDescriptor");
	struct fnstate * vs, * fs;
	struct rpipe * rp;
	char why[256] = "";
	id o;
	int ok;

	if (vf == nil || ff == nil || !lavOpen()) {
		lavError(err, @"MTLLibraryErrorDomain", "a render pipeline needs a vertex and a fragment function");
		return nil;
	}
	vs = STATE(vf, struct fnstate);
	fs = STATE(ff, struct fnstate);
	if (vs->type != 0 || fs->type != 1) {
		lavError(err, @"MTLLibraryErrorDomain", "wrong kinds of functions");
		return nil;
	}
	o = makeObject(gRenderPipeClass, sizeof(struct rpipe));
	if (o == nil) {
		return nil;
	}
	rp = STATE(o, struct rpipe);
	memset(rp, 0, sizeof(*rp));
	rp->fbind = -2;
	pthread_mutex_lock(&gCCLock);
	ok = pipeCacheLoad(vs, fs, rp);
	if (!ok) {
		ok = lavcc_compile_program_c(STATE(vs->library, struct libstate)->lib, [vs->name UTF8String], vs->consts, vs->nconsts,
		    STATE(fs->library, struct libstate)->lib, [fs->name UTF8String], fs->consts, fs->nconsts, &rp->vs, &rp->fs, &rp->link, why,
		    sizeof(why));
		if (ok) {
			pipeCacheStore(vs, fs, rp);
		}
	}
	pthread_mutex_unlock(&gCCLock);
	if (!ok) {
		lavlog("render pipeline %s + %s: the compiler refused it: %s", [vs->name UTF8String], [fs->name UTF8String], why);
		lavError(err, @"MTLLibraryErrorDomain", why);
		[o release];
		return nil;
	}
	/* the attributes the vertex function reads, from the vertex descriptor */
	for (uint32_t j = 0; j < rp->vs.nin; j++) {
		const struct lav5_io * in = &rp->vs.in[j];
		id a, l;
		uint32_t fmt, isfloat, buffer;

		if (in->sysval || in->compmask == 0 || in->slot < 15) {
			continue;
		}
		if (rp->nattr >= 16 || vd == nil) {
			lavlog("render pipeline: attribute %u is read but the vertex descriptor has none", in->slot - 15);
			goto refuse;
		}
		a = msgAt(msg0(vd, "attributes"), "objectAtIndexedSubscript:", in->slot - 15);
		buffer = (uint32_t)msgU(a, "bufferIndex");
		if (!vertexFormat(msgU(a, "format"), &fmt, &isfloat) || buffer >= LAV5_MAX_BUFFERS) {
			lavlog("render pipeline: attribute %u has vertex format %llu (not served) or buffer %u", in->slot - 15,
			    (unsigned long long)msgU(a, "format"), buffer);
			goto refuse;
		}
		rp->attr[rp->nattr++] = (struct lav5_attr){ in->slot - 15, fmt, (uint32_t)msgU(a, "offset"), buffer, isfloat, 0 };
		l = msgAt(msg0(vd, "layouts"), "objectAtIndexedSubscript:", buffer);
		if (msgU(l, "stepFunction") != 1) {
			lavlog("render pipeline: buffer %u steps per instance or constant: not served", buffer);
			goto refuse;
		}
		rp->stride[buffer] = (uint32_t)msgU(l, "stride");
	}
	if (rp->nattr != rp->vs.nattr) {
		lavlog("render pipeline: the vertex function has %u attributes, the descriptor gave %u", rp->vs.nattr, rp->nattr);
		goto refuse;
	}
	if (!codeBuf(&rp->vcode, &rp->vs.k, NO) || !codeBuf(&rp->fcode, &rp->fs.k, NO)) {
		lavError(err, @"MTLLibraryErrorDomain", "no memory for the code");
		[o release];
		return nil;
	}
	/* blending and the write mask, from each colour attachment */
	for (uint64_t ai = 0; ai < 8; ai++) {
		if (!readBlend(desc, ai, &rp->blend[ai])) {
			goto refuse;
		}
	}
	rp->vfn = [vf retain];
	rp->ffn = [ff retain];
	lavlog("render pipeline %s + %s: %u+%u words, %u attributes", [vs->name UTF8String], [fs->name UTF8String], rp->vs.k.nwords,
	    rp->fs.k.nwords, rp->nattr);
	return o;
refuse:
	lavError(err, @"MTLLibraryErrorDomain", "the render pipeline uses something this driver does not serve (see the log)");
	[o release];
	return nil;
}

static id
deviceNewRenderPipeline(__unused id self, __unused SEL _cmd, id desc, NSError ** err)
{
	return makeRenderPipeline(desc, err);
}

static id
deviceNewRenderPipelineOptions(__unused id self, __unused SEL _cmd, id desc, __unused uint64_t options, id * reflection, NSError ** err)
{
	if (reflection != NULL) {
		*reflection = nil;
	}
	return makeRenderPipeline(desc, err);
}

static void
deviceNewRenderPipelineAsync(__unused id self, __unused SEL _cmd, id desc, id handler)
{
	NSError * err = nil;
	id p = makeRenderPipeline(desc, &err);

	((LavPipelineDone)handler)(p, err);
	[p release];
}

static void
rpDealloc(id self, __unused SEL _cmd)
{
	struct rpipe * st = STATE(self, struct rpipe);
	struct objc_super sup = { self, class_getSuperclass(gRenderPipeClass) };

	lavlog("render pipeline freed (code handles %u, %u)", st->vcode.handle, st->fcode.handle);
	pthread_mutex_lock(&gGPULock);
	if (st->vcode.handle != 0) {
		lavBufFree(&st->vcode);
	}
	if (st->fcode.handle != 0) {
		lavBufFree(&st->fcode);
	}
	pthread_mutex_unlock(&gGPULock);
	lavcc_stage_free(&st->vs);
	lavcc_stage_free(&st->fs);
	[st->vfn release];
	[st->ffn release];
	[st->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static id rpLabel(id self, __unused SEL _cmd) { return STATE(self, struct rpipe)->label; }

static void
rpSetLabel(id self, __unused SEL _cmd, id label)
{
	struct rpipe * st = STATE(self, struct rpipe);

	[st->label release];
	st->label = [label copy];
}

/* ---- the render encoder ---- */

struct renc {
	id commandBuffer;                       /* retained */
	BOOL ended;
	id label;
	id target;                              /* retained */
	id xtarget[8];                          /* retained: colour targets 1.. */
	id pipeline;                            /* retained */
	id vbuf[LAV5_MAX_BUFFERS];
	uint64_t voff[LAV5_MAX_BUFFERS];
	uint8_t * vbytes[LAV5_MAX_BUFFERS];
	uint32_t vblen[LAV5_MAX_BUFFERS];
	id fbuf[LAV5_MAX_BUFFERS];
	uint64_t foff[LAV5_MAX_BUFFERS];
	uint8_t * fbytes[LAV5_MAX_BUFFERS];
	uint32_t fblen[LAV5_MAX_BUFFERS];
	float viewport[6];
	uint32_t scissor[4];
	uint32_t cull, front_cw;
	id ftex[16], fsamp[16];
	float blend_color[4];
};

static id
cbRenderEncoder(id self, __unused SEL _cmd, id passdesc)
{
	id atts = msg0(passdesc, "colorAttachments");
	id att = msgAt(atts, "objectAtIndexedSubscript:", 0);
	id tex = msg0(att, "texture");
	id e;
	struct renc * re;
	struct texstate * ts;

	if (tex == nil || object_getClass(tex) != gTextureClass) {
		lavlog("renderCommandEncoder: colour attachment 0 is not a texture of this driver (drawables come with M5.5)");
		return nil;
	}
	if (msg0(msg0(passdesc, "depthAttachment"), "texture") != nil || msg0(msg0(passdesc, "stencilAttachment"), "texture") != nil) {
		lavlog("renderCommandEncoder: depth and stencil attachments are not served (ignored)");
	}
	e = makeObject(gRenderClass, sizeof(struct renc));
	if (e == nil) {
		return nil;
	}
	re = STATE(e, struct renc);
	memset(re, 0, sizeof(*re));
	re->commandBuffer = [self retain];
	re->target = [tex retain];
	ts = STATE(tex, struct texstate);
	for (uint64_t i = 1; i < 8; i++) {
		id a = msgAt(atts, "objectAtIndexedSubscript:", i), t = msg0(a, "texture");

		if (t == nil) {
			continue;
		}
		if (object_getClass(t) != gTextureClass || STATE(t, struct texstate)->width != ts->width ||
		    STATE(t, struct texstate)->height != ts->height) {
			lavlog("renderCommandEncoder: colour attachment %llu is not a texture of this driver of the same size as attachment 0", (unsigned long long)i);
			continue;
		}
		re->xtarget[i] = [t retain];
		if (msgU(a, "loadAction") == 2) {
			struct op * cop = cbAddOp(STATE(self, struct cbstate));
			LavClear c = ((LavClear (*)(id, SEL))objc_msgSend)(a, sel_registerName("clearColor"));

			cop->kind = OP_CLEAR;
			cop->dst = [t retain];
			cop->clear[0] = c.r;
			cop->clear[1] = c.g;
			cop->clear[2] = c.b;
			cop->clear[3] = c.a;
		}
	}
	re->viewport[2] = (float)ts->width;
	re->viewport[3] = (float)ts->height;
	re->viewport[5] = 1;
	re->scissor[2] = ts->width;
	re->scissor[3] = ts->height;
	re->front_cw = 1;
	if (msgU(att, "loadAction") == 1 && ts->surface != NULL && !ts->synced) {
		surfaceCopy(ts, NO);
	}
	if (msgU(att, "loadAction") == 2) {
		struct op * op = cbAddOp(STATE(self, struct cbstate));
		LavClear c = ((LavClear (*)(id, SEL))objc_msgSend)(att, sel_registerName("clearColor"));

		op->kind = OP_CLEAR;
		op->dst = [tex retain];
		op->clear[0] = c.r;
		op->clear[1] = c.g;
		op->clear[2] = c.b;
		op->clear[3] = c.a;
	}
	return [e autorelease];
}

static void
rencDealloc(id self, __unused SEL _cmd)
{
	struct renc * re = STATE(self, struct renc);
	struct objc_super sup = { self, class_getSuperclass(gRenderClass) };

	for (int i = 0; i < LAV5_MAX_BUFFERS; i++) {
		[re->vbuf[i] release];
		[re->fbuf[i] release];
		free(re->vbytes[i]);
		free(re->fbytes[i]);
	}
	for (int i = 0; i < 16; i++) {
		[re->ftex[i] release];
		[re->fsamp[i] release];
	}
	for (int i = 0; i < 8; i++) {
		[re->xtarget[i] release];
	}
	[re->pipeline release];
	[re->target release];
	[re->commandBuffer release];
	[re->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static void
rencSetPipeline(id self, __unused SEL _cmd, id pipeline)
{
	struct renc * re = STATE(self, struct renc);

	[pipeline retain];
	[re->pipeline release];
	re->pipeline = pipeline;
}

static void
rencBind(id * bufs, uint64_t * offs, uint8_t ** bytes, uint32_t * blen, id buffer, uint64_t offset, uint64_t index)
{
	if (index >= LAV5_MAX_BUFFERS) {
		lavlog("setBuffer at index %llu: the table holds %d", (unsigned long long)index, LAV5_MAX_BUFFERS);
		return;
	}
	[buffer retain];
	[bufs[index] release];
	bufs[index] = buffer;
	offs[index] = offset;
	free(bytes[index]);
	bytes[index] = NULL;
	blen[index] = 0;
}

static void
rencBindBytes(id * bufs, uint8_t ** bytes, uint32_t * blen, const void * data, uint64_t length, uint64_t index)
{
	if (index >= LAV5_MAX_BUFFERS || length == 0 || length > 4096) {
		lavlog("setBytes of %llu at index %llu refused", (unsigned long long)length, (unsigned long long)index);
		return;
	}
	[bufs[index] release];
	bufs[index] = nil;
	free(bytes[index]);
	bytes[index] = malloc((size_t)length);
	memcpy(bytes[index], data, (size_t)length);
	blen[index] = (uint32_t)length;
}

static void
rencSetVBuffer(id self, __unused SEL _cmd, id buffer, uint64_t offset, uint64_t index)
{
	struct renc * re = STATE(self, struct renc);

	rencBind(re->vbuf, re->voff, re->vbytes, re->vblen, buffer, offset, index);
}

static void
rencSetFBuffer(id self, __unused SEL _cmd, id buffer, uint64_t offset, uint64_t index)
{
	struct renc * re = STATE(self, struct renc);

	rencBind(re->fbuf, re->foff, re->fbytes, re->fblen, buffer, offset, index);
}

static void
rencSetVOffset(id self, __unused SEL _cmd, uint64_t offset, uint64_t index)
{
	if (index < LAV5_MAX_BUFFERS) {
		STATE(self, struct renc)->voff[index] = offset;
	}
}

static void
rencSetFOffset(id self, __unused SEL _cmd, uint64_t offset, uint64_t index)
{
	if (index < LAV5_MAX_BUFFERS) {
		STATE(self, struct renc)->foff[index] = offset;
	}
}

static void
rencSetVBytes(id self, __unused SEL _cmd, const void * data, uint64_t length, uint64_t index)
{
	struct renc * re = STATE(self, struct renc);

	rencBindBytes(re->vbuf, re->vbytes, re->vblen, data, length, index);
}

static void
rencSetFBytes(id self, __unused SEL _cmd, const void * data, uint64_t length, uint64_t index)
{
	struct renc * re = STATE(self, struct renc);

	rencBindBytes(re->fbuf, re->fbytes, re->fblen, data, length, index);
}

static void
rencSetViewport(id self, __unused SEL _cmd, LavViewport v)
{
	struct renc * re = STATE(self, struct renc);

	re->viewport[0] = (float)v.x;
	re->viewport[1] = (float)v.y;
	re->viewport[2] = (float)v.w;
	re->viewport[3] = (float)v.h;
	re->viewport[4] = (float)v.znear;
	re->viewport[5] = (float)v.zfar;
}

static void
rencSetScissor(id self, __unused SEL _cmd, LavScissor r)
{
	struct renc * re = STATE(self, struct renc);

	re->scissor[0] = (uint32_t)r.x;
	re->scissor[1] = (uint32_t)r.y;
	re->scissor[2] = (uint32_t)r.w;
	re->scissor[3] = (uint32_t)r.h;
}

static void
rencSetFTexture(id self, __unused SEL _cmd, id tex, uint64_t index)
{
	struct renc * re = STATE(self, struct renc);

	if (index >= 16) {
		lavlog("setFragmentTexture at index %llu: the table holds 16", (unsigned long long)index);
		return;
	}
	if (tex != nil && object_getClass(tex) != gTextureClass) {
		lavlog("setFragmentTexture: not a texture of this driver");
		return;
	}
	[tex retain];
	[re->ftex[index] release];
	re->ftex[index] = tex;
}

static void
rencSetFSampler(id self, __unused SEL _cmd, id samp, uint64_t index)
{
	struct renc * re = STATE(self, struct renc);

	if (index >= 16) {
		lavlog("setFragmentSamplerState at index %llu: the table holds 16", (unsigned long long)index);
		return;
	}
	[samp retain];
	[re->fsamp[index] release];
	re->fsamp[index] = samp;
}

static void
rencSetFSamplerLod(id self, SEL cmd, id samp, __unused float lo, __unused float hi, uint64_t index)
{
	rencSetFSampler(self, cmd, samp, index);
}

static void
rencSetFTextures(id self, SEL cmd, const id * texs, NSRange range)
{
	for (NSUInteger i = 0; i < range.length; i++) {
		rencSetFTexture(self, cmd, texs[i], range.location + i);
	}
}

static void
rencSetFSamplers(id self, SEL cmd, const id * samps, NSRange range)
{
	for (NSUInteger i = 0; i < range.length; i++) {
		rencSetFSampler(self, cmd, samps[i], range.location + i);
	}
}

static void
rencSetVTexture(__unused id self, __unused SEL _cmd, __unused id tex, __unused uint64_t index)
{
	lavlog("setVertexTexture: sampling in the vertex stage is not served");
}

static void
rencSetBlendColor(id self, __unused SEL _cmd, float r, float g, float b, float a)
{
	struct renc * re = STATE(self, struct renc);

	re->blend_color[0] = r;
	re->blend_color[1] = g;
	re->blend_color[2] = b;
	re->blend_color[3] = a;
}

/* ---- samplers ---- */

static id
deviceNewSampler(__unused id self, __unused SEL _cmd, id desc)
{
	id o = makeObject(gSamplerClass, sizeof(struct lav5_samp));
	struct lav5_samp * sp;
	uint64_t mode[2] = { msgU(desc, "sAddressMode"), msgU(desc, "tAddressMode") };
	uint32_t wrap[2];

	if (o == nil) {
		return nil;
	}
	for (int i = 0; i < 2; i++) {
		wrap[i] = mode[i] == 2 ? 0 : mode[i] == 3 ? 2 : 1;
		if (mode[i] > 3) {
			lavlog("newSamplerState: address mode %llu approximated by clamp to edge", (unsigned long long)mode[i]);
		}
	}
	sp = STATE(o, struct lav5_samp);
	memset(sp, 0, sizeof(*sp));
	sp->mag = msgU(desc, "magFilter") == 1;
	sp->min = msgU(desc, "minFilter") == 1;
	sp->mip_linear = msgU(desc, "mipFilter") == 2;
	sp->wrap_s = wrap[0];
	sp->wrap_t = wrap[1];
	sp->valid = 1;
	if (!(msgU(desc, "normalizedCoordinates") & 0xff)) {
		lavlog("newSamplerState: unnormalized coordinates are not served (ignored)");
	}
	return o;
}

static id sampLabel(__unused id self, __unused SEL _cmd) { return nil; }
static void sampSetLabel(__unused id self, __unused SEL _cmd, __unused id label) { }

static void
rencSetCull(id self, __unused SEL _cmd, uint64_t mode)
{
	STATE(self, struct renc)->cull = (uint32_t)mode;
}

static void
rencSetWinding(id self, __unused SEL _cmd, uint64_t w)
{
	STATE(self, struct renc)->front_cw = w == 0;
}

static void
rencIgnore(__unused id self, __unused SEL _cmd, __unused uint64_t a)
{
}

static void
rencIgnoreObj(__unused id self, __unused SEL _cmd, __unused id a)
{
	static int once;

	if (!once++) {
		lavlog("a render state object (depth/stencil, ...) is not served yet; ignored");
	}
}

static void
rencAddDraw(id self, uint64_t type, uint64_t first, uint64_t count, uint64_t instances, uint64_t base, id ibuf, uint64_t ioff,
    uint32_t isize)
{
	struct renc * re = STATE(self, struct renc);
	struct op * op;
	struct draw * d;
	static const uint32_t prims[5] = { 9, 2, 3, 4, 6 };     /* point, line, line strip, triangle, triangle strip */

	if (re->pipeline == nil || type >= 5) {
		lavlog("a draw with no pipeline state, or of primitive type %llu", (unsigned long long)type);
		return;
	}
	op = cbAddOp(STATE(re->commandBuffer, struct cbstate));
	d = calloc(1, sizeof(*d));
	op->kind = OP_DRAW;
	op->draw = d;
	d->pipeline = [re->pipeline retain];
	d->target = [re->target retain];
	for (int i = 1; i < 8; i++) {
		d->xtarget[i] = [re->xtarget[i] retain];
	}
	for (int i = 0; i < LAV5_MAX_BUFFERS; i++) {
		d->vbuf[i] = [re->vbuf[i] retain];
		d->voff[i] = re->voff[i];
		d->fbuf[i] = [re->fbuf[i] retain];
		d->foff[i] = re->foff[i];
		if (re->vbytes[i] != NULL) {
			d->vbytes[i] = malloc(re->vblen[i]);
			memcpy(d->vbytes[i], re->vbytes[i], re->vblen[i]);
			d->vblen[i] = re->vblen[i];
		}
		if (re->fbytes[i] != NULL) {
			d->fbytes[i] = malloc(re->fblen[i]);
			memcpy(d->fbytes[i], re->fbytes[i], re->fblen[i]);
			d->fblen[i] = re->fblen[i];
		}
	}
	d->prim = prims[type];
	d->ibuf = [ibuf retain];
	d->ioff = ioff;
	d->isize = isize;
	d->first = (uint32_t)first;
	d->count = (uint32_t)count;
	d->instances = (uint32_t)instances;
	d->base_instance = (uint32_t)base;
	memcpy(d->viewport, re->viewport, sizeof(d->viewport));
	memcpy(d->scissor, re->scissor, sizeof(d->scissor));
	d->cull = re->cull;
	d->front_cw = re->front_cw;
	for (int i = 0; i < 16; i++) {
		d->ftex[i] = [re->ftex[i] retain];
		d->fsamp[i] = [re->fsamp[i] retain];
	}
	memcpy(d->blend_color, re->blend_color, sizeof(d->blend_color));
}

static void
rencDraw(id self, __unused SEL _cmd, uint64_t type, uint64_t first, uint64_t count)
{
	rencAddDraw(self, type, first, count, 1, 0, nil, 0, 0);
}

static void
rencDrawInstanced(id self, __unused SEL _cmd, uint64_t type, uint64_t first, uint64_t count, uint64_t instances)
{
	rencAddDraw(self, type, first, count, instances, 0, nil, 0, 0);
}

static void
rencDrawBase(id self, __unused SEL _cmd, uint64_t type, uint64_t first, uint64_t count, uint64_t instances, uint64_t base)
{
	rencAddDraw(self, type, first, count, instances, base, nil, 0, 0);
}

static void
rencDrawIndexedBase(id self, __unused SEL _cmd, uint64_t type, uint64_t count, uint64_t itype, id ibuf, uint64_t ioff,
    uint64_t instances, int64_t basevertex, uint64_t base)
{
	/* the vertex offset (VFD_INDEX_OFFSET) is the base vertex */
	rencAddDraw(self, type, (uint32_t)basevertex, count, instances, base, ibuf, ioff, itype == 0 ? 1 : 2);
}

static void
rencDrawIndexed(id self, SEL cmd, uint64_t type, uint64_t count, uint64_t itype, id ibuf, uint64_t ioff)
{
	rencDrawIndexedBase(self, cmd, type, count, itype, ibuf, ioff, 1, 0, 0);
}

static void
rencDrawIndexedInst(id self, SEL cmd, uint64_t type, uint64_t count, uint64_t itype, id ibuf, uint64_t ioff, uint64_t instances)
{
	rencDrawIndexedBase(self, cmd, type, count, itype, ibuf, ioff, instances, 0, 0);
}

static void
rencEnd(id self, __unused SEL _cmd)
{
	STATE(self, struct renc)->ended = YES;
}

static id rencLabel(id self, __unused SEL _cmd) { return STATE(self, struct renc)->label; }

static void
rencSetLabel(id self, __unused SEL _cmd, id label)
{
	struct renc * re = STATE(self, struct renc);

	[re->label release];
	re->label = [label copy];
}

static uint8_t
clamp8(double v)
{
	return v <= 0 ? 0 : v >= 1 ? 255 : (uint8_t)(v * 255.0 + 0.5);
}

static BOOL
runClear(struct op * op)
{
	struct texstate * t = STATE(op->dst, struct texstate);
	uint32_t r = clamp8(op->clear[0]), g = clamp8(op->clear[1]), b = clamp8(op->clear[2]), a = clamp8(op->clear[3]);
	uint32_t px = t->format == PF_BGRA8 ? (b | g << 8 | r << 16 | a << 24) : (r | g << 8 | b << 16 | a << 24);

	if (!pfIsDraw(t->format)) {
		memset(t->b.cpu, 0, (size_t)t->pitch * t->height);     /* a clear of a format not drawn into: zero */
		texCpuDone(t, 0, (size_t)t->pitch * t->height);
		return YES;
	}

	for (uint32_t y = 0; y < t->height; y++) {
		uint32_t * row = (uint32_t *)((uint8_t *)t->b.cpu + (size_t)y * t->pitch);

		for (uint32_t x = 0; x < t->width; x++) {
			row[x] = px;
		}
	}
	texCpuDone(t, 0, (size_t)t->pitch * t->height);
	return YES;
}

static BOOL
bindStage(struct lav5_binding * out, struct lav5_vbuf * vb, id * bufs, uint64_t * offs, uint8_t ** bytes, uint32_t * blen,
    const uint32_t * stride, size_t * boff, const char * name)
{
	for (int i = 0; i < LAV5_MAX_BUFFERS; i++) {
		if (bufs[i] != nil) {
			struct bufstate * b = STATE(bufs[i], struct bufstate);

			if (offs[i] > b->b.len) {
				lavlog("draw of %s: the offset of buffer %d is past its end", name, i);
				return NO;
			}
			out[i].gpu = b->b.gpu + offs[i];
			out[i].cpu = (uint8_t *)b->b.cpu + offs[i];
			out[i].bytes = b->b.len - offs[i];
		} else if (bytes[i] != NULL) {
			uint32_t len = (blen[i] + 15) & ~15u;

			if (*boff + len > gBytes.len) {
				lavlog("draw of %s: too many setBytes", name);
				return NO;
			}
			memset((uint8_t *)gBytes.cpu + *boff, 0, len);
			memcpy((uint8_t *)gBytes.cpu + *boff, bytes[i], blen[i]);
			out[i].gpu = gBytes.gpu + *boff;
			out[i].cpu = (uint8_t *)gBytes.cpu + *boff;
			out[i].bytes = len;
			*boff += len;
		}
		if (vb != NULL) {
			vb[i].gpu = out[i].gpu;
			vb[i].bytes = out[i].bytes;
			vb[i].stride = stride[i];
		}
	}
	return YES;
}

static int gDrawTrace = 0;      /* log every draw's target, textures and buffers: on while /var/tmp/lavtrace exists (checked per draw) */
static struct lbuf gPvt[2];     /* private memory of the vertex and the fragment stage, grown on demand */


static BOOL
runDraw(struct op * op)
{
	struct draw * d = op->draw;
	struct rpipe * rp = STATE(d->pipeline, struct rpipe);
	struct texstate * t = STATE(d->target, struct texstate);
	struct lav5_render * r = calloc(1, sizeof(*r));
	uint32_t * w;
	const char * why = NULL;
	size_t n, boff = 0;
	BOOL ok;
	const char * name = rp->vs.k.name[0] ? rp->vs.k.name : "(render)";

	gDrawTrace = access("/var/tmp/lavtrace", F_OK) == 0;
	if (!pfIsDraw(t->format)) {
		lavlog("draw of %s skipped: the colour target has pixel format %u, only RGBA8/BGRA8 are drawn into", name, t->format);
		free(r);
		return YES;
	}
	/* a draw that writes no colour in any target does nothing here (there is no depth or stencil to write); the RB hangs on it */
	{
		BOOL any = rp->blend[0].writemask != 0;

		for (int i = 1; i < 8; i++) {
			if (d->xtarget[i] != nil && rp->blend[i].writemask != 0) {
				any = YES;
			}
		}
		if (!any) {
			lavlog("draw of %s skipped: the colour write mask is 0", name);
			free(r);
			return YES;
		}
	}
	if (gBytes.handle == 0 && !lavBufNew(65536, &gBytes)) {
		free(r);
		return NO;
	}
	uint64_t tp0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
	const struct rspec * sp = rpipeSpecialise(rp, d);
	const struct lav5_stage * vsx = sp != NULL ? &sp->vs : &rp->vs, * fsx = sp != NULL ? &sp->fs : &rp->fs;

	gTSpec += clock_gettime_nsec_np(CLOCK_MONOTONIC) - tp0;
	r->vs = vsx;
	r->fs = fsx;
	r->link = sp != NULL ? &sp->link : &rp->link;
	r->vs_code = sp != NULL ? sp->vcode.gpu : rp->vcode.gpu;
	for (int i = 0; i < 2; i++) {
		const struct lav5_stage * st = i == 0 ? vsx : fsx;
		uint32_t bytes = lav5_pvtmem_bytes(st);

		if (bytes == 0) {
			continue;
		}
		if (bytes > gPvt[i].len) {
			struct lbuf nb;

			if (!lavBufNew(bytes, &nb)) {
				lavlog("draw of %s: no memory for %u bytes of private memory", name, bytes);
				free(r);
				return NO;
			}
			gPvt[i] = nb;
		}
		if (i == 0) {
			r->vs_pvt_gpu = gPvt[i].gpu;
		} else {
			r->fs_pvt_gpu = gPvt[i].gpu;
		}
	}
	r->fs_code = sp != NULL ? sp->fcode.gpu : rp->fcode.gpu;
	r->rt_gpu = t->b.gpu;
	r->rt_width = t->width;
	r->rt_height = t->height;
	r->rt_pitch = t->pitch;
	r->rt_format = t->format == PF_BGRA8 ? 0 : 1;
	r->nattr = rp->nattr;
	memcpy(r->attr, rp->attr, sizeof(r->attr));
	if (!bindStage(r->vs_buf, r->vbuf, d->vbuf, d->voff, d->vbytes, d->vblen, rp->stride, &boff, name) ||
	    !bindStage(r->fs_buf, NULL, d->fbuf, d->foff, d->fbytes, d->fblen, NULL, &boff, name)) {
		free(r);
		return NO;
	}
	r->prim = d->prim;
	r->first = d->first;
	r->count = d->count;
	r->instances = d->instances;
	r->base_instance = d->base_instance;
	if (d->isize != 0 && d->ibuf != nil) {
		struct bufstate * ib = STATE(d->ibuf, struct bufstate);

		if (d->ioff >= ib->b.len) {
			lavlog("indexed draw: the index buffer offset is past its end");
			free(r);
			return NO;
		}
		r->index_gpu = ib->b.gpu + d->ioff;
		r->index_size = d->isize;
		r->index_max = (uint32_t)((ib->b.len - d->ioff) / (d->isize == 1 ? 2 : 4));
	}
	if (r->prim == 6 && r->index_gpu == 0 && r->first == 0 && r->count >= 3 && r->count <= 4096) {
		/* a triangle strip is drawn as a triangle list from a small index table: the first strip draw hung the CP */
		static struct lbuf stripIdx;
		uint32_t * ix;
		uint32_t n = 0;

		if (stripIdx.len == 0 && !lavBufNew(4096 * 3 * 4, &stripIdx)) {
			lavlog("draw of %s: no memory for the strip indices", name);
			free(r);
			return NO;
		}
		ix = (uint32_t *)stripIdx.cpu;
		for (uint32_t k = 0; k + 2 < r->count; k++) {
			ix[n++] = (k & 1) ? k + 1 : k;
			ix[n++] = (k & 1) ? k : k + 1;
			ix[n++] = k + 2;
		}
		r->prim = 4;
		r->index_gpu = stripIdx.gpu;
		r->index_size = 2;
		r->index_max = n;
		r->count = n;
	}
	memcpy(r->viewport, d->viewport, sizeof(r->viewport));
	memcpy(r->scissor, d->scissor, sizeof(r->scissor));
	r->cull = d->cull;
	r->front_cw = d->front_cw;
	r->blend = rp->blend[0];
	r->nrt = 1;
	for (int i = 1; i < 8; i++) {
		r->blend_x[i] = rp->blend[i];
		if (d->xtarget[i] != nil) {
			struct texstate * xt = STATE(d->xtarget[i], struct texstate);

			r->rt[i] = (struct lav5_rt){ xt->b.gpu, xt->pitch, xt->format == PF_BGRA8 ? 0u : 1u, 1 };
			r->nrt = i + 1;
		}
	}
	r->rt[0] = (struct lav5_rt){ r->rt_gpu, r->rt_pitch, r->rt_format, 1 };
	memcpy(r->blend_color, d->blend_color, sizeof(r->blend_color));
	for (int i = 0; i < 16; i++) {
		if (d->ftex[i] != nil) {
			struct texstate * ts = STATE(d->ftex[i], struct texstate);

			const struct pfmt * pf = pfmtOf(ts->format);

			if (ts->surface != NULL && ts != t && (size_t)ts->height * ts->pitch <= (1u << 20)) {
				/* a texture over an IOSurface that is only sampled (a glyph atlas): its pixels live in the surface, bring them in */
				uint64_t tc0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);

				surfaceCopy(ts, NO);
				gTCopy += clock_gettime_nsec_np(CLOCK_MONOTONIC) - tc0;
			}
			r->tex[i] = (struct lav5_tex){ ts->b.gpu, ts->width, ts->height, ts->pitch, pf != NULL ? pf->bgra : 0, 1,
			    pf != NULL ? pf->tfmt : 0, pf != NULL ? pf->nch : 4, pf != NULL ? pf->alpha_only : 0, ts->type == 0,
			    pf != NULL ? pf->srgb : 0 };
		}
		if (d->fsamp[i] != nil) {
			r->samp[i] = *STATE(d->fsamp[i], struct lav5_samp);
		}
	}
	r->flush_gpu = gStream.gpu + SCRATCH_OFFSET + 64;
	r->hole_gpu = gStream.gpu + SCRATCH_OFFSET + 128;
	/* CoreAnimation clears an attachment with a draw of attachment_clear_frag (it returns zero for every target). On the 1080x2340
	 * targets that draw hung the SoC, so it is done here as a fill with zero of the scissor rectangle of each target written */
	if (strncmp([STATE(rp->ffn, struct fnstate)->name UTF8String], "attachment_clear_frag", 21) == 0) {
		uint32_t sx = d->scissor[0], sy = d->scissor[1], sw = d->scissor[2], sh = d->scissor[3];

		for (int i = 0; i < 8; i++) {
			struct texstate * tt = i == 0 ? t : (d->xtarget[i] != nil ? STATE(d->xtarget[i], struct texstate) : NULL);

			if (tt == NULL || rp->blend[i].writemask == 0 || !pfIsDraw(tt->format)) {
				continue;
			}
			for (uint32_t y = sy; y < sy + sh && y < tt->height; y++) {
				uint32_t x1 = sx + sw < tt->width ? sx + sw : tt->width;

				if (sx < x1) {
					memset((uint8_t *)tt->b.cpu + (size_t)y * tt->pitch + (size_t)sx * 4, 0, (size_t)(x1 - sx) * 4);
				}
			}
			if (sy < tt->height) {
				texCpuDoneRows(tt, sy, sy + sh < tt->height ? sy + sh : tt->height);
			}
		}
		lavlog("draw: attachment clear done on the CPU (%u x %u at %u,%u)", sw, sh, sx, sy);
		free(r);
		return YES;
	}
	/* a draw over many rows goes in bands of BAND rows, one submit each: a full-screen draw on the 1080x2340 target hung the whole SoC
	 * (the GPU stalls the bus), while the same draw on a smaller scissor completes */
	uint32_t bandY = r->scissor[1], bandH = r->scissor[3], bandStep = bandH;
	if (bandH > LAV_BAND_ROWS) {
		bandStep = LAV_BAND_ROWS;
		r->scissor[3] = bandStep;
		if (gDrawTrace) {
			lavlog("draw: %u rows submitted in bands of %u", bandH, bandStep);
		}
	}
	tp0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
	w = malloc(STREAM_WORDS * 4);
	n = lav5_build_render(w, STREAM_WORDS, r, &why);
	gTBuild += clock_gettime_nsec_np(CLOCK_MONOTONIC) - tp0;
	if (n == 0) {
		lavlog("draw refused: %s", why != NULL ? why : "?");
		free(w);
		free(r);
		return NO;
	}
	if (gDrawTrace) {
		char tb[200];
		size_t o = 0;

		for (int i = 0; i < 16 && o < sizeof(tb) - 24; i++) {
			if (r->tex[i].valid) {
				o += (size_t)snprintf(tb + o, sizeof(tb) - o, " t%d=%ux%u/f%u", i, r->tex[i].width, r->tex[i].height, r->tex[i].tfmt);
			}
		}
		tb[o] = 0;
		for (int i = 1; i < 8; i++) {
			if (d->xtarget[i] != nil) {
				struct texstate * xt = STATE(d->xtarget[i], struct texstate);

				lavlog("draw detail: colour target %d is %ux%u fmt %u pitch %u, write mask %x", i, xt->width, xt->height, xt->format, xt->pitch,
				    rp->blend[i].writemask);
			}
		}
		lavlog("draw detail: target %ux%u fmt %u pitch %u, blend %u mask %x, viewport %.0f,%.0f %.0fx%.0f, vs %u fs %u words, textures:%s",
		    r->rt_width, r->rt_height, r->rt_format, r->rt_pitch, r->blend.enable, r->blend.writemask, r->viewport[0], r->viewport[1],
		    r->viewport[2], r->viewport[3], 0u, 0u, tb);
	}
	if (gDrawTrace) {
		lavlog("draw detail: fs ncolor %u regid %x %x half %u %u nrt %u rt1 gpu %llx", (unsigned)r->fs->ncolor, (unsigned)r->fs->color_regid[0],
		    (unsigned)r->fs->color_regid[1], (unsigned)r->fs->color_half[0], (unsigned)r->fs->color_half[1], r->nrt,
		    (unsigned long long)r->rt[1].gpu);
		lavlog("draw detail: fs instrlen %u constlen %u maxreg %d maxhalf %d dbl %u ssbo %u ubo %u pvt %u kill %u pixlod %u in %u totin %u ij %x %x frag %x face %x mask %x",
		    r->fs->k.instrlen, r->fs->k.constlen, r->fs->k.max_reg, r->fs->k.max_half_reg, r->fs->k.double_threadsize, r->fs->k.num_ssbos,
		    r->fs->k.num_ubos, r->fs->k.pvtmem_size, r->fs->has_kill, r->fs->need_pixlod, r->fs->nin, r->fs->total_in, r->fs->ij_regid[0],
		    r->fs->ij_regid[1], r->fs->frag_coord_regid, r->fs->face_regid, r->fs->frag_coord_compmask);
		lavlog("draw detail: vs instrlen %u constlen %u nout %u nattr %u pos %x cnt %u", r->vs->k.instrlen, r->vs->k.constlen, r->vs->nout,
		    r->vs->nattr, r->vs->position_regid, r->link->cnt);
	}
	if (gDrawTrace) {
		/* the fragment constant buffers bound: their first 16 words (the stage reads uniform ranges out of them) */
		for (int i = 0; i < LAV5_MAX_BUFFERS; i++) {
			if (r->fs_buf[i].cpu != NULL && r->fs_buf[i].bytes != 0) {
				const uint32_t * q = r->fs_buf[i].cpu;
				unsigned nw = r->fs_buf[i].bytes < 64 ? (unsigned)r->fs_buf[i].bytes / 4 : 16;
				char tb[200];
				size_t o = 0;

				for (unsigned k = 0; k < nw && o < sizeof(tb) - 12; k++) {
					o += (size_t)snprintf(tb + o, sizeof(tb) - o, " %x", q[k]);
				}
				lavlog("draw buffer fs[%d] %llu bytes:%s", i, (unsigned long long)r->fs_buf[i].bytes, tb);
				if (r->fs_buf[i].bytes >= 192) {
					o = 0;
					for (unsigned k = 32; k < 48 && o < sizeof(tb) - 12; k++) {
						o += (size_t)snprintf(tb + o, sizeof(tb) - o, " %x", q[k]);
					}
					lavlog("draw buffer fs[%d] words 32..47:%s", i, tb);
				}
			}
		}
		/* what each texture the draw samples holds right now (CPU view): a 4x4 grid, how many 32-bit words are not zero, two of them */
		for (int i = 0; i < 16; i++) {
			if (d->ftex[i] != nil) {
				struct texstate * ts = STATE(d->ftex[i], struct texstate);
				unsigned nz = 0;
				uint32_t first = 0, mid = 0;

				if (ts->width < 4 || ts->b.cpu == NULL) {
					continue;
				}
				for (unsigned gy = 0; gy < 4; gy++) {
					for (unsigned gx = 0; gx < 4; gx++) {
						uint32_t px = *(volatile uint32_t *)((uint8_t *)ts->b.cpu + (size_t)(ts->height < 4 ? 0 : (ts->height * (2 * gy + 1)) / 8) * ts->pitch +
						    (size_t)((ts->width * (2 * gx + 1)) / 8) * 4);

						nz += px != 0;
						if (gx == 0 && gy == 0) {
							first = px;
						}
						if (gx == 2 && gy == 2) {
							mid = px;
						}
					}
				}
				lavlog("draw input: texture %d %ux%u fmt %u pitch %u: %u of 16 words non-zero, first %08x, mid %08x", i, ts->width, ts->height,
				    ts->format, ts->pitch, nz, first, mid);
			}
		}
	}
	if (gDrawTrace && t->width == 1080 && t->height == 2340 && rp->nattr > 0) {
		/* the first attribute of the first vertices, as floats: where the draw puts its geometry */
		uint32_t bi = rp->attr[0].buffer;
		const uint8_t * vb = NULL;

		if (bi < LAV5_MAX_BUFFERS) {
			if (d->vbuf[bi] != nil) {
				vb = (const uint8_t *)STATE(d->vbuf[bi], struct bufstate)->b.cpu + d->voff[bi];
			} else if (d->vbytes[bi] != NULL) {
				vb = d->vbytes[bi];
			}
		}
		if (vb != NULL && rp->stride[bi] >= 8 && rp->attr[0].is_float) {
			char tb[240];
			size_t o = 0;

			for (uint32_t k = 0; k < d->count && k < 8 && o < sizeof(tb) - 40; k++) {
				const float * f = (const float *)(vb + (size_t)k * rp->stride[bi] + rp->attr[0].offset);

				o += (size_t)snprintf(tb + o, sizeof(tb) - o, " (%.1f,%.1f)", f[0], f[1]);
			}
			lavlog("draw geometry: %s %u vertices, attr0 buffer %u stride %u:%s", [STATE(rp->ffn, struct fnstate)->name UTF8String], d->count, bi,
			    rp->stride[bi], tb);
		}
	}
	uint32_t camBefore = 0;

	if (gDrawTrace && t->width == 1080 && t->height == 2340) {
		camBefore = *(volatile uint32_t *)((uint8_t *)t->b.cpu + (size_t)2115 * t->pitch + 867 * 4);
	}
	tp0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
	ok = lavSubmit(w, n);
	gNSub++;
	for (uint32_t off = bandStep; ok && off < bandH; off += bandStep) {
		r->scissor[1] = bandY + off;
		r->scissor[3] = bandH - off < bandStep ? bandH - off : bandStep;
		n = lav5_build_render(w, STREAM_WORDS, r, &why);
		ok = n != 0 && lavSubmit(w, n);
		gNSub++;
	}
	gTSubmit += clock_gettime_nsec_np(CLOCK_MONOTONIC) - tp0;
	gPixels += (uint64_t)bandH * d->scissor[2];
	if (access("/var/tmp/lavprof", F_OK) == 0) {
		uint64_t ns = clock_gettime_nsec_np(CLOCK_MONOTONIC) - tp0, px = (uint64_t)bandH * d->scissor[2];

		lavlog("prof: %s fs %u words (base %u) %llu kpix %llu ms = %llu ns/kpix; nrt %u blend %u masks %x/%x textures %d",
		    [STATE(rp->ffn, struct fnstate)->name UTF8String], r->fs->k.nwords, rp->fs.k.nwords, (unsigned long long)(px / 1000u),
		    (unsigned long long)(ns / 1000000u), px >= 1000 ? (unsigned long long)(ns / (px / 1000u)) : 0ull, r->nrt, r->blend.enable, r->blend.writemask, r->blend_x[1].writemask,
		    (int)__builtin_popcount(0) + (r->tex[0].valid + r->tex[1].valid + r->tex[2].valid + r->tex[3].valid + r->tex[4].valid));
	}
	if (gDrawTrace && ok) {
		/* what the draw left in each target it writes: a 4x4 grid of the scissor rectangle, how many samples are not zero, and two */
		for (int i = 0; i < 8; i++) {
			struct texstate * tt = i == 0 ? t : (d->xtarget[i] != nil ? STATE(d->xtarget[i], struct texstate) : NULL);
			unsigned nz = 0;
			uint32_t first = 0, mid = 0;

			if (tt == NULL || (i == 0 ? rp->blend[0].writemask : rp->blend[i].writemask) == 0 || tt->width == 0 || tt->height == 0) {
				continue;
			}
			for (unsigned gy = 0; gy < 4; gy++) {
				for (unsigned gx = 0; gx < 4; gx++) {
					uint32_t x = d->scissor[0] + (d->scissor[2] * (2 * gx + 1)) / 8, y = d->scissor[1] + (d->scissor[3] * (2 * gy + 1)) / 8;
					uint32_t px;

					if (x >= tt->width || y >= tt->height) {
						continue;
					}
					px = *(volatile uint32_t *)((uint8_t *)tt->b.cpu + (size_t)y * tt->pitch + (size_t)x * 4);
					nz += px != 0;
					if (gx == 0 && gy == 0) {
						first = px;
					}
					if (gx == 2 && gy == 2) {
						mid = px;
					}
				}
			}
			lavlog("draw result: target %d %ux%u scissor %u,%u %ux%u verts %u: %u of 16 samples non-zero, first %08x, mid %08x", i, tt->width, tt->height,
			    d->scissor[0], d->scissor[1], d->scissor[2], d->scissor[3], d->count, nz, first, mid);
		}
	}
	if (gDrawTrace && ok && t->width == 1080 && t->height == 2340) {
		uint32_t camAfter = *(volatile uint32_t *)((uint8_t *)t->b.cpu + (size_t)2115 * t->pitch + 867 * 4);

		if (camAfter != camBefore) {
			lavlog("draw: camera button pixel %08x -> %08x (%s, %u vertices, textures %s)", camBefore, camAfter,
			    [STATE(rp->ffn, struct fnstate)->name UTF8String], d->count, d->ftex[4] != nil ? "t4" : "no t4");
		}
	}
	if (gDrawTrace || !ok) {
		lavlog("draw: %u vertices x %u instances, prim %u, %zu words, %s", d->count, d->instances, d->prim, n, ok ? "done" : "FAILED");
	}
	free(w);
	free(r);
	return ok;
}
