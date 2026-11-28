/*
 * LavenderMetal: the plug-in Metal loads when the kernel service names it (M5.1, M5.2).
 *
 * How Metal finds a device (the system's own code, MTLRegisterDevices): it matches the IOKit class IOAcceleratorES, reads the
 * service's MetalPluginName ("LavenderMetal") and MetalPluginClassName ("LavenderMetalDevice"), opens
 * /System/Library/Extensions/LavenderMetal.bundle, takes that class, insists that it is a subclass of Metal's private
 * _MTLDevice that conforms to MTLDevice and MTLDeviceSPI, and sends it -initWithAcceleratorPort:.
 *
 * M5.2 adds the objects a program uses to move bytes with the GPU:
 *   buffers          memory the kernel service hands out (physically contiguous, below 4 GB, mapped into the process), so the
 *                    CPU sees it through -contents and the GPU addresses it through -gpuAddress
 *   command queue    -newCommandQueue, -commandBuffer
 *   command buffer   -blitCommandEncoder, -commit, -waitUntilCompleted, completed handlers, -status
 *   blit encoder     -fillBuffer:range:value: (the GPU's command processor writes the words: CP_MEM_WRITE) and
 *                    -copyFromBuffer:... (the 2D core blits the 64-byte aligned middle, the same CP_BLIT sequence as the kernel's
 *                    step 18; an unaligned head and tail are copied by the CPU and counted in the log)
 * A command buffer runs when it is committed, op by op, each op one synchronous submission to the service.
 *
 * Everything Metal sends that nothing implements is answered with zero and written once to /var/root/lavmetal.log
 * ("asked and not implemented"); the return type is read from the protocols the class declares, so structs come back sized.
 *
 * _MTLDevice and the other bases are not in the SDK (they live in the system's Metal.framework), so the classes are made at run
 * time, in the constructor that runs when the bundle is loaded. Built without ARC.
 */
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <IOKit/IOKitLib.h>
#include "lava5.h"
#include "lavcc.h"
#include <mach/mach.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <execinfo.h>
#include <signal.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <sys/stat.h>

#define LAV_LOG "/var/root/lavmetal.log"

static void
lavlog(const char * fmt, ...)
{
	char line[512];
	va_list ap;
	FILE * f;

	va_start(ap, fmt);
	vsnprintf(line, sizeof(line), fmt, ap);
	va_end(ap);
	f = fopen(LAV_LOG, "a");
	if (f != NULL) {
		fprintf(f, "%ld LavenderMetal[%d]: %s\n", (long)time(NULL), (int)getpid(), line);
		fclose(f);
	}
	NSLog(@"LavenderMetal: %s", line);
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* The service: LavenderGPU (kernel/LavenderGPU.cpp). Buffers, and streams of PM4 words submitted by reference.         */
/* ------------------------------------------------------------------------------------------------------------------ */

enum { kLavInfo = 0, kLavBufNew = 1, kLavBufFree = 2, kLavSubmit = 3, kLavSubmitBuf = 4, kLavMetal = 5, kLavSegs = 6 };

struct lav_buf_new { uint64_t size; uint32_t handle, pad; uint64_t address; };
struct lav_buf_free { uint32_t handle, pad; };
struct lav_submit_out { uint32_t fence, status; };
struct lav_submit_buf { uint32_t handle, offset, dwords, pad; };

struct lbuf {
	uint32_t handle;
	int32_t slab;                   /* -1: a buffer of the service of its own; else the slab it is carved from */
	uint32_t page, npages;          /* within the slab */
	uint64_t gpu;                   /* the GPU's address (= physical, below 4 GB) */
	void * cpu;                     /* the process's mapping */
	size_t len;
	mach_vm_address_t map;
	int cached;                     /* write-back memory of its own (lav_buf_new.pad bit 0): the CPU's accesses are followed by a flush of what they touched */
};

static io_connect_t gConn;
static struct lbuf gStream;         /* [0, 16 KB): the PM4 words of a submission; [32 KB, 33 KB): scratch the GPU writes events to */
static pthread_mutex_t gGPULock = PTHREAD_MUTEX_INITIALIZER;

#define STREAM_WORDS    4096u
static void lavProbeBatch(void);
#define BATCH_WORDS     8192u       /* the words of the queued draws: they go out as one stream, [0, 32 KB) of gStream */
#define LAV_BAND_ROWS   400u        /* rows of a draw per submit */
#define SCRATCH_OFFSET  0x8000u

static BOOL
lavBufNewDirectF(size_t len, struct lbuf * b, uint32_t flags)
{
	struct lav_buf_new bn;
	size_t out = sizeof(bn);
	mach_vm_size_t msize = 0;
	kern_return_t kr;

	memset(&bn, 0, sizeof(bn));
	bn.size = len;
	bn.pad = flags;
	kr = IOConnectCallStructMethod(gConn, kLavBufNew, &bn, sizeof(bn), &bn, &out);
	if (kr != KERN_SUCCESS) {
		lavlog("buffer of %zu bytes: kr %#x", len, kr);
		return NO;
	}
	b->handle = bn.handle;
	b->slab = -1;
	b->gpu = bn.address;
	b->len = (size_t)bn.size;
	b->map = 0;
	kr = IOConnectMapMemory64(gConn, bn.handle, mach_task_self(), &b->map, &msize, kIOMapAnywhere);
	if (kr != KERN_SUCCESS) {
		struct lav_buf_free bf = { bn.handle, 0 };

		lavlog("map of buffer %u: kr %#x", bn.handle, kr);
		IOConnectCallStructMethod(gConn, kLavBufFree, &bf, sizeof(bf), NULL, NULL);
		return NO;
	}
	b->cpu = (void *)(uintptr_t)b->map;
	b->cached = (flags & 1) != 0;
	return YES;
}

static BOOL
lavBufNewDirect(size_t len, struct lbuf * b)
{
	return lavBufNewDirectF(len, b, 0);
}

static void
lavBufFreeDirect(struct lbuf * b)
{
	struct lav_buf_free bf = { b->handle, 0 };

	if (b->handle == 0) {
		return;
	}
	IOConnectUnmapMemory64(gConn, b->handle, mach_task_self(), b->map);
	IOConnectCallStructMethod(gConn, kLavBufFree, &bf, sizeof(bf), NULL, NULL);
	memset(b, 0, sizeof(*b));
}

/*
 * The service holds 64 buffers for every process, and CoreAnimation wants hundreds (each texture, buffer and pipeline takes one), so
 * the plug-in asks the service for a few big slabs and carves its buffers out of them, a page (4 KB) at a time. A request of half a
 * slab or more still gets a buffer of its own. Everything a slab hands out is zeroed, as a service buffer is.
 */
#define POOL_PAGE       4096u
#define POOL_MAX_SLABS  48
#define POOL_SLAB_BYTES (8u << 20)

struct slab {
	struct lbuf big;
	uint8_t * used;                 /* one byte per page */
	uint32_t npages;
};

static struct slab gSlabs[POOL_MAX_SLABS];
static pthread_mutex_t gPoolLock = PTHREAD_MUTEX_INITIALIZER;

/* the first run of NPAGES free pages in the slab, or -1 */
static int64_t
slabFind(const struct slab * sl, uint32_t npages)
{
	uint32_t run = 0;

	for (uint32_t i = 0; i < sl->npages; i++) {
		run = sl->used[i] ? 0 : run + 1;
		if (run == npages) {
			return (int64_t)i + 1 - npages;
		}
	}
	return -1;
}

static BOOL
slabAdd(void)
{
	/* physically contiguous memory below 4 GB is scarce on this port (the service answers kIOReturnNoMemory for 8 MB, 2 MB, soon
	 * for 512 KB): start from the biggest slab that was last possible and only go down */
	static const size_t sizes[5] = { POOL_SLAB_BYTES, 2u << 20, 512u << 10, 128u << 10, 32u << 10 };
	static int first;

	for (int i = 0; i < POOL_MAX_SLABS; i++) {
		if (gSlabs[i].big.handle != 0) {
			continue;
		}
		for (int k = first; k < 5; k++) {
			if (lavBufNewDirect(sizes[k], &gSlabs[i].big)) {
				first = k;
				gSlabs[i].npages = (uint32_t)(gSlabs[i].big.len / POOL_PAGE);
				gSlabs[i].used = calloc(gSlabs[i].npages, 1);
				lavlog("slab %d: %zu bytes at GPU %#llx", i, gSlabs[i].big.len, (unsigned long long)gSlabs[i].big.gpu);
				return gSlabs[i].used != NULL;
			}
		}
		return NO;
	}
	return NO;
}

static BOOL
lavBufNew(size_t len, struct lbuf * b)
{
	uint32_t npages = (uint32_t)((len + POOL_PAGE - 1) / POOL_PAGE);

	memset(b, 0, sizeof(*b));
	if (npages == 0) {
		npages = 1;
	}
	if ((size_t)npages * POOL_PAGE >= POOL_SLAB_BYTES / 2 || gConn == 0 || gStream.handle == 0) {
		/* big (or the stream buffer, made before the pool is usable): its own */
		return lavBufNewDirect((size_t)npages * POOL_PAGE, b);
	}
	pthread_mutex_lock(&gPoolLock);
	for (int attempt = 0; attempt < 2; attempt++) {
		for (int i = 0; i < POOL_MAX_SLABS; i++) {
			struct slab * sl = &gSlabs[i];
			int64_t at;

			if (sl->big.handle == 0 || sl->npages < npages || (at = slabFind(sl, npages)) < 0) {
				continue;
			}
			memset(sl->used + at, 1, npages);
			b->handle = sl->big.handle;
			b->slab = i;
			b->page = (uint32_t)at;
			b->npages = npages;
			b->gpu = sl->big.gpu + (uint64_t)at * POOL_PAGE;
			b->cpu = (uint8_t *)sl->big.cpu + (size_t)at * POOL_PAGE;
			b->len = (size_t)npages * POOL_PAGE;
			b->map = 0;
			pthread_mutex_unlock(&gPoolLock);
			memset(b->cpu, 0, b->len);
			return YES;
		}
		if (attempt == 0 && !slabAdd()) {
			break;
		}
	}
	pthread_mutex_unlock(&gPoolLock);
	/* the slabs could not grow (or the request is bigger than a slab of the size that was left): try a buffer of its own */
	return lavBufNewDirect((size_t)npages * POOL_PAGE, b);
}

static void
lavBufFree(struct lbuf * b)
{
	if (b->handle == 0) {
		return;
	}
	if (b->slab >= 0) {
		pthread_mutex_lock(&gPoolLock);
		memset(gSlabs[b->slab].used + b->page, 0, b->npages);
		pthread_mutex_unlock(&gPoolLock);
		memset(b, 0, sizeof(*b));
		return;
	}
	lavBufFreeDirect(b);
}

static BOOL
lavOpen(void)
{
	static dispatch_once_t once;

	dispatch_once(&once, ^{
		static const char * const names[] = { "LavenderGPU", "IOAcceleratorES" };
		kern_return_t kr = kIOReturnNotFound;

		for (unsigned i = 0; i < 2 && gConn == 0; i++) {
			io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching(names[i]));

			if (service == 0) {
				continue;
			}
			kr = IOServiceOpen(service, mach_task_self(), 0, &gConn);
			IOObjectRelease(service);
			if (kr != KERN_SUCCESS) {
				lavlog("IOServiceOpen(%s): kr %#x", names[i], kr);
				gConn = 0;
			}
		}
		if (gConn == 0) {
			lavlog("no GPU service (last kr %#x)", kr);
			return;
		}
		if (!lavBufNewDirect(65536, &gStream)) {
			gConn = 0;
			return;
		}
		lavlog("the GPU service is open; stream buffer at GPU %#llx", (unsigned long long)gStream.gpu);
		lavProbeBatch();
	});
	return gConn != 0 && gStream.handle != 0;
}

/* One synchronous submission: the words go into the stream buffer and the service reads them from there. */
static BOOL
lavSubmitNow(const uint32_t * words, size_t count)
{
	struct lav_submit_buf sb = { gStream.handle, 0, (uint32_t)count, 0 };
	struct lav_submit_out out = { 0, 0 };
	size_t outsize = sizeof(out);
	kern_return_t kr;

	if (count == 0 || count > BATCH_WORDS) {
		return NO;
	}
	memcpy(gStream.cpu, words, count * 4);
	uint64_t t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
	kr = IOConnectCallStructMethod(gConn, kLavSubmitBuf, &sb, sizeof(sb), &out, &outsize);
	{
		uint64_t ms = (clock_gettime_nsec_np(CLOCK_MONOTONIC) - t0) / 1000000u;

		if (ms >= 100) {
			lavlog("submit of %zu words took %llu ms", count, (unsigned long long)ms);
		}
	}
	if (kr != KERN_SUCCESS || out.fence == 0) {
		lavlog("submit of %zu words: kr %#x fence %u status %u", count, kr, out.fence, out.status);
		return NO;
	}
	return YES;
}

/*
 * Batching: a draw's stream is ~1000 words and every submit costs ~2 ms of fixed work in the service (fence, cache
 * invalidate, wait), so the draws of a command buffer are queued here and go to the GPU as one stream. Whatever needs the
 * GPU's result or touches memory the GPU uses (any op but a draw, the surface sync, a CPU copy into a texture) calls
 * lavFlush() first; lavSubmit() flushes by itself so the order is kept. gBytesUsed is how much of gBytes (setBytes data
 * the queued draws read) is taken: it starts again at 0 after a flush. /var/tmp/lavnobatch turns it off.
 */
static size_t gBatchMax = 4096;              /* words per submit the kernel takes: 4096 before patch 0367, BATCH_WORDS after (probed at open) */
static uint32_t gPend[BATCH_WORDS];
static size_t gPendN;
static size_t gBytesUsed;
static unsigned gCommitGen = 1;
static int gNoBatch;

static BOOL
lavFlush(void)
{
	size_t n = gPendN;

	gPendN = 0;
	gBytesUsed = 0;
	if (n == 0) {
		return YES;
	}
	return lavSubmitNow(gPend, n);
}

static BOOL
lavSubmit(const uint32_t * words, size_t count)
{
	if (!lavFlush()) {
		return NO;
	}
	return lavSubmitNow(words, count);
}

static BOOL
lavQueue(const uint32_t * words, size_t count)
{
	if (gNoBatch || gBatchMax <= 4096 || count > gBatchMax / 2) {
		return lavSubmit(words, count);
	}
	if (gPendN + count > gBatchMax) {
		size_t keep = gBytesUsed;

		if (!lavFlush()) {
			return NO;
		}
		gBytesUsed = keep;                          /* the draw being queued has its setBytes data in place already */
	}
	memcpy(gPend + gPendN, words, count * 4);
	gPendN += count;
	return YES;
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* PM4 streams                                                                                                         */
/* ------------------------------------------------------------------------------------------------------------------ */

static uint32_t
parity(uint32_t v)
{
	v ^= v >> 16;
	v ^= v >> 8;
	v ^= v >> 4;
	return (0x9669u >> (v & 0xf)) & 1u;
}

/* A harmless stream of CP_NOPs longer than the old kernel limit: if the service takes it, draws can be batched. */
static void
lavProbeBatch(void)
{
	static uint32_t nop[4100];
	uint32_t cnt = 4099, op = 0x10;

	nop[0] = 0x70000000u | cnt | (parity(cnt) << 15) | (op << 16) | (parity(op) << 23);
	if (lavSubmitNow(nop, 4100)) {
		gBatchMax = BATCH_WORDS;
	}
	lavlog("batching: the service takes streams of %zu words%s", gBatchMax, gBatchMax > 4096 ? " (draws are queued and sent together)" : " (old kernel: one submit per draw)");
}

struct stream {
	uint32_t w[STREAM_WORDS];
	size_t n;
};

static void
put(struct stream * s, uint32_t v)
{
	if (s->n < STREAM_WORDS) {
		s->w[s->n++] = v;
	}
}

static void
pkt7(struct stream * s, uint32_t opcode, uint32_t cnt)
{
	put(s, 0x70000000u | (cnt & 0x3fff) | (parity(cnt) << 15) | ((opcode & 0x7f) << 16) | (parity(opcode) << 23));
}

static void
pkt4(struct stream * s, uint32_t reg, uint32_t cnt)
{
	put(s, 0x40000000u | cnt | (parity(cnt) << 7) | ((reg & 0x3ffffu) << 8) | (parity(reg & 0x3ffffu) << 27));
}

static void
reg1(struct stream * s, uint32_t reg, uint32_t v)
{
	pkt4(s, reg, 1);
	put(s, v);
}

#define CP_MEM_WRITE            61u
#define CP_SET_RENDER_MODE      0x6cu
#define CP_BLIT                 0x2cu
#define CP_WAIT_FOR_IDLE        38u
#define CP_EVENT_WRITE          0x46u

/* a run of words written by the command processor at `gpu` (a multiple of 4), at most 1000 words a packet */
static void
emitWrite(struct stream * s, uint64_t gpu, const uint32_t * words, size_t count)
{
	while (count > 0) {
		size_t k = count > 1000 ? 1000 : count;

		pkt7(s, CP_MEM_WRITE, (uint32_t)(2 + k));
		put(s, (uint32_t)gpu);
		put(s, (uint32_t)(gpu >> 32));
		for (size_t i = 0; i < k; i++) {
			put(s, words[i]);
		}
		words += k;
		gpu += k * 4;
		count -= k;
	}
}

/* the 2D core's state, as the kernel's step 18 sets it (emit_setup) */
static void
emit2DSetup(struct stream * s)
{
	reg1(s, 0xe141u, 0x8);                  /* RB_RENDER_CNTL */
	reg1(s, 0x2100u, 0x86000000u);          /* RB_2D_BLIT_CNTL */
	reg1(s, 0x2180u, 0x86000000u);          /* GRAS_2D_BLIT_CNTL */
	reg1(s, 0x2184u, 0x9);
	reg1(s, 0xe140u, 1u << 17);             /* RB_CNTL: BYPASS */
	reg1(s, 0x0cc6u, 1u << 1);              /* RB_MODE_CNTL */
	reg1(s, 0x0ec3u, 0xc);                  /* SP_MODE_CNTL */
	reg1(s, 0x0e22u, 0x244u | (1u << 7));   /* TPL1_MODE_CNTL */
	reg1(s, 0x0e04u, 0x2);                  /* HLSQ_MODE_CNTL */
	reg1(s, 0xe000u, 0x181);                /* GRAS_CL_CNTL */
}

/* a linear byte copy as an R8 "image" of one row: n is a multiple of 64, src and dst 64-byte aligned */
static void
emitBlit(struct stream * s, uint64_t src, uint64_t dst, uint32_t n)
{
	uint32_t pitch = n >> 6;

	pkt7(s, CP_SET_RENDER_MODE, 1);
	put(s, 5);                              /* BLIT2D */
	pkt4(s, 0x2107u, 9);                    /* RB_2D_SRC_INFO .. */
	put(s, 0x3);
	put(s, (uint32_t)src);
	put(s, (uint32_t)(src >> 32));
	put(s, pitch | ((pitch * 2) << 16));
	for (int i = 0; i < 5; i++) {
		put(s, 0);
	}
	reg1(s, 0x2181u, 0x3);                  /* GRAS_2D_SRC_INFO */
	pkt4(s, 0x2110u, 9);                    /* RB_2D_DST_INFO .. */
	put(s, 0x3);
	put(s, (uint32_t)dst);
	put(s, (uint32_t)(dst >> 32));
	put(s, pitch | ((pitch * 2) << 16));
	for (int i = 0; i < 5; i++) {
		put(s, 0);
	}
	reg1(s, 0x2182u, 0x3);                  /* GRAS_2D_DST_INFO */
	pkt4(s, 0x2000u, 5);                    /* TPL1_A2D_SRC_TEXTURE_INFO .. */
	put(s, 0x3);
	put(s, n | (1u << 15));                 /* width n, height 1 */
	put(s, (uint32_t)src);
	put(s, (uint32_t)(src >> 32));
	put(s, pitch << 9);
	reg1(s, 0x2040u, 0x3u << 3);            /* SP_A2D_OUTPUT_INFO */
	pkt7(s, CP_BLIT, 5);
	put(s, 3);                              /* BLIT_OP_SCALE */
	put(s, 0);
	put(s, n - 1);
	put(s, 0);
	put(s, n - 1);
}

/* the end of a run of 2D blits: leave the 2D mode and push the colour cache out to memory */
static void
emit2DEnd(struct stream * s, uint32_t token)
{
	uint64_t flag = gStream.gpu + SCRATCH_OFFSET;

	pkt7(s, CP_SET_RENDER_MODE, 1);
	put(s, 8);                              /* END2D */
	pkt7(s, CP_WAIT_FOR_IDLE, 0);
	pkt7(s, CP_EVENT_WRITE, 4);
	put(s, 0x1du | (1u << 30));             /* PC_CCU_FLUSH_COLOR_TS */
	put(s, (uint32_t)flag);
	put(s, (uint32_t)(flag >> 32));
	put(s, token);
	pkt7(s, CP_WAIT_FOR_IDLE, 0);
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Metal's classes. Protocol-driven signatures: a message nobody implements is logged once and answered with zeros.     */
/* ------------------------------------------------------------------------------------------------------------------ */

static Class gDeviceBase;
static id gDevice;
static unsigned int gPort;
static Protocol * gProtocols[96];
static unsigned gProtocolCount;
static Class gBufferClass, gQueueClass, gCommandBufferClass, gBlitClass;
static Class gTextureClass, gRenderPipeClass, gRenderClass, gSamplerClass;

static void
keepProtocol(Protocol * p)
{
	if (gProtocolCount < sizeof(gProtocols) / sizeof(gProtocols[0])) {
		gProtocols[gProtocolCount++] = p;
	}
}

/* the protocol and Metal's private one of the same name (MTLBufferSPI ...), where this system has it */
static void
declareNamed(Class cls, const char * const * names)
{
	char spi[96];

	for (; *names != NULL; names++) {
		for (int private = 0; private < 2; private++) {
			Protocol * p;

			snprintf(spi, sizeof(spi), "%s%s", *names, private ? "SPI" : "");
			if ((p = objc_getProtocol(spi)) != NULL) {
				class_addProtocol(cls, p);
				keepProtocol(p);
			}
		}
	}
}

/* every protocol whose name begins with `prefix` (for the device and the queue, which have several private ones) */
static void
declareFamily(Class cls, const char * prefix)
{
	unsigned int count = 0;
	Protocol * __unsafe_unretained * all = objc_copyProtocolList(&count);

	for (unsigned int i = 0; i < count; i++) {
		if (strncmp(protocol_getName(all[i]), prefix, strlen(prefix)) == 0) {
			class_addProtocol(cls, all[i]);
			keepProtocol(all[i]);
		}
	}
	free(all);
}

static id
stubSignature(id self, SEL _cmd, SEL selector)
{
	for (unsigned i = 0; i < gProtocolCount; i++) {
		for (int required = 1; required >= 0; required--) {
			struct objc_method_description d = protocol_getMethodDescription(gProtocols[i], selector, required, YES);

			if (d.name != NULL && d.types != NULL) {
				return ((id (*)(Class, SEL, const char *))objc_msgSend)(objc_getClass("NSMethodSignature"),
				    sel_registerName("signatureWithObjCTypes:"), d.types);
			}
		}
	}
	struct objc_super sup = { self, class_getSuperclass(object_getClass(self)) };

	id sig;

	(void)_cmd;
	sig = ((id (*)(struct objc_super *, SEL, SEL))objc_msgSendSuper)(&sup, sel_registerName("methodSignatureForSelector:"), selector);
	if (sig == nil) {
		/* a selector no protocol declares (Metal's SPI): answered with an 8-byte zero rather than an exception, and logged */
		lavlog("unknown selector -%s on %s: answered with zero", sel_getName(selector), class_getName(object_getClass(self)));
		sig = ((id (*)(Class, SEL, const char *))objc_msgSend)(objc_getClass("NSMethodSignature"),
		    sel_registerName("signatureWithObjCTypes:"), "Q@:");
	}
	return sig;
}

static void
stubForward(id self, SEL _cmd, id invocation)
{
	static NSMutableSet * seen;
	SEL selector = ((SEL (*)(id, SEL))objc_msgSend)(invocation, sel_registerName("selector"));
	id signature = ((id (*)(id, SEL))objc_msgSend)(invocation, sel_registerName("methodSignature"));
	NSUInteger length = ((NSUInteger (*)(id, SEL))objc_msgSend)(signature, sel_registerName("methodReturnLength"));
	NSString * name = NSStringFromSelector(selector);

	(void)_cmd;
	@synchronized(gDeviceBase) {
		if (seen == nil) {
			seen = [NSMutableSet new];
		}
		if (![seen containsObject:name]) {
			[seen addObject:name];
			lavlog("asked and not implemented: -%s (%s; returns %lu bytes, answered with zero)", [name UTF8String],
			    class_getName(object_getClass(self)), (unsigned long)length);
		}
	}
	if (length != 0) {
		void * zero = calloc(1, length);

		((void (*)(id, SEL, void *))objc_msgSend)(invocation, sel_registerName("setReturnValue:"), zero);
		free(zero);
	}
}

static void
addStubForwarding(Class cls)
{
	class_addMethod(cls, sel_registerName("methodSignatureForSelector:"), (IMP)stubSignature, "@@::");
	class_addMethod(cls, sel_registerName("forwardInvocation:"), (IMP)stubForward, "v@:@");
}

/* each of our objects carries a block of state after its instance variables */
#define STATE(self, type) ((type *)object_getIndexedIvars(self))

static id
makeObject(Class cls, size_t extra)
{
	id o = class_createInstance(cls, extra);

	if (o != nil) {
		o = ((id (*)(id, SEL))objc_msgSend)(o, sel_registerName("init"));
	}
	return o;
}

static id
objDevice(__unused id self, __unused SEL _cmd)
{
	return gDevice;
}

/* ---- buffers ---- */

struct bufstate {
	struct lbuf b;
	uint64_t options;
	id label;
};

static id
deviceNewBuffer(__unused id self, __unused SEL _cmd, uint64_t length, uint64_t options)
{
	id o;
	struct bufstate * st;
	BOOL ok;

	if (length == 0 || length > (16ull << 20) || !lavOpen()) {
		lavlog("newBufferWithLength:%llu refused", (unsigned long long)length);
		return nil;
	}
	o = makeObject(gBufferClass, sizeof(struct bufstate));
	if (o == nil) {
		return nil;
	}
	st = STATE(o, struct bufstate);
	memset(st, 0, sizeof(*st));
	pthread_mutex_lock(&gGPULock);
	ok = lavBufNew((size_t)length, &st->b);
	pthread_mutex_unlock(&gGPULock);
	if (!ok) {
		return nil;
	}
	memset(st->b.cpu, 0, st->b.len);
	st->options = options;
	return o;
}

static id
deviceNewBufferBytes(id self, SEL _cmd, const void * bytes, uint64_t length, uint64_t options)
{
	id o = deviceNewBuffer(self, _cmd, length, options);

	if (o != nil && bytes != NULL) {
		memcpy(STATE(o, struct bufstate)->b.cpu, bytes, (size_t)length);
	}
	return o;
}

static void
bufferDealloc(id self, __unused SEL _cmd)
{
	struct bufstate * st = STATE(self, struct bufstate);
	struct objc_super sup = { self, class_getSuperclass(gBufferClass) };

	pthread_mutex_lock(&gGPULock);
	lavBufFree(&st->b);
	pthread_mutex_unlock(&gGPULock);
	[st->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static void *
bufferContents(id self, __unused SEL _cmd)
{
	return STATE(self, struct bufstate)->b.cpu;
}

static uint64_t
bufferLength(id self, __unused SEL _cmd)
{
	return STATE(self, struct bufstate)->b.len;
}

static uint64_t
bufferOptions(id self, __unused SEL _cmd)
{
	return STATE(self, struct bufstate)->options;
}

static uint64_t
bufferStorageMode(id self, __unused SEL _cmd)
{
	return (STATE(self, struct bufstate)->options >> 4) & 0xf;
}

static uint64_t
bufferCacheMode(id self, __unused SEL _cmd)
{
	return STATE(self, struct bufstate)->options & 0xf;
}

static uint64_t
bufferHazardMode(id self, __unused SEL _cmd)
{
	return (STATE(self, struct bufstate)->options >> 8) & 0x3;
}

static uint64_t
bufferGPUAddress(id self, __unused SEL _cmd)
{
	return STATE(self, struct bufstate)->b.gpu;
}

static void
bufferDidModify(__unused id self, __unused SEL _cmd, __unused NSRange r)
{
}

static uint64_t
bufferPurgeable(__unused id self, __unused SEL _cmd, __unused uint64_t state)
{
	return 1;                               /* MTLPurgeableStateNonVolatile */
}

static id
bufferLabel(id self, __unused SEL _cmd)
{
	return STATE(self, struct bufstate)->label;
}

static void
bufferSetLabel(id self, __unused SEL _cmd, id label)
{
	struct bufstate * st = STATE(self, struct bufstate);

	[st->label release];
	st->label = [label copy];
}

/* ---- queue, command buffer, blit encoder ---- */

enum opkind { OP_FILL, OP_COPY, OP_DISPATCH, OP_CLEAR, OP_DRAW, OP_TEXCOPY };

/* a compute dispatch as the encoder had it when it was encoded: the pipeline, what is bound, the grid */
struct dispatch {
	id pipeline;                            /* retained */
	id buf[LAV5_MAX_BUFFERS];               /* retained */
	uint64_t off[LAV5_MAX_BUFFERS];
	uint8_t * bytes[LAV5_MAX_BUFFERS];      /* setBytes:, a copy */
	uint32_t blen[LAV5_MAX_BUFFERS];
	uint32_t groups[3], local[3];
};

/* a draw as the render encoder had it when it was encoded */
struct draw {
	id pipeline;                            /* retained */
	id target;                              /* retained texture */
	id xtarget[8];                          /* retained: colour targets 1.. */
	id vbuf[LAV5_MAX_BUFFERS];              /* retained: setVertexBuffer: */
	uint64_t voff[LAV5_MAX_BUFFERS];
	uint8_t * vbytes[LAV5_MAX_BUFFERS];     /* setVertexBytes:, a copy */
	uint32_t vblen[LAV5_MAX_BUFFERS];
	id fbuf[LAV5_MAX_BUFFERS];
	uint64_t foff[LAV5_MAX_BUFFERS];
	uint8_t * fbytes[LAV5_MAX_BUFFERS];
	uint32_t fblen[LAV5_MAX_BUFFERS];
	uint32_t prim, first, count, instances, base_instance;
	id ibuf;                                /* retained: the index buffer of an indexed draw */
	uint64_t ioff;
	uint32_t isize;                         /* 0 not indexed, 1 16-bit, 2 32-bit */
	float viewport[6];
	uint32_t scissor[4];
	uint32_t cull, front_cw;
	id ftex[16], fsamp[16];                 /* retained */
	float blend_color[4];
};

struct op {
	enum opkind kind;
	struct draw * draw;
	double clear[4];
	id src, dst;                    /* retained buffers */
	uint64_t srcOff, dstOff, size;
	uint8_t value;
	struct dispatch * disp;
	uint64_t cx, cy, cw, ch, cdx, cdy;      /* OP_TEXCOPY: source origin and size, destination origin (level 0) */
};

struct cbstate {
	id queue;
	NSMutableArray * handlers;
	NSMutableArray * scheduled;
	NSMutableArray * drawables;     /* presentDrawable: */
	struct op * ops;
	size_t nops, cap;
	int status;                     /* MTLCommandBufferStatus */
	id label;
};

struct encstate {
	id commandBuffer;
	BOOL ended;
	id label;
};

static id
deviceNewQueue(__unused id self, __unused SEL _cmd)
{
	if (!lavOpen()) {
		return nil;
	}
	return makeObject(gQueueClass, 0);
}

static id
deviceNewQueueMax(id self, SEL _cmd, __unused uint64_t max)
{
	return deviceNewQueue(self, _cmd);
}

static id
queueCommandBuffer(id self, __unused SEL _cmd)
{
	id o = makeObject(gCommandBufferClass, sizeof(struct cbstate));
	struct cbstate * st;

	if (o == nil) {
		return nil;
	}
	st = STATE(o, struct cbstate);
	memset(st, 0, sizeof(*st));
	st->queue = [self retain];
	st->handlers = [NSMutableArray new];
	st->scheduled = [NSMutableArray new];
	st->status = 1;                         /* Enqueued */
	return [o autorelease];                 /* not a new/copy method: the caller owns nothing */
}

/* _MTLCommandQueue's own dealloc releases a dispatch queue this object never made (-> dispatch_release(NULL)): skip it,
 * go to the class above it */
static void
queueDealloc(id self, __unused SEL _cmd)
{
	struct objc_super sup = { self, class_getSuperclass(class_getSuperclass(gQueueClass)) };

	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static id
queueLabel(__unused id self, __unused SEL _cmd)
{
	return nil;
}

/* MTLCommandQueueSPI, which CARenderer calls: completion handlers run on the thread that finishes the command buffer here, so the
 * queue given is accepted and not used */
static void
queueSetCompletionQueue(__unused id self, __unused SEL _cmd, __unused id q)
{
}

static id
queueCompletionQueue(__unused id self, __unused SEL _cmd)
{
	return nil;
}

static void
cbFreeOps(struct cbstate * st)
{
	for (size_t i = 0; i < st->nops; i++) {
		[st->ops[i].src release];
		[st->ops[i].dst release];
		if (st->ops[i].draw != NULL) {
			struct draw * w = st->ops[i].draw;

			[w->ibuf release];

			[w->pipeline release];
			[w->target release];
			for (int j = 1; j < 8; j++) {
				[w->xtarget[j] release];
			}
			for (int j = 0; j < 16; j++) {
				[w->ftex[j] release];
				[w->fsamp[j] release];
			}
			for (int j = 0; j < LAV5_MAX_BUFFERS; j++) {
				[w->vbuf[j] release];
				[w->fbuf[j] release];
				free(w->vbytes[j]);
				free(w->fbytes[j]);
			}
			free(w);
		}
		if (st->ops[i].disp != NULL) {
			struct dispatch * d = st->ops[i].disp;

			[d->pipeline release];
			for (int j = 0; j < LAV5_MAX_BUFFERS; j++) {
				[d->buf[j] release];
				free(d->bytes[j]);
			}
			free(d);
		}
	}
	free(st->ops);
	st->ops = NULL;
	st->nops = st->cap = 0;
}

static void
cbDealloc(id self, __unused SEL _cmd)
{
	struct cbstate * st = STATE(self, struct cbstate);
	struct objc_super sup = { self, class_getSuperclass(gCommandBufferClass) };

	cbFreeOps(st);
	[st->queue release];
	[st->handlers release];
	[st->scheduled release];
	[st->drawables release];
	[st->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static struct op *
cbAddOp(struct cbstate * st)
{
	if (st->nops == st->cap) {
		st->cap = st->cap ? st->cap * 2 : 16;
		st->ops = realloc(st->ops, st->cap * sizeof(struct op));
	}
	memset(&st->ops[st->nops], 0, sizeof(struct op));
	return &st->ops[st->nops++];
}

static id
cbBlitEncoder(id self, __unused SEL _cmd)
{
	id e = makeObject(gBlitClass, sizeof(struct encstate));
	struct encstate * es;

	if (e == nil) {
		return nil;
	}
	es = STATE(e, struct encstate);
	memset(es, 0, sizeof(*es));
	es->commandBuffer = [self retain];
	return [e autorelease];
}

static void
encDealloc(id self, __unused SEL _cmd)
{
	struct encstate * es = STATE(self, struct encstate);
	struct objc_super sup = { self, class_getSuperclass(gBlitClass) };

	[es->commandBuffer release];
	[es->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static void
encFill(id self, __unused SEL _cmd, id buffer, NSRange range, uint8_t value)
{
	struct cbstate * st = STATE(STATE(self, struct encstate)->commandBuffer, struct cbstate);
	struct op * op = cbAddOp(st);

	op->kind = OP_FILL;
	op->dst = [buffer retain];
	op->dstOff = range.location;
	op->size = range.length;
	op->value = value;
}

static void
encCopy(id self, __unused SEL _cmd, id src, uint64_t srcOff, id dst, uint64_t dstOff, uint64_t size)
{
	struct cbstate * st = STATE(STATE(self, struct encstate)->commandBuffer, struct cbstate);
	struct op * op = cbAddOp(st);

	op->kind = OP_COPY;
	op->src = [src retain];
	op->dst = [dst retain];
	op->srcOff = srcOff;
	op->dstOff = dstOff;
	op->size = size;
}

static void
encEnd(id self, __unused SEL _cmd)
{
	STATE(self, struct encstate)->ended = YES;
}

static void
encNothing(__unused id self, __unused SEL _cmd, __unused id resource)
{
	/* synchronizeResource: the memory is the CPU's and the GPU's both; there is nothing to copy back */
}

static id
encLabel(id self, __unused SEL _cmd)
{
	return STATE(self, struct encstate)->label;
}

static void
encSetLabel(id self, __unused SEL _cmd, id label)
{
	struct encstate * es = STATE(self, struct encstate);

	[es->label release];
	es->label = [label copy];
}

/* run one op: the GPU does what it can, the CPU the unaligned edges */
static BOOL
runFill(struct op * op)
{
	struct bufstate * d = STATE(op->dst, struct bufstate);
	uint8_t * cpu = (uint8_t *)d->b.cpu + op->dstOff;
	uint64_t gpu = d->b.gpu + op->dstOff;
	uint64_t size = op->size;
	uint64_t head = (4 - (gpu & 3)) & 3;
	uint64_t body, tail;
	uint32_t pattern = op->value * 0x01010101u;

	if (op->dstOff + size > d->b.len) {
		lavlog("fill beyond the buffer (%llu + %llu > %zu)", (unsigned long long)op->dstOff, (unsigned long long)size, d->b.len);
		return NO;
	}
	if (head > size) {
		head = size;
	}
	body = (size - head) & ~3ull;
	tail = size - head - body;
	if (head != 0 || tail != 0) {
		lavlog("fill: %llu head and %llu tail bytes by the CPU", (unsigned long long)head, (unsigned long long)tail);
	}
	memset(cpu, op->value, (size_t)head);
	if (body != 0) {
		struct stream * s = calloc(1, sizeof(*s));
		uint32_t words[1000];

		for (size_t i = 0; i < 1000; i++) {
			words[i] = pattern;
		}
		for (uint64_t done = 0; done < body;) {
			s->n = 0;
			while (done < body && s->n + 1010 < STREAM_WORDS) {
				size_t k = (size_t)((body - done) / 4);

				if (k > 1000) {
					k = 1000;
				}
				emitWrite(s, gpu + head + done, words, k);
				done += k * 4;
			}
			if (!lavSubmit(s->w, s->n)) {
				free(s);
				return NO;
			}
		}
		free(s);
	}
	memset(cpu + head + body, op->value, (size_t)tail);
	return YES;
}

#define BLIT_CHUNK 4096u

static BOOL
runCopy(struct op * op)
{
	struct bufstate * sb = STATE(op->src, struct bufstate);
	struct bufstate * db = STATE(op->dst, struct bufstate);
	uint8_t * scpu = (uint8_t *)sb->b.cpu + op->srcOff;
	uint8_t * dcpu = (uint8_t *)db->b.cpu + op->dstOff;
	uint64_t sgpu = sb->b.gpu + op->srcOff;
	uint64_t dgpu = db->b.gpu + op->dstOff;
	uint64_t size = op->size, head = 0, body = 0, tail = 0;

	if (op->srcOff + size > sb->b.len || op->dstOff + size > db->b.len) {
		lavlog("copy beyond a buffer");
		return NO;
	}
	if (((sgpu ^ dgpu) & 63) == 0) {
		head = (64 - (dgpu & 63)) & 63;
		if (head > size) {
			head = size;
		}
		body = (size - head) & ~63ull;
	}
	tail = size - head - body;
	if (body == 0) {
		memmove(dcpu, scpu, (size_t)size);
		lavlog("copy of %llu bytes entirely by the CPU (too small or misaligned)", (unsigned long long)size);
		return YES;
	}
	if (head != 0 || tail != 0) {
		lavlog("copy: %llu head and %llu tail bytes by the CPU", (unsigned long long)head, (unsigned long long)tail);
	}
	memmove(dcpu, scpu, (size_t)head);
	{
		struct stream * s = calloc(1, sizeof(*s));
		uint64_t done = 0;
		BOOL ok = YES;

		while (done < body && ok) {
			s->n = 0;
			emit2DSetup(s);
			while (done < body && s->n + 140 < STREAM_WORDS) {
				uint32_t n = (uint32_t)((body - done) > BLIT_CHUNK ? BLIT_CHUNK : (body - done));

				emitBlit(s, sgpu + head + done, dgpu + head + done, n);
				done += n;
			}
			emit2DEnd(s, 0x2d2d0000u | (uint32_t)(done >> 6));
			ok = lavSubmit(s->w, s->n);
		}
		free(s);
		if (!ok) {
			return NO;
		}
	}
	memmove(dcpu + head + body, scpu + head + body, (size_t)tail);
	return YES;
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Compute: libraries, functions, pipeline states, the compute encoder (M5.3 step d)                                    */
/* The shader compiler is hakios-metal's AIR front end and Mesa's ir3, in this process (lavcc.cpp); the dispatch is     */
/* lava5.c's A5xx stream. A kernel it cannot serve (textures, threadgroup memory ...) is refused when its pipeline is    */
/* made, by name, in the log.                                                                                           */
/* ------------------------------------------------------------------------------------------------------------------ */

static Class gLibraryClass, gFunctionClass, gPipelineClass, gComputeClass;
static pthread_mutex_t gCCLock = PTHREAD_MUTEX_INITIALIZER;
static struct lbuf gBytes;                  /* the bytes setBytes: gave, for the dispatch being run */

struct libstate {
	struct lavcc_lib * lib;
	uint64_t hash;                          /* of the metallib bytes: the key of the pipeline cache */
	id label;
};

struct fnstate {
	id library;                             /* retained */
	NSString * name;
	int type;                               /* the metallib's: 0 vertex, 1 fragment, 2 kernel */
	id label;
	struct lavcc_const * consts;            /* the function constants the pipeline specialised it with (malloc'd), or NULL */
	unsigned nconsts;
};

struct pipestate {
	id function;                            /* retained */
	struct lav5_kernel k;
	struct lbuf code;
	id label;
};

struct cenc {
	id commandBuffer;                       /* retained */
	BOOL ended;
	id label;
	id pipeline;                            /* retained */
	id buf[LAV5_MAX_BUFFERS];               /* retained */
	uint64_t off[LAV5_MAX_BUFFERS];
	uint8_t * bytes[LAV5_MAX_BUFFERS];
	uint32_t blen[LAV5_MAX_BUFFERS];
};

typedef struct { uint64_t width, height, depth; } LavSize;

static void
lavError(NSError ** err, NSString * domain, const char * why)
{
	if (err != NULL) {
		*err = [NSError errorWithDomain:domain code:1
		    userInfo:@{ NSLocalizedDescriptionKey: [NSString stringWithUTF8String:why] }];
	}
}

static uint64_t
lavHash64(uint64_t h, const void * p, size_t n)
{
	const uint8_t * b = p;

	for (size_t i = 0; i < n; i++) {
		h = (h ^ b[i]) * 0x100000001b3ull;
	}
	return h;
}

static id
makeLibrary(const void * bytes, size_t size, NSError ** err)
{
	char why[256] = "";
	struct lavcc_lib * lib;
	id o;

	pthread_mutex_lock(&gCCLock);
	lib = lavcc_open(bytes, size, why, sizeof(why));
	pthread_mutex_unlock(&gCCLock);
	if (lib == NULL) {
		lavlog("library of %zu bytes refused: %s", size, why);
		lavError(err, @"MTLLibraryErrorDomain", why);
		return nil;
	}
	o = makeObject(gLibraryClass, sizeof(struct libstate));
	if (o == nil) {
		lavcc_close(lib);
		return nil;
	}
	memset(STATE(o, struct libstate), 0, sizeof(struct libstate));
	STATE(o, struct libstate)->lib = lib;
	STATE(o, struct libstate)->hash = lavHash64(0xcbf29ce484222325ull, bytes, size);
	lavlog("library of %zu bytes, %u functions", size, lavcc_function_count(lib));
	return o;
}

static id
deviceNewLibraryData(__unused id self, __unused SEL _cmd, id data, NSError ** err)
{
	const void * p = NULL;
	size_t n = 0;
	id map, lib;

	if (data == nil) {
		lavError(err, @"MTLLibraryErrorDomain", "no data");
		return nil;
	}
	map = (id)dispatch_data_create_map((dispatch_data_t)data, &p, &n);
	lib = makeLibrary(p, n, err);
	[map release];
	return lib;
}

static id
deviceNewLibraryURL(__unused id self, __unused SEL _cmd, id url, NSError ** err)
{
	NSData * d = url != nil ? [NSData dataWithContentsOfURL:url] : nil;

	if (d == nil) {
		lavError(err, @"MTLLibraryErrorDomain", "the library file cannot be read");
		return nil;
	}
	return makeLibrary([d bytes], [d length], err);
}

static id
deviceNewPipelineLibrary(__unused id self, __unused SEL _cmd, id path, NSError ** err)
{
	lavlog("newPipelineLibraryWithFilePath: %s: pipeline libraries are not supported", path != nil ? [[path description] UTF8String] : "(nil)");
	lavError(err, @"MTLLibraryErrorDomain", "pipeline libraries are not supported");
	return nil;
}

static id
deviceNewLibraryFile(__unused id self, __unused SEL _cmd, id path, NSError ** err)
{
	NSData * d = path != nil ? [NSData dataWithContentsOfFile:path] : nil;

	if (d == nil) {
		lavError(err, @"MTLLibraryErrorDomain", "the library file cannot be read");
		return nil;
	}
	return makeLibrary([d bytes], [d length], err);
}

static id
deviceNewLibrarySource(__unused id self, __unused SEL _cmd, __unused id source, __unused id options, NSError ** err)
{
	lavlog("newLibraryWithSource: there is no Metal source compiler on this device");
	lavError(err, @"MTLLibraryErrorDomain", "this device has no Metal source compiler; give it a .metallib");
	return nil;
}

static id
deviceNoLibrary(id self, SEL _cmd)
{
	NSBundle * b = [NSBundle mainBundle];
	NSString * path = b != nil ? [b pathForResource:@"default" ofType:@"metallib"] : nil;

	if (path != nil) {
		return deviceNewLibraryFile(self, _cmd, path, NULL);
	}
	lavlog("newDefaultLibrary: no default library");
	return nil;
}

static id
deviceNoLibraryBundle(id self, SEL _cmd, id bundle, NSError ** err)
{
	NSString * path = [bundle isKindOfClass:[NSBundle class]]
	    ? [(NSBundle *)bundle pathForResource:@"default" ofType:@"metallib"]
	    : nil;

	if (path != nil) {
		return deviceNewLibraryFile(self, _cmd, path, err);
	}
	lavlog("newDefaultLibraryWithBundle: no default library");
	lavError(err, @"MTLLibraryErrorDomain", "no default library");
	return nil;
}

static void
libDealloc(id self, __unused SEL _cmd)
{
	struct libstate * st = STATE(self, struct libstate);
	struct objc_super sup = { self, class_getSuperclass(gLibraryClass) };

	if (st->lib != NULL) {
		lavcc_close(st->lib);
	}
	[st->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static id
libFunctionNames(id self, __unused SEL _cmd)
{
	struct lavcc_lib * lib = STATE(self, struct libstate)->lib;
	NSMutableArray * a = [NSMutableArray array];

	for (unsigned i = 0; i < lavcc_function_count(lib); i++) {
		[a addObject:[NSString stringWithUTF8String:lavcc_function_name(lib, i, NULL)]];
	}
	return a;
}

static id
libNewFunction(id self, __unused SEL _cmd, id name)
{
	struct lavcc_lib * lib = STATE(self, struct libstate)->lib;
	id o;

	for (unsigned i = 0; name != nil && i < lavcc_function_count(lib); i++) {
		int type = 0;
		const char * n = lavcc_function_name(lib, i, &type);

		if (strcmp(n, [name UTF8String]) == 0) {
			struct fnstate * fs;

			o = makeObject(gFunctionClass, sizeof(struct fnstate));
			if (o == nil) {
				return nil;
			}
			fs = STATE(o, struct fnstate);
			memset(fs, 0, sizeof(*fs));
			fs->library = [self retain];
			fs->name = [name copy];
			fs->type = type;
			return o;
		}
	}
	lavlog("newFunctionWithName:%s: not in the library", name != nil ? [name UTF8String] : "(nil)");
	return nil;
}

/* Function constants. MTLFunctionConstantValues has no getter, so the setters are wrapped: each value set is kept on the object
 * (an associated NSMutableData of struct lavcc_const) and copied into the function made from it, to specialise the compile. */
static char gFCKey;
static IMP gFCSetAt, gFCSetRange, gFCSetName, gFCReset, gFCCopy;

static BOOL
fcBits(const void * v, uint64_t type, uint64_t * bits, unsigned * esize)
{
	unsigned n = 0, w = 0;
	uint64_t b = 0;

	if (type >= 3 && type <= 6) { n = (unsigned)type - 2; w = 32; }
	else if (type >= 16 && type <= 19) { n = (unsigned)type - 15; w = 16; }
	else if (type >= 29 && type <= 32) { n = (unsigned)type - 28; w = 32; }
	else if (type >= 33 && type <= 36) { n = (unsigned)type - 32; w = 32; }
	else if (type >= 37 && type <= 40) { n = (unsigned)type - 36; w = 16; }
	else if (type >= 41 && type <= 44) { n = (unsigned)type - 40; w = 16; }
	else if (type >= 45 && type <= 48) { n = (unsigned)type - 44; w = 8; }
	else if (type >= 49 && type <= 52) { n = (unsigned)type - 48; w = 8; }
	else if (type >= 53 && type <= 56) { n = (unsigned)type - 52; w = 8; }
	else {
		return NO;
	}
	*esize = n * w / 8;
	if (type >= 53) {
		for (unsigned i = 0; i < n; i++) {
			b |= (uint64_t)(((const uint8_t *)v)[i] != 0) << i;
		}
	} else if (n * w <= 64) {
		memcpy(&b, v, n * w / 8);
	} else {
		return NO;
	}
	*bits = b;
	return YES;
}

/* one value set on a MTLFunctionConstantValues: by index, or (NAME not empty) by name, which a function resolves to its own index */
struct fcrec {
	uint32_t index;
	uint64_t bits;
	char name[56];
};

static void
fcRecord(id obj, uint32_t index, uint64_t bits, const char * name)
{
	@synchronized (obj) {
		NSMutableData * d = objc_getAssociatedObject(obj, &gFCKey);
		struct fcrec e;

		memset(&e, 0, sizeof(e));
		e.index = index;
		e.bits = bits;
		if (name != NULL) {
			snprintf(e.name, sizeof(e.name), "%s", name);
		}
		if (d == nil) {
			d = [NSMutableData data];
			objc_setAssociatedObject(obj, &gFCKey, d, OBJC_ASSOCIATION_RETAIN);
		}
		struct fcrec * c = [d mutableBytes];
		for (NSUInteger i = 0; i < [d length] / sizeof(*c); i++) {
			if (name != NULL ? strcmp(c[i].name, e.name) == 0 : (c[i].name[0] == 0 && c[i].index == index)) {
				c[i].bits = bits;
				return;
			}
		}
		[d appendBytes:&e length:sizeof(e)];
	}
}

static void
fcSetAt(id self, SEL cmd, const void * value, uint64_t type, uint64_t index)
{
	uint64_t bits;
	unsigned es;

	if (value != NULL && fcBits(value, type, &bits, &es)) {
		fcRecord(self, (uint32_t)index, bits, NULL);
	} else {
		lavlog("function constant %llu: type %llu not served", (unsigned long long)index, (unsigned long long)type);
	}
	((void (*)(id, SEL, const void *, uint64_t, uint64_t))gFCSetAt)(self, cmd, value, type, index);
}

static void
fcSetRange(id self, SEL cmd, const void * values, uint64_t type, NSRange range)
{
	uint64_t bits;
	unsigned es;
	const void * p = values;

	for (NSUInteger i = 0; p != NULL && i < range.length; i++) {
		if (!fcBits(p, type, &bits, &es)) {
			break;
		}
		fcRecord(self, (uint32_t)(range.location + i), bits, NULL);
		p = (const uint8_t *)p + es;
	}
	((void (*)(id, SEL, const void *, uint64_t, NSRange))gFCSetRange)(self, cmd, values, type, range);
}

static void
fcSetName(id self, SEL cmd, const void * value, uint64_t type, id name)
{
	uint64_t bits;
	unsigned es;

	if (value != NULL && name != nil && fcBits(value, type, &bits, &es)) {
		fcRecord(self, 0xffffffffu, bits, [name UTF8String]);
	} else {
		lavlog("function constant by name %s (type %llu): not served", name != nil ? [name UTF8String] : "(nil)", (unsigned long long)type);
	}
	((void (*)(id, SEL, const void *, uint64_t, id))gFCSetName)(self, cmd, value, type, name);
}

static void
fcReset(id self, SEL cmd)
{
	objc_setAssociatedObject(self, &gFCKey, nil, OBJC_ASSOCIATION_RETAIN);
	((void (*)(id, SEL))gFCReset)(self, cmd);
}

static id
fcCopy(id self, SEL cmd, void * zone)
{
	id c = ((id (*)(id, SEL, void *))gFCCopy)(self, cmd, zone);
	NSData * d = objc_getAssociatedObject(self, &gFCKey);

	if (c != nil && d != nil) {
		objc_setAssociatedObject(c, &gFCKey, [[d mutableCopy] autorelease], OBJC_ASSOCIATION_RETAIN);
	}
	return c;
}

static void
fcHook(Class cls, SEL sel, IMP imp, IMP * orig)
{
	Method m = class_getInstanceMethod(cls, sel);

	if (m == NULL) {
		lavlog("function constants: %s is not on %s", sel_getName(sel), class_getName(cls));
		return;
	}
	*orig = method_setImplementation(m, imp);
}

static void
installConstantHooks(void)
{
	Class cls = NSClassFromString(@"MTLFunctionConstantValuesInternal");

	if (cls == Nil) {
		cls = NSClassFromString(@"MTLFunctionConstantValues");
	}
	if (cls == Nil) {
		lavlog("function constants: no MTLFunctionConstantValues class");
		return;
	}
	fcHook(cls, sel_registerName("setConstantValue:type:atIndex:"), (IMP)fcSetAt, &gFCSetAt);
	fcHook(cls, sel_registerName("setConstantValues:type:withRange:"), (IMP)fcSetRange, &gFCSetRange);
	fcHook(cls, sel_registerName("setConstantValue:type:withName:"), (IMP)fcSetName, &gFCSetName);
	fcHook(cls, sel_registerName("reset"), (IMP)fcReset, &gFCReset);
	fcHook(cls, sel_registerName("copyWithZone:"), (IMP)fcCopy, &gFCCopy);
	lavlog("function constants: setters of %s wrapped", class_getName(cls));
}

/* give function F the constants recorded on the values object CONSTANTS (names resolved to F's own indices) */
static void
fnTakeConstants(id f, id constants)
{
	struct fnstate * st;
	NSData * d;
	struct lavcc_fcname decl[128];
	unsigned ndecl, nrec;
	const struct fcrec * rec;

	if (f == nil || constants == nil) {
		return;
	}
	st = STATE(f, struct fnstate);
	@synchronized (constants) {
		d = [[objc_getAssociatedObject(constants, &gFCKey) copy] autorelease];
	}
	free(st->consts);
	st->consts = NULL;
	st->nconsts = 0;
	if (d == nil || [d length] == 0) {
		lavlog("function %s: no function constants recorded", [st->name UTF8String]);
		return;
	}
	nrec = (unsigned)([d length] / sizeof(struct fcrec));
	rec = [d bytes];
	ndecl = lavcc_function_constants(STATE(st->library, struct libstate)->lib, [st->name UTF8String], decl, 128);
	st->consts = calloc(nrec, sizeof(struct lavcc_const));
	for (unsigned i = 0; i < nrec; i++) {
		int64_t idx = rec[i].index;

		if (rec[i].name[0] != 0) {
			idx = -1;
			for (unsigned k = 0; k < ndecl; k++) {
				if (strcmp(decl[k].name, rec[i].name) == 0) {
					idx = decl[k].index;
					break;
				}
			}
		}
		if (idx < 0) {
			continue;       /* a constant of another function of the same set */
		}
		st->consts[st->nconsts++] = (struct lavcc_const){ (uint32_t)idx, rec[i].bits };
	}
	{
		char b[600];
		size_t o = 0;

		for (unsigned i = 0; i < st->nconsts && o < sizeof(b) - 40; i++) {
			o += (size_t)snprintf(b + o, sizeof(b) - o, " %u=%llx", st->consts[i].index, (unsigned long long)st->consts[i].bits);
		}
		lavlog("function %s: %u declared, %u set, specialised with %u:%s", [st->name UTF8String], ndecl, nrec, st->nconsts, b);
	}
}

static id
libNewFunctionConstants(id self, SEL cmd, id name, id constants, NSError ** err)
{
	id f = libNewFunction(self, cmd, name);

	fnTakeConstants(f, constants);
	if (f == nil) {
		lavError(err, @"MTLLibraryErrorDomain", "no such function");
	}
	return f;
}

static id
libNewFunctionDescriptor(id self, SEL cmd, id desc, NSError ** err)
{
	id f = libNewFunction(self, cmd, desc != nil ? [desc name] : nil);

	if (desc != nil && [desc respondsToSelector:@selector(constantValues)]) {
		fnTakeConstants(f, [desc constantValues]);
	}

	if (f == nil) {
		lavError(err, @"MTLLibraryErrorDomain", "no such function");
	}
	return f;
}

static id
fnNewSpecialized(id self, __unused SEL _cmd, __unused id desc, __unused id archive, __unused id cache, NSError ** err)
{
	if (err != NULL) {
		*err = nil;
	}
	return [self retain];
}

static id
libLabel(id self, __unused SEL _cmd)
{
	return STATE(self, struct libstate)->label;
}

static void
libSetLabel(id self, __unused SEL _cmd, id label)
{
	struct libstate * st = STATE(self, struct libstate);

	[st->label release];
	st->label = [label copy];
}

static void
fnDealloc(id self, __unused SEL _cmd)
{
	struct fnstate * st = STATE(self, struct fnstate);
	struct objc_super sup = { self, class_getSuperclass(gFunctionClass) };

	[st->library release];
	[st->name release];
	[st->label release];
	free(st->consts);
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static id
fnName(id self, __unused SEL _cmd)
{
	return STATE(self, struct fnstate)->name;
}

static uint64_t
fnType(id self, __unused SEL _cmd)
{
	return (uint64_t)STATE(self, struct fnstate)->type + 1;       /* MTLFunctionType: vertex 1, fragment 2, kernel 3 */
}

static id
fnLabel(id self, __unused SEL _cmd)
{
	return STATE(self, struct fnstate)->label;
}

static void
fnSetLabel(id self, __unused SEL _cmd, id label)
{
	struct fnstate * st = STATE(self, struct fnstate);

	[st->label release];
	st->label = [label copy];
}

static id
makePipeline(id function, NSError ** err)
{
	struct fnstate * fs;
	struct libstate * ls;
	struct pipestate * ps;
	char why[256] = "";
	const char * refuse;
	struct lav5_kernel k;
	id o;
	size_t codelen;

	if (function == nil || !lavOpen()) {
		lavError(err, @"MTLLibraryErrorDomain", "no function, or no GPU");
		return nil;
	}
	fs = STATE(function, struct fnstate);
	ls = STATE(fs->library, struct libstate);
	if (fs->type != 2) {
		lavlog("pipeline of %s: not a kernel function", [fs->name UTF8String]);
		lavError(err, @"MTLLibraryErrorDomain", "not a kernel function");
		return nil;
	}
	pthread_mutex_lock(&gCCLock);
	if (!lavcc_compile(ls->lib, [fs->name UTF8String], &k, why, sizeof(why))) {
		pthread_mutex_unlock(&gCCLock);
		lavlog("pipeline of %s: the compiler refused it: %s", [fs->name UTF8String], why);
		lavError(err, @"MTLLibraryErrorDomain", why);
		return nil;
	}
	pthread_mutex_unlock(&gCCLock);
	if ((refuse = lav5_refuse_kernel(&k)) != NULL) {
		lavlog("pipeline of %s: %s", [fs->name UTF8String], refuse);
		lavError(err, @"MTLLibraryErrorDomain", refuse);
		lavcc_kernel_free(&k);
		return nil;
	}
	o = makeObject(gPipelineClass, sizeof(struct pipestate));
	if (o == nil) {
		lavcc_kernel_free(&k);
		return nil;
	}
	ps = STATE(o, struct pipestate);
	memset(ps, 0, sizeof(*ps));
	ps->k = k;
	codelen = (size_t)k.instrlen * 128;
	if (codelen < (size_t)k.nwords * 4) {
		codelen = (size_t)k.nwords * 4;
	}
	pthread_mutex_lock(&gGPULock);
	if (!lavBufNew((codelen + 4095) & ~(size_t)4095, &ps->code)) {
		pthread_mutex_unlock(&gGPULock);
		lavcc_kernel_free(&ps->k);
		ps->function = nil;
		lavError(err, @"MTLLibraryErrorDomain", "no memory for the code");
		[o release];
		return nil;
	}
	pthread_mutex_unlock(&gGPULock);
	memset(ps->code.cpu, 0, ps->code.len);
	memcpy(ps->code.cpu, k.words, (size_t)k.nwords * 4);
	ps->function = [function retain];
	lavlog("pipeline %s: %u words, instrlen %u, constlen %u, %u ssbos, %u ranges, code at GPU %#llx", k.name, k.nwords,
	    k.instrlen, k.constlen, k.num_ssbos, k.nranges, (unsigned long long)ps->code.gpu);
	return o;
}

static id
deviceNewPipeline(__unused id self, __unused SEL _cmd, id function, NSError ** err)
{
	return makePipeline(function, err);
}

static id
deviceNewPipelineOptions(__unused id self, __unused SEL _cmd, id function, __unused uint64_t options, id * reflection, NSError ** err)
{
	if (reflection != NULL) {
		*reflection = nil;
	}
	return makePipeline(function, err);
}

static id
deviceNewPipelineDescriptor(__unused id self, __unused SEL _cmd, id descriptor, __unused uint64_t options, id * reflection, NSError ** err)
{
	id function = ((id (*)(id, SEL))objc_msgSend)(descriptor, sel_registerName("computeFunction"));

	if (reflection != NULL) {
		*reflection = nil;
	}
	return makePipeline(function, err);
}

typedef void (^LavPipelineDone)(id, NSError *);

static void
deviceNewPipelineAsync(__unused id self, __unused SEL _cmd, id function, id handler)
{
	NSError * err = nil;
	id p = makePipeline(function, &err);

	((LavPipelineDone)handler)(p, err);
	[p release];
}

static void
pipeDealloc(id self, __unused SEL _cmd)
{
	struct pipestate * st = STATE(self, struct pipestate);
	struct objc_super sup = { self, class_getSuperclass(gPipelineClass) };

	pthread_mutex_lock(&gGPULock);
	lavBufFree(&st->code);
	pthread_mutex_unlock(&gGPULock);
	lavcc_kernel_free(&st->k);
	[st->function release];
	[st->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static uint64_t
pipeMaxThreads(__unused id self, __unused SEL _cmd)
{
	return 256;
}

static uint64_t
pipeWidth(id self, __unused SEL _cmd)
{
	return STATE(self, struct pipestate)->k.double_threadsize ? 64 : 32;
}

static uint64_t
pipeStaticShared(__unused id self, __unused SEL _cmd)
{
	return 0;
}

static id
pipeLabel(id self, __unused SEL _cmd)
{
	return STATE(self, struct pipestate)->label;
}

static void
pipeSetLabel(id self, __unused SEL _cmd, id label)
{
	struct pipestate * st = STATE(self, struct pipestate);

	[st->label release];
	st->label = [label copy];
}

/* ---- the compute encoder ---- */

static id
cbComputeEncoder(id self, __unused SEL _cmd)
{
	id e = makeObject(gComputeClass, sizeof(struct cenc));
	struct cenc * ce;

	if (e == nil) {
		return nil;
	}
	ce = STATE(e, struct cenc);
	memset(ce, 0, sizeof(*ce));
	ce->commandBuffer = [self retain];
	return [e autorelease];
}

static id
cbComputeEncoderDispatchType(id self, SEL cmd, __unused uint64_t type)
{
	return cbComputeEncoder(self, cmd);
}

static id
cbComputeEncoderDescriptor(id self, SEL cmd, __unused id descriptor)
{
	return cbComputeEncoder(self, cmd);
}

static void
cencDealloc(id self, __unused SEL _cmd)
{
	struct cenc * ce = STATE(self, struct cenc);
	struct objc_super sup = { self, class_getSuperclass(gComputeClass) };

	for (int i = 0; i < LAV5_MAX_BUFFERS; i++) {
		[ce->buf[i] release];
		free(ce->bytes[i]);
	}
	[ce->pipeline release];
	[ce->commandBuffer release];
	[ce->label release];
	((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

static void
cencSetPipeline(id self, __unused SEL _cmd, id pipeline)
{
	struct cenc * ce = STATE(self, struct cenc);

	[pipeline retain];
	[ce->pipeline release];
	ce->pipeline = pipeline;
}

static void
cencSetBuffer(id self, __unused SEL _cmd, id buffer, uint64_t offset, uint64_t index)
{
	struct cenc * ce = STATE(self, struct cenc);

	if (index >= LAV5_MAX_BUFFERS) {
		lavlog("setBuffer at index %llu: the table holds %d", (unsigned long long)index, LAV5_MAX_BUFFERS);
		return;
	}
	[buffer retain];
	[ce->buf[index] release];
	ce->buf[index] = buffer;
	ce->off[index] = offset;
	free(ce->bytes[index]);
	ce->bytes[index] = NULL;
	ce->blen[index] = 0;
}

static void
cencSetBufferOffset(id self, __unused SEL _cmd, uint64_t offset, uint64_t index)
{
	if (index < LAV5_MAX_BUFFERS) {
		STATE(self, struct cenc)->off[index] = offset;
	}
}

static void
cencSetBuffers(id self, SEL cmd, const id * buffers, const uint64_t * offsets, NSRange range)
{
	for (NSUInteger i = 0; i < range.length; i++) {
		cencSetBuffer(self, cmd, buffers[i], offsets[i], range.location + i);
	}
}

static void
cencSetBytes(id self, __unused SEL _cmd, const void * bytes, uint64_t length, uint64_t index)
{
	struct cenc * ce = STATE(self, struct cenc);

	if (index >= LAV5_MAX_BUFFERS || length == 0 || length > 4096) {
		lavlog("setBytes of %llu at index %llu refused", (unsigned long long)length, (unsigned long long)index);
		return;
	}
	[ce->buf[index] release];
	ce->buf[index] = nil;
	free(ce->bytes[index]);
	ce->bytes[index] = malloc((size_t)length);
	memcpy(ce->bytes[index], bytes, (size_t)length);
	ce->blen[index] = (uint32_t)length;
}

static void
cencAddDispatch(id self, uint32_t * groups, uint32_t * local)
{
	struct cenc * ce = STATE(self, struct cenc);
	struct cbstate * st = STATE(ce->commandBuffer, struct cbstate);
	struct op * op;
	struct dispatch * d;

	if (ce->pipeline == nil) {
		lavlog("a dispatch with no pipeline state set");
		return;
	}
	op = cbAddOp(st);
	d = calloc(1, sizeof(*d));
	op->kind = OP_DISPATCH;
	op->disp = d;
	d->pipeline = [ce->pipeline retain];
	for (int i = 0; i < LAV5_MAX_BUFFERS; i++) {
		d->buf[i] = [ce->buf[i] retain];
		d->off[i] = ce->off[i];
		if (ce->bytes[i] != NULL) {
			d->bytes[i] = malloc(ce->blen[i]);
			memcpy(d->bytes[i], ce->bytes[i], ce->blen[i]);
			d->blen[i] = ce->blen[i];
		}
	}
	memcpy(d->groups, groups, sizeof(d->groups));
	memcpy(d->local, local, sizeof(d->local));
}

static void
cencDispatchGroups(id self, __unused SEL _cmd, LavSize groups, LavSize local)
{
	uint32_t g[3] = { (uint32_t)groups.width, (uint32_t)groups.height, (uint32_t)groups.depth };
	uint32_t l[3] = { (uint32_t)local.width, (uint32_t)local.height, (uint32_t)local.depth };

	cencAddDispatch(self, g, l);
}

/* Metal runs exactly GRID threads; this GPU runs whole groups, so the grid is rounded up to them */
static void
cencDispatchThreads(id self, __unused SEL _cmd, LavSize grid, LavSize local)
{
	uint32_t l[3] = { (uint32_t)local.width, (uint32_t)local.height, (uint32_t)local.depth };
	uint32_t g[3];

	if (l[0] == 0 || l[1] == 0 || l[2] == 0) {
		lavlog("dispatchThreads with a threadgroup of zero");
		return;
	}
	g[0] = (uint32_t)((grid.width + l[0] - 1) / l[0]);
	g[1] = (uint32_t)((grid.height + l[1] - 1) / l[1]);
	g[2] = (uint32_t)((grid.depth + l[2] - 1) / l[2]);
	if (g[0] * l[0] != grid.width || g[1] * l[1] != grid.height || g[2] * l[2] != grid.depth) {
		lavlog("dispatchThreads: a grid of %llux%llux%llu is not whole threadgroups of %ux%ux%u; the last group runs whole",
		    (unsigned long long)grid.width, (unsigned long long)grid.height, (unsigned long long)grid.depth, l[0], l[1], l[2]);
	}
	cencAddDispatch(self, g, l);
}

static void
cencEnd(id self, __unused SEL _cmd)
{
	STATE(self, struct cenc)->ended = YES;
}

static void
cencNothing(__unused id self, __unused SEL _cmd, __unused uint64_t a)
{
}

static void
cencSharedLength(__unused id self, __unused SEL _cmd, uint64_t length, __unused uint64_t index)
{
	if (length != 0) {
		lavlog("setThreadgroupMemoryLength:%llu: threadgroup memory is not served yet", (unsigned long long)length);
	}
}

static void
cencUseResource(__unused id self, __unused SEL _cmd, __unused id resource, __unused uint64_t usage)
{
}

static id
cencLabel(id self, __unused SEL _cmd)
{
	return STATE(self, struct cenc)->label;
}

static void
cencSetLabel(id self, __unused SEL _cmd, id label)
{
	struct cenc * ce = STATE(self, struct cenc);

	[ce->label release];
	ce->label = [label copy];
}

/* run one dispatch: bind, build the stream, submit it, wait for the flush */
static BOOL
runDispatch(struct op * op)
{
	struct dispatch * d = op->disp;
	struct pipestate * ps = STATE(d->pipeline, struct pipestate);
	struct lav5_dispatch ld;
	uint32_t * w;
	const char * why = NULL;
	size_t n, boff = 0;
	BOOL ok;

	memset(&ld, 0, sizeof(ld));
	ld.k = &ps->k;
	ld.code_gpu = ps->code.gpu;
	memcpy(ld.groups, d->groups, sizeof(ld.groups));
	memcpy(ld.local, d->local, sizeof(ld.local));
	ld.flush_gpu = gStream.gpu + SCRATCH_OFFSET + 64;
	ld.hole_gpu = gStream.gpu + SCRATCH_OFFSET + 128;
	if (gBytes.handle == 0 && !lavBufNew(65536, &gBytes)) {
		return NO;
	}
	for (int i = 0; i < LAV5_MAX_BUFFERS; i++) {
		if (d->buf[i] != nil) {
			struct bufstate * b = STATE(d->buf[i], struct bufstate);

			if (d->off[i] > b->b.len) {
				lavlog("dispatch of %s: the offset of buffer %d is past its end", ps->k.name, i);
				return NO;
			}
			ld.buf[i].gpu = b->b.gpu + d->off[i];
			ld.buf[i].cpu = (uint8_t *)b->b.cpu + d->off[i];
			ld.buf[i].bytes = b->b.len - d->off[i];
		} else if (d->bytes[i] != NULL) {
			uint32_t len = (d->blen[i] + 15) & ~15u;

			if (boff + len > gBytes.len) {
				lavlog("dispatch of %s: too many setBytes", ps->k.name);
				return NO;
			}
			memset((uint8_t *)gBytes.cpu + boff, 0, len);
			memcpy((uint8_t *)gBytes.cpu + boff, d->bytes[i], d->blen[i]);
			ld.buf[i].gpu = gBytes.gpu + boff;
			ld.buf[i].cpu = (uint8_t *)gBytes.cpu + boff;
			ld.buf[i].bytes = len;
			boff += len;
		} else if (ps->k.buffer_mask >> i & 1) {
			lavlog("dispatch of %s: nothing bound at buffer index %d", ps->k.name, i);
			return NO;
		}
	}
	w = malloc(STREAM_WORDS * 4);
	n = lav5_build(w, STREAM_WORDS, &ld, &why);
	if (n == 0) {
		lavlog("dispatch of %s refused: %s", ps->k.name, why != NULL ? why : "?");
		free(w);
		return NO;
	}
	ok = lavSubmit(w, n);
	lavlog("dispatch of %s: %ux%ux%u groups of %ux%ux%u, %zu words, %s", ps->k.name, ld.groups[0], ld.groups[1], ld.groups[2],
	    ld.local[0], ld.local[1], ld.local[2], n, ok ? "done" : "FAILED");
	free(w);
	return ok;
}

/* hang hunt: a commit that has run for more than 12 s gets a backtrace of its thread written to the log (SIGUSR2 to it) */
static volatile pthread_t gCommitThread;
static volatile int gCommitBusy, gCommitStage, gCommitDumped;
static volatile time_t gCommitT0;

static void
lavSigDump(__unused int sig)
{
	void * bt[40];
	int n = backtrace(bt, 40), fd = open(LAV_LOG, O_WRONLY | O_APPEND);
	char buf[200];

	if (fd < 0) {
		return;
	}
	for (int i = 0; i < n; i++) {
		Dl_info di;
		int len;

		if (dladdr(bt[i], &di) && di.dli_fname != NULL) {
			len = snprintf(buf, sizeof(buf), "stuck commit bt %d: %s %s+%ld  %p image+%#lx\n", i, di.dli_fname, di.dli_sname != NULL ? di.dli_sname : "?",
			    di.dli_saddr != NULL ? (long)((char *)bt[i] - (char *)di.dli_saddr) : 0L, bt[i], (unsigned long)((char *)bt[i] - (char *)di.dli_fbase));
		} else {
			len = snprintf(buf, sizeof(buf), "stuck commit bt %d: %p\n", i, bt[i]);
		}
		write(fd, buf, (size_t)len);
	}
	close(fd);
}

static void *
lavWatchdog(__unused void * arg)
{
	for (;;) {
		sleep(5);
		if (gCommitBusy && !gCommitDumped && time(NULL) - gCommitT0 > 12) {
			gCommitDumped = 1;
			lavlog("commit stuck for %ld s at stage %d: dumping its thread", (long)(time(NULL) - gCommitT0), gCommitStage);
			pthread_kill(gCommitThread, SIGUSR2);
		}
	}
	return NULL;
}

__attribute__((constructor)) static void
lavWatchdogStart(void)
{
	pthread_t t;

	signal(SIGUSR2, lavSigDump);
	if (pthread_create(&t, NULL, lavWatchdog, NULL) == 0) {
		pthread_detach(t);
	}
}

typedef void (^LavHandler)(id);

/* where the time of draws goes (ns, reset at each commit, printed with 'commit time'); filled by runDraw */
static uint64_t gTSpec, gTCopy, gTBuild, gTSubmit, gNSub, gPixels;

static void cbSyncSurfaces(struct cbstate * st);
static BOOL runClear(struct op * op);
static BOOL runDraw(struct op * op);
static BOOL runTexCopy(struct op * op);

static void
cbCommit(id self, __unused SEL _cmd)
{
	struct cbstate * st = STATE(self, struct cbstate);
	BOOL ok = YES;
	uint64_t tKind[8] = { 0 }, tStart = 0, tPrev = 0, tMax = 0, tSync = 0;
	unsigned tLast = 0, tMaxKind = 0;

	if (st->status >= 2) {
		return;
	}
	st->status = 2;                         /* Committed */
	gCommitThread = pthread_self();
	gCommitT0 = time(NULL);
	gCommitStage = 0;
	gCommitDumped = 0;
	gCommitBusy = 1;
	gCommitGen++;
	gNoBatch = access("/var/tmp/lavnobatch", F_OK) == 0;
	gTSpec = gTCopy = gTBuild = gTSubmit = gNSub = gPixels = 0;
	BOOL prof = access("/var/tmp/lavprof", F_OK) == 0;
	if (prof) {
		lavlog("commit: %zu ops, %lu drawables, %lu scheduled, %lu completed handlers", st->nops, (unsigned long)[st->drawables count],
		    (unsigned long)[st->scheduled count], (unsigned long)[st->handlers count]);
	}
	for (id h in st->scheduled) {
		((LavHandler)h)(self);
	}
	st->status = 3;                         /* Scheduled */
	pthread_mutex_lock(&gGPULock);
	tStart = clock_gettime_nsec_np(CLOCK_MONOTONIC);
	for (size_t i = 0; i < st->nops && ok; i++) {
		uint64_t to = clock_gettime_nsec_np(CLOCK_MONOTONIC);

		if (i != 0) {
			tKind[tLast < 8 ? tLast : 7] += to - tPrev;
			if (to - tPrev > tMax) {
				tMax = to - tPrev;
				tMaxKind = tLast;
			}
		}
		tPrev = to;
		tLast = (unsigned)st->ops[i].kind;
		gCommitStage = (int)(i * 100) + (int)st->ops[i].kind;
		if (st->ops[i].kind != OP_DRAW && !lavFlush()) {
			ok = NO;
			break;
		}
		switch (st->ops[i].kind) {
		case OP_FILL:
			ok = runFill(&st->ops[i]);
			break;
		case OP_COPY:
			ok = runCopy(&st->ops[i]);
			break;
		case OP_CLEAR:
			ok = runClear(&st->ops[i]);
			break;
		case OP_DRAW:
			ok = runDraw(&st->ops[i]);
			break;
		case OP_TEXCOPY:
			ok = runTexCopy(&st->ops[i]);
			break;
		default:
			ok = runDispatch(&st->ops[i]);
			break;
		}
	}
	if (st->nops != 0) {
		uint64_t to = clock_gettime_nsec_np(CLOCK_MONOTONIC);

		tKind[tLast < 8 ? tLast : 7] += to - tPrev;
		if (to - tPrev > tMax) {
			tMax = to - tPrev;
			tMaxKind = tLast;
		}
		tPrev = to;
	}
	gCommitStage = -1;                      /* syncing surfaces */
	if (!lavFlush()) {
		ok = NO;
	}
	if (ok) {
		cbSyncSurfaces(st);
	}
	tSync = clock_gettime_nsec_np(CLOCK_MONOTONIC) - tPrev;
	gCommitStage = -2;
	pthread_mutex_unlock(&gGPULock);
	cbFreeOps(st);
	if (ok) {
		for (id d in st->drawables) {
			((void (*)(id, SEL))objc_msgSend)(d, sel_registerName("present"));
		}
	}
	[st->drawables removeAllObjects];
	st->status = ok ? 4 : 5;                /* Completed or Error */
	gCommitBusy = 0;
	if (!ok || prof) {
		lavlog("commit: %s, running %lu completed handlers", ok ? "completed" : "ERROR", (unsigned long)[st->handlers count]);
		lavlog("draw phases (ms): specialise %llu, texture copy-in %llu, build stream %llu, submit+wait %llu in %llu submits, %llu kpixels",
		    (unsigned long long)(gTSpec / 1000000u), (unsigned long long)(gTCopy / 1000000u), (unsigned long long)(gTBuild / 1000000u),
		    (unsigned long long)(gTSubmit / 1000000u), (unsigned long long)gNSub, (unsigned long long)(gPixels / 1000u));
		lavlog("commit time: total %llu ms; by op kind (ms) 0:%llu 1:%llu 2:%llu 3:%llu 4:%llu 5:%llu 6:%llu 7+:%llu; sync %llu ms; slowest op kind %u %llu ms",
		    (unsigned long long)((clock_gettime_nsec_np(CLOCK_MONOTONIC) - tStart) / 1000000u), (unsigned long long)(tKind[0] / 1000000u),
		    (unsigned long long)(tKind[1] / 1000000u), (unsigned long long)(tKind[2] / 1000000u), (unsigned long long)(tKind[3] / 1000000u),
		    (unsigned long long)(tKind[4] / 1000000u), (unsigned long long)(tKind[5] / 1000000u), (unsigned long long)(tKind[6] / 1000000u),
		    (unsigned long long)(tKind[7] / 1000000u), (unsigned long long)(tSync / 1000000u), tMaxKind, (unsigned long long)(tMax / 1000000u));
	}
	for (id h in st->handlers) {
		((LavHandler)h)(self);
	}
	if (prof) {
		lavlog("commit: handlers returned");
	}
}

static void
cbEnqueue(__unused id self, __unused SEL _cmd)
{
}

static void
cbWait(__unused id self, __unused SEL _cmd)
{
	/* commit runs everything before it returns */
}

static void
cbAddHandler(id self, __unused SEL _cmd, id block)
{
	[STATE(self, struct cbstate)->handlers addObject:[[block copy] autorelease]];
}

static void
cbAddScheduled(id self, __unused SEL _cmd, id block)
{
	[STATE(self, struct cbstate)->scheduled addObject:[[block copy] autorelease]];
}

static uint64_t
cbStatus(id self, __unused SEL _cmd)
{
	return (uint64_t)STATE(self, struct cbstate)->status;
}

static id
cbError(id self, __unused SEL _cmd)
{
	if (STATE(self, struct cbstate)->status != 5) {
		return nil;
	}
	return [NSError errorWithDomain:@"MTLCommandBufferErrorDomain" code:1
	    userInfo:@{ NSLocalizedDescriptionKey: @"LavenderMetal: the GPU did not complete the work" }];
}

static id
cbQueue(id self, __unused SEL _cmd)
{
	return STATE(self, struct cbstate)->queue;
}

static id
cbLabel(id self, __unused SEL _cmd)
{
	return STATE(self, struct cbstate)->label;
}

static void
cbSetLabel(id self, __unused SEL _cmd, id label)
{
	struct cbstate * st = STATE(self, struct cbstate);

	[st->label release];
	st->label = [label copy];
}

static double
cbNoTime(__unused id self, __unused SEL _cmd)
{
	return 0;
}

#include "LavenderRender.inc"

/* ---- the device ---- */

static id
deviceInit(id self, __unused SEL _cmd, unsigned int port)
{
	struct objc_super sup = { self, gDeviceBase };
	id (*send)(struct objc_super *, SEL) = (id (*)(struct objc_super *, SEL))objc_msgSendSuper;

	self = send(&sup, sel_registerName("init"));
	gPort = port;
	gDevice = self;
	lavlog("a device for the accelerator port %u (%s)", port, self != nil ? "made" : "_MTLDevice's init gave nil");
	return self;
}

static id
deviceName(__unused id self, __unused SEL _cmd)
{
	return @"Lavender Adreno 512";
}

static uint64_t
deviceRegistryID(__unused id self, __unused SEL _cmd)
{
	return gPort;
}

static uint64_t
deviceMaxBufferLength(__unused id self, __unused SEL _cmd)
{
	return 16ull << 20;
}

static BOOL
deviceYes(__unused id self, __unused SEL _cmd)
{
	return YES;
}

static BOOL
deviceNo(__unused id self, __unused SEL _cmd)
{
	return NO;
}

static BOOL
deviceNoFeatureSet(__unused id self, __unused SEL _cmd, __unused uint64_t set)
{
	return NO;
}

static uint64_t
deviceZero(__unused id self, __unused SEL _cmd)
{
	return 0;
}

static Class
newClass(const char * name, const char * base)
{
	Class b = objc_getClass(base);
	Class cls;

	if (b == Nil) {
		lavlog("this system has no %s: %s is made on NSObject", base, name);
		b = objc_getClass("NSObject");
	}
	cls = objc_allocateClassPair(b, name, 0);
	if (cls == Nil) {
		lavlog("could not make the class %s", name);
	}
	return cls;
}

#define ADD(cls, sel, imp, types) class_addMethod(cls, sel_registerName(sel), (IMP)imp, types)

static void
registerComputeClasses(Class device)
{
	static const char * const lib[] = { "MTLLibrary", NULL };
	static const char * const fn[] = { "MTLFunction", NULL };
	static const char * const pipe[] = { "MTLComputePipelineState", NULL };
	static const char * const comp[] = { "MTLComputeCommandEncoder", "MTLCommandEncoder", NULL };

	if ((gLibraryClass = newClass("LavenderMetalLibrary", "_MTLLibrary")) != Nil) {
		declareNamed(gLibraryClass, lib);
		addStubForwarding(gLibraryClass);
		ADD(gLibraryClass, "dealloc", libDealloc, "v@:");
		ADD(gLibraryClass, "device", objDevice, "@@:");
		ADD(gLibraryClass, "label", libLabel, "@@:");
		ADD(gLibraryClass, "setLabel:", libSetLabel, "v@:@");
		ADD(gLibraryClass, "functionNames", libFunctionNames, "@@:");
		ADD(gLibraryClass, "newFunctionWithName:", libNewFunction, "@@:@");
		ADD(gLibraryClass, "newFunctionWithName:constantValues:error:", libNewFunctionConstants, "@@:@@^@");
		ADD(gLibraryClass, "newFunctionWithDescriptor:error:", libNewFunctionDescriptor, "@@:@^@");
		ADD(gFunctionClass, "newSpecializedFunctionWithDescriptor:destinationArchive:functionCache:error:", fnNewSpecialized, "@@:@@@^@");
		objc_registerClassPair(gLibraryClass);
		ADD(device, "newLibraryWithData:error:", deviceNewLibraryData, "@@:@^@");
		ADD(device, "newLibraryWithURL:error:", deviceNewLibraryURL, "@@:@^@");
		ADD(device, "newLibraryWithFile:error:", deviceNewLibraryFile, "@@:@^@");
		ADD(device, "newPipelineLibraryWithFilePath:error:", deviceNewPipelineLibrary, "@@:@^@");
		ADD(device, "newLibraryWithSource:options:error:", deviceNewLibrarySource, "@@:@@^@");
		ADD(device, "newDefaultLibrary", deviceNoLibrary, "@@:");
		ADD(device, "newDefaultLibraryWithBundle:error:", deviceNoLibraryBundle, "@@:@^@");
	}
	if ((gFunctionClass = newClass("LavenderMetalFunction", "_MTLFunction")) != Nil) {
		declareNamed(gFunctionClass, fn);
		addStubForwarding(gFunctionClass);
		ADD(gFunctionClass, "dealloc", fnDealloc, "v@:");
		ADD(gFunctionClass, "device", objDevice, "@@:");
		ADD(gFunctionClass, "name", fnName, "@@:");
		ADD(gFunctionClass, "functionType", fnType, "Q@:");
		ADD(gFunctionClass, "label", fnLabel, "@@:");
		ADD(gFunctionClass, "setLabel:", fnSetLabel, "v@:@");
		objc_registerClassPair(gFunctionClass);
	}
	if ((gPipelineClass = newClass("LavenderMetalComputePipeline", "_MTLComputePipelineState")) != Nil) {
		declareNamed(gPipelineClass, pipe);
		addStubForwarding(gPipelineClass);
		ADD(gPipelineClass, "dealloc", pipeDealloc, "v@:");
		ADD(gPipelineClass, "device", objDevice, "@@:");
		ADD(gPipelineClass, "label", pipeLabel, "@@:");
		ADD(gPipelineClass, "setLabel:", pipeSetLabel, "v@:@");
		ADD(gPipelineClass, "maxTotalThreadsPerThreadgroup", pipeMaxThreads, "Q@:");
		ADD(gPipelineClass, "threadExecutionWidth", pipeWidth, "Q@:");
		ADD(gPipelineClass, "staticThreadgroupMemoryLength", pipeStaticShared, "Q@:");
		objc_registerClassPair(gPipelineClass);
		ADD(device, "newComputePipelineStateWithFunction:error:", deviceNewPipeline, "@@:@^@");
		ADD(device, "newComputePipelineStateWithFunction:options:reflection:error:", deviceNewPipelineOptions, "@@:@Q^@^@");
		ADD(device, "newComputePipelineStateWithDescriptor:options:reflection:error:", deviceNewPipelineDescriptor, "@@:@Q^@^@");
		ADD(device, "newComputePipelineStateWithFunction:completionHandler:", deviceNewPipelineAsync, "v@:@@?");
	}
	if ((gComputeClass = newClass("LavenderMetalComputeEncoder", "_MTLCommandEncoder")) != Nil) {
		declareNamed(gComputeClass, comp);
		addStubForwarding(gComputeClass);
		ADD(gComputeClass, "dealloc", cencDealloc, "v@:");
		ADD(gComputeClass, "device", objDevice, "@@:");
		ADD(gComputeClass, "label", cencLabel, "@@:");
		ADD(gComputeClass, "setLabel:", cencSetLabel, "v@:@");
		ADD(gComputeClass, "setComputePipelineState:", cencSetPipeline, "v@:@");
		ADD(gComputeClass, "setBuffer:offset:atIndex:", cencSetBuffer, "v@:@QQ");
		ADD(gComputeClass, "setBufferOffset:atIndex:", cencSetBufferOffset, "v@:QQ");
		ADD(gComputeClass, "setBuffers:offsets:withRange:", cencSetBuffers, "v@:r^@r^Q{_NSRange=QQ}");
		ADD(gComputeClass, "setBytes:length:atIndex:", cencSetBytes, "v@:r^vQQ");
		ADD(gComputeClass, "dispatchThreadgroups:threadsPerThreadgroup:", cencDispatchGroups,
		    "v@:{_MTLSize=QQQ}{_MTLSize=QQQ}");
		ADD(gComputeClass, "dispatchThreads:threadsPerThreadgroup:", cencDispatchThreads, "v@:{_MTLSize=QQQ}{_MTLSize=QQQ}");
		ADD(gComputeClass, "setThreadgroupMemoryLength:atIndex:", cencSharedLength, "v@:QQ");
		ADD(gComputeClass, "useResource:usage:", cencUseResource, "v@:@Q");
		ADD(gComputeClass, "memoryBarrierWithScope:", cencNothing, "v@:Q");
		ADD(gComputeClass, "endEncoding", cencEnd, "v@:");
		objc_registerClassPair(gComputeClass);
	}
}

#include "LavenderRenderReg.inc"

static void
registerObjectClasses(Class device)
{
	static const char * const buffer[] = { "MTLBuffer", "MTLResource", NULL };
	static const char * const cb[] = { "MTLCommandBuffer", NULL };
	static const char * const blit[] = { "MTLBlitCommandEncoder", "MTLCommandEncoder", NULL };

	if ((gBufferClass = newClass("LavenderMetalBuffer", "_MTLResource")) != Nil) {
		declareNamed(gBufferClass, buffer);
		addStubForwarding(gBufferClass);
		ADD(gBufferClass, "dealloc", bufferDealloc, "v@:");
		ADD(gBufferClass, "device", objDevice, "@@:");
		ADD(gBufferClass, "label", bufferLabel, "@@:");
		ADD(gBufferClass, "setLabel:", bufferSetLabel, "v@:@");
		ADD(gBufferClass, "contents", bufferContents, "^v@:");
		ADD(gBufferClass, "length", bufferLength, "Q@:");
		ADD(gBufferClass, "resourceOptions", bufferOptions, "Q@:");
		ADD(gBufferClass, "storageMode", bufferStorageMode, "Q@:");
		ADD(gBufferClass, "cpuCacheMode", bufferCacheMode, "Q@:");
		ADD(gBufferClass, "hazardTrackingMode", bufferHazardMode, "Q@:");
		ADD(gBufferClass, "allocatedSize", bufferLength, "Q@:");
		ADD(gBufferClass, "gpuAddress", bufferGPUAddress, "Q@:");
		ADD(gBufferClass, "didModifyRange:", bufferDidModify, "v@:{_NSRange=QQ}");
		ADD(gBufferClass, "setPurgeableState:", bufferPurgeable, "Q@:Q");
		objc_registerClassPair(gBufferClass);
		ADD(device, "newBufferWithLength:options:", deviceNewBuffer, "@@:QQ");
		ADD(device, "newBufferWithBytes:length:options:", deviceNewBufferBytes, "@@:r^vQQ");
	}
	if ((gQueueClass = newClass("LavenderMetalQueue", "_MTLCommandQueue")) != Nil) {
		declareFamily(gQueueClass, "MTLCommandQueue");
		addStubForwarding(gQueueClass);
		ADD(gQueueClass, "device", objDevice, "@@:");
		ADD(gQueueClass, "label", queueLabel, "@@:");
		ADD(gQueueClass, "setCompletionQueue:", queueSetCompletionQueue, "v@:@");
		ADD(gQueueClass, "completionQueue", queueCompletionQueue, "@@:");
		ADD(gQueueClass, "dealloc", queueDealloc, "v@:");
		ADD(gQueueClass, "commandBuffer", queueCommandBuffer, "@@:");
		ADD(gQueueClass, "commandBufferWithUnretainedReferences", queueCommandBuffer, "@@:");
		objc_registerClassPair(gQueueClass);
		ADD(device, "newCommandQueue", deviceNewQueue, "@@:");
		ADD(device, "newCommandQueueWithMaxCommandBufferCount:", deviceNewQueueMax, "@@:Q");
	}
	if ((gCommandBufferClass = newClass("LavenderMetalCommandBuffer", "_MTLCommandBuffer")) != Nil) {
		declareNamed(gCommandBufferClass, cb);
		addStubForwarding(gCommandBufferClass);
		ADD(gCommandBufferClass, "dealloc", cbDealloc, "v@:");
		ADD(gCommandBufferClass, "device", objDevice, "@@:");
		ADD(gCommandBufferClass, "label", cbLabel, "@@:");
		ADD(gCommandBufferClass, "setLabel:", cbSetLabel, "v@:@");
		ADD(gCommandBufferClass, "commandQueue", cbQueue, "@@:");
		ADD(gCommandBufferClass, "commit", cbCommit, "v@:");
		ADD(gCommandBufferClass, "enqueue", cbEnqueue, "v@:");
		ADD(gCommandBufferClass, "waitUntilCompleted", cbWait, "v@:");
		ADD(gCommandBufferClass, "waitUntilScheduled", cbWait, "v@:");
		ADD(gCommandBufferClass, "addCompletedHandler:", cbAddHandler, "v@:@?");
		ADD(gCommandBufferClass, "addScheduledHandler:", cbAddScheduled, "v@:@?");
		ADD(gCommandBufferClass, "status", cbStatus, "Q@:");
		ADD(gCommandBufferClass, "error", cbError, "@@:");
		ADD(gCommandBufferClass, "blitCommandEncoder", cbBlitEncoder, "@@:");
		ADD(gCommandBufferClass, "computeCommandEncoder", cbComputeEncoder, "@@:");
		ADD(gCommandBufferClass, "computeCommandEncoderWithDispatchType:", cbComputeEncoderDispatchType, "@@:Q");
		ADD(gCommandBufferClass, "computeCommandEncoderWithDescriptor:", cbComputeEncoderDescriptor, "@@:@");
		ADD(gCommandBufferClass, "GPUStartTime", cbNoTime, "d@:");
		ADD(gCommandBufferClass, "GPUEndTime", cbNoTime, "d@:");
		ADD(gCommandBufferClass, "kernelStartTime", cbNoTime, "d@:");
		ADD(gCommandBufferClass, "kernelEndTime", cbNoTime, "d@:");
		objc_registerClassPair(gCommandBufferClass);
	}
	if ((gBlitClass = newClass("LavenderMetalBlitEncoder", "_MTLCommandEncoder")) != Nil) {
		declareNamed(gBlitClass, blit);
		addStubForwarding(gBlitClass);
		ADD(gBlitClass, "dealloc", encDealloc, "v@:");
		ADD(gBlitClass, "device", objDevice, "@@:");
		ADD(gBlitClass, "label", encLabel, "@@:");
		ADD(gBlitClass, "setLabel:", encSetLabel, "v@:@");
		ADD(gBlitClass, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", encCopy, "v@:@Q@QQ");
		ADD(gBlitClass, "fillBuffer:range:value:", encFill, "v@:@{_NSRange=QQ}C");
		ADD(gBlitClass, "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
		    encCopyTexture, "v@:@QQ{_MTLOrigin=QQQ}{_MTLSize=QQQ}@QQ{_MTLOrigin=QQQ}");
		ADD(gBlitClass, "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
		    encCopyTextureOpt, "v@:@QQ{_MTLOrigin=QQQ}{_MTLSize=QQQ}@QQ{_MTLOrigin=QQQ}Q");
		ADD(gBlitClass, "generateMipmapsForTexture:", encGenMips, "v@:@");
		ADD(gBlitClass, "endEncoding", encEnd, "v@:");
		ADD(gBlitClass, "synchronizeResource:", encNothing, "v@:@");
		objc_registerClassPair(gBlitClass);
	}
	registerComputeClasses(device);
	installConstantHooks();
	registerRenderClasses(device);
}

__attribute__((constructor)) static void
lavenderMetalLoad(void)
{
	Class cls;

	if (strcmp(getprogname(), "SpringBoard") == 0) {
		/* one GPU client only: backboardd. SpringBoard drew its lock screen glyphs differently when it was given a Metal device */
		lavlog("SpringBoard gets no Metal device");
		return;
	}
	gDeviceBase = objc_getClass("_MTLDevice");
	if (gDeviceBase == Nil) {
		lavlog("the system has no _MTLDevice; this plug-in registers nothing");
		return;
	}
	cls = objc_allocateClassPair(gDeviceBase, "LavenderMetalDevice", 0);
	if (cls == Nil) {
		lavlog("LavenderMetalDevice could not be made (already there?)");
		return;
	}
	declareFamily(cls, "MTLDevice");
	addStubForwarding(cls);
	ADD(cls, "initWithAcceleratorPort:", deviceInit, "@@:I");
	ADD(cls, "name", deviceName, "@@:");
	ADD(cls, "registryID", deviceRegistryID, "Q@:");
	ADD(cls, "maxBufferLength", deviceMaxBufferLength, "Q@:");
	ADD(cls, "hasUnifiedMemory", deviceYes, "B@:");
	ADD(cls, "isLowPower", deviceNo, "B@:");
	ADD(cls, "isRemovable", deviceNo, "B@:");
	ADD(cls, "isHeadless", deviceNo, "B@:");
	ADD(cls, "supportsFeatureSet:", deviceNoFeatureSet, "B@:Q");
	ADD(cls, "dedicatedMemorySize", deviceZero, "Q@:");
	registerObjectClasses(cls);
	objc_registerClassPair(cls);
	lavlog("loaded: LavenderMetalDevice and its objects are registered (base %s)", class_getName(gDeviceBase));
}
