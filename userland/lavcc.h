/* lavcc: a .metallib's kernel functions compiled for the Adreno 512, behind a C interface (lavcc.cpp) */
#ifndef LAVCC_H
#define LAVCC_H

#include <stddef.h>
#include <stdint.h>
#include "lava5.h"

#ifdef __cplusplus
extern "C" {
#endif

struct lavcc_lib;

/* the library from its bytes (copied); NULL with the reason in WHY */
struct lavcc_lib * lavcc_open(const void * data, size_t size, char * why, size_t whysize);
void lavcc_close(struct lavcc_lib * l);
unsigned lavcc_function_count(struct lavcc_lib * l);
const char * lavcc_function_name(struct lavcc_lib * l, unsigned i, int * type);
/* 1 with OUT filled (free it with lavcc_kernel_free), 0 with the reason in WHY */
int lavcc_compile(struct lavcc_lib * l, const char * name, struct lav5_kernel * out, char * why, size_t whysize);
void lavcc_kernel_free(struct lav5_kernel * k);
/* a vertex and a fragment function compiled and linked (render pipeline); 1 with VS, FS and LINK filled (free each stage with
 * lavcc_stage_free), 0 with the reason in WHY */
int lavcc_compile_program(struct lavcc_lib * lv, const char * vname, struct lavcc_lib * lf, const char * fname, struct lav5_stage * vs,
    struct lav5_stage * fs, struct lav5_link * link, char * why, size_t whysize);
void lavcc_stage_free(struct lav5_stage * s);
/* a function constant as the pipeline supplies it: INDEX is [[function_constant(n)]], BITS the value as the bits of its type */
struct lavcc_const {
	uint32_t index;
	uint64_t bits;
};
/* as lavcc_compile_program, with the function constants of each stage (NULL, 0: none) */
int lavcc_compile_program_c(struct lavcc_lib * lv, const char * vname, const struct lavcc_const * vc, unsigned nvc, struct lavcc_lib * lf,
    const char * fname, const struct lavcc_const * fc, unsigned nfc, struct lav5_stage * vs, struct lav5_stage * fs, struct lav5_link * link,
    char * why, size_t whysize);
/* which bytes of the fragment function's constant buffer at [[buffer(BINDING)]] (LENGTH of them) its control flow depends on: MASK gets
 * LENGTH bytes, 1 where so. 1 on success, 0 with the reason in WHY. */
int lavcc_control_bytes(struct lavcc_lib * lf, const char * fname, const struct lavcc_const * fc, unsigned nfc, unsigned binding, size_t length,
    uint8_t * mask, char * why, size_t whysize);
/* as lavcc_compile_program_c, the fragment stage compiled for the buffer at BINDING holding BYTES (LENGTH of them), of which FOLD (1 per
 * byte) are known: the branches they decide are decided at compile time (CoreAnimation's uber shader: 15 000 instructions down to ~50) */
int lavcc_compile_program_known(struct lavcc_lib * lv, const char * vname, const struct lavcc_const * vc, unsigned nvc, struct lavcc_lib * lf,
    const char * fname, const struct lavcc_const * fc, unsigned nfc, unsigned binding, const uint8_t * bytes, const uint8_t * fold, size_t length,
    struct lav5_stage * vs, struct lav5_stage * fs, struct lav5_link * link, char * why, size_t whysize);
/* the function constants function FNAME declares: name and index of each, up to MAX; returns how many (0 when none or not found) */
struct lavcc_fcname {
	char name[64];
	uint32_t index;
};
unsigned lavcc_function_constants(struct lavcc_lib * l, const char * fname, struct lavcc_fcname * out, unsigned max);
/* function I of the library compiled alone: 1 with a SUMMARY of what it uses, 0 with the reason in WHY, -1 past the end */
int lavcc_probe(struct lavcc_lib * l, unsigned i, char * summary, size_t ssize, char * why, size_t whysize);

void lavcc_survey_pairs(struct lavcc_lib * l);

#ifdef __cplusplus
}
#endif

#endif
