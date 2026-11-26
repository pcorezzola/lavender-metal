// lavcc: the compiler of hakios-metal (MIT) behind a C interface, for the plug-in: a .metallib's bytes in, a function by
// name, a lav5_kernel out (machine code and the fields the A5xx dispatch builds its stream from).
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdio>
#include "hakios-metal/compiler.hpp"
extern "C" {
#include "air_module.h"
#include "util/ralloc.h"
}
#include "hakios-metal/metallib.hpp"
#include "lava5.h"
#include "lavcc.h"

using namespace hakios::metal;

struct lavcc_lib {
    std::vector<uint8_t> bytes;
    MetalLibrary lib;
};

extern "C" struct lavcc_lib* lavcc_open(const void* data, size_t size, char* why, size_t whysize)
{
    auto* l = new lavcc_lib();
    l->bytes.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    if (!l->lib.parse(l->bytes.data(), l->bytes.size())) {
        snprintf(why, whysize, "%s", l->lib.why().c_str());
        delete l;
        return nullptr;
    }
    return l;
}

extern "C" void lavcc_close(struct lavcc_lib* l) { delete l; }

extern "C" unsigned lavcc_function_count(struct lavcc_lib* l) { return unsigned(l->lib.functions().size()); }

extern "C" const char* lavcc_function_name(struct lavcc_lib* l, unsigned i, int* type)
{
    if (i >= l->lib.functions().size()) return nullptr;
    if (type) *type = l->lib.functions()[i].type;
    return l->lib.functions()[i].name.c_str();
}

extern "C" int lavcc_compile(struct lavcc_lib* l, const char* name, struct lav5_kernel* out, char* why, size_t whysize)
{
    const MetalFunction* fn = l->lib.find(name);
    if (!fn) {
        snprintf(why, whysize, "no function %s", name);
        return 0;
    }
    if (fn->type != MetalFunction::kKernel) {
        snprintf(why, whysize, "%s is not a kernel function", name);
        return 0;
    }
    const uint8_t* bc = nullptr;
    uint64_t bcSize = 0;
    if (!l->lib.bitcode(*fn, &bc, &bcSize)) {
        snprintf(why, whysize, "no bitcode for %s", name);
        return 0;
    }
    std::string w;
    auto k = compileAir(bc, bcSize, &w);
    if (!k) {
        snprintf(why, whysize, "%s", w.c_str());
        return 0;
    }
    const Kernel& c = k->kernel;
    memset(out, 0, sizeof *out);
    out->nwords = c.numWords;
    out->words = static_cast<uint32_t*>(malloc(c.numWords * 4));
    memcpy(out->words, c.words, c.numWords * 4);
    out->instrlen = c.instrlen;
    out->constlen = c.constlen;
    out->max_reg = c.maxReg;
    out->max_half_reg = c.maxHalfReg;
    out->double_threadsize = c.doubleThreadsize;
    out->branchstack = c.branchstackHw;
    out->localid_regid = c.localIdRegid;
    out->wgid_regid = c.workGroupIdRegid;
    out->nimm = c.numImmediates;
    if (c.numImmediates) {
        out->imm = static_cast<uint32_t*>(malloc(c.numImmediates * 4));
        memcpy(out->imm, c.immediates, c.numImmediates * 4);
    }
    out->imm_vec4 = c.immediatesOffsetVec4;
    out->dp_vec4 = c.driverParamsOffsetVec4;
    out->num_ubos = c.numUbos;
    out->num_ssbos = c.numSsbos;
    out->num_ibos = c.numIbos;
    out->ssbo_undeclared = c.ssboUndeclared;
    out->shared_size = c.sharedSize > k->threadgroupBytes ? c.sharedSize : k->threadgroupBytes;
    out->pvtmem_size = c.pvtmemSize;
    out->const_data_size = c.constantDataSize;
    out->nranges = c.numUboRanges < LAV5_MAX_RANGES ? c.numUboRanges : LAV5_MAX_RANGES;
    for (uint32_t i = 0; i < out->nranges; i++) {
        out->ranges[i] = { c.uboRanges[i].block, c.uboRanges[i].offsetVec4, c.uboRanges[i].startDword, c.uboRanges[i].sizeDwords };
    }
    for (const auto& r : k->resources) {
        if ((r.kind == Resource::kConstantBuffer || r.kind == Resource::kDeviceBuffer) && r.index < LAV5_MAX_BUFFERS)
            out->buffer_mask |= 1u << r.index;
        else
            out->other_resources++;
    }
    if (k->hasThreadgroupArg) out->other_resources++;
    snprintf(out->name, sizeof out->name, "%s", k->name.c_str());
    return 1;
}

extern "C" void lavcc_kernel_free(struct lav5_kernel* k)
{
    free(k->words);
    free(k->imm);
    k->words = nullptr;
    k->imm = nullptr;
}

// ---- the 3D side: a vertex and a fragment function compiled, linked and flattened for lava5r.c ----

static void flatten(const Kernel& c, const std::vector<Resource>& resources, struct lav5_kernel* out)
{
    memset(out, 0, sizeof *out);
    out->nwords = c.numWords;
    out->words = static_cast<uint32_t*>(malloc(c.numWords * 4));
    memcpy(out->words, c.words, c.numWords * 4);
    out->instrlen = c.instrlen;
    out->constlen = c.constlen;
    out->max_reg = c.maxReg;
    out->max_half_reg = c.maxHalfReg;
    out->double_threadsize = c.doubleThreadsize;
    out->branchstack = c.branchstackHw;
    out->nimm = c.numImmediates;
    if (c.numImmediates) {
        out->imm = static_cast<uint32_t*>(malloc(c.numImmediates * 4));
        memcpy(out->imm, c.immediates, c.numImmediates * 4);
    }
    out->imm_vec4 = c.immediatesOffsetVec4;
    out->dp_vec4 = c.driverParamsOffsetVec4;
    out->num_ubos = c.numUbos;
    out->num_ssbos = c.numSsbos;
    out->num_ibos = c.numIbos;
    out->shared_size = c.sharedSize;
    out->pvtmem_size = c.pvtmemSize;
    out->const_data_size = c.constantDataSize;
    out->nranges = c.numUboRanges < LAV5_MAX_RANGES ? c.numUboRanges : LAV5_MAX_RANGES;
    for (uint32_t i = 0; i < out->nranges; i++)
        out->ranges[i] = { c.uboRanges[i].block, c.uboRanges[i].offsetVec4, c.uboRanges[i].startDword, c.uboRanges[i].sizeDwords };
    for (const auto& r : resources) {
        if ((r.kind == Resource::kConstantBuffer || r.kind == Resource::kDeviceBuffer) && r.index < LAV5_MAX_BUFFERS)
            out->buffer_mask |= 1u << r.index;
        else
            out->other_resources++;
    }
}

static void flatten_stage(const Stage& s, const std::vector<Resource>& resources, struct lav5_stage* out)
{
    flatten(s.k, resources, &out->k);
    out->is_fragment = s.kind == kStageFragment;
    out->nin = s.numInputs < LAV5_MAX_IO ? s.numInputs : LAV5_MAX_IO;
    for (uint32_t i = 0; i < out->nin; i++)
        out->in[i] = { s.inputs[i].slot, s.inputs[i].regid, s.inputs[i].compmask, s.inputs[i].inloc, s.inputs[i].sysval,
                       s.inputs[i].bary, s.inputs[i].flat, s.inputs[i].half };
    out->nout = s.numOutputs < 32 ? s.numOutputs : 32;
    for (uint32_t i = 0; i < out->nout; i++) {
        out->out[i].slot = s.outputs[i].slot;
        out->out[i].regid = s.outputs[i].regid;
        out->out[i].half = s.outputs[i].half;
    }
    out->nattr = s.numAttributes;
    out->vertex_id_regid = s.vertexIdRegid;
    out->instance_id_regid = s.instanceIdRegid;
    out->position_regid = s.positionRegid;
    out->point_size_regid = s.pointSizeRegid;
    out->clip_mask = s.clipMask;
    out->cull_mask = s.cullMask;
    out->total_in = s.totalIn;
    out->frag_coord_regid = s.fragCoordRegid;
    out->face_regid = s.faceRegid;
    out->frag_coord_compmask = s.fragcoordCompmask;
    for (uint32_t i = 0; i < 7; i++) out->ij_regid[i] = s.ijRegid[i];
    for (uint32_t i = 0; i < 8; i++) {
        out->color_regid[i] = s.colorRegid[i];
        out->color_half[i] = s.colorHalf[i];
    }
    out->ncolor = s.numColorOutputs;
    out->depth_regid = s.depthRegid;
    out->need_pixlod = s.needPixlod;
    out->has_kill = s.hasKill;
    out->writes_pos = s.writesPos;
    out->ntexslots = s.k.numTextureSlots;
    for (uint32_t i = 0; i < 32 && i < kMaxTextureSlots; i++) out->fb_target[i] = s.k.textures[i].fbTarget;
}

static std::unique_ptr<CompiledStage> stage_of(struct lavcc_lib* l, const char* name, uint8_t type, std::string* why,
                                               const FunctionConstants& constants = {}, const KnownBuffer* known = nullptr)
{
    const MetalFunction* fn = l->lib.find(name);
    if (!fn) { *why = std::string("no function ") + name; return nullptr; }
    if (fn->type != type) { *why = std::string(name) + " is not the right kind of function"; return nullptr; }
    const uint8_t* bc = nullptr;
    uint64_t n = 0;
    if (!l->lib.bitcode(*fn, &bc, &n)) { *why = std::string("no bitcode for ") + name; return nullptr; }
    return compileStage(bc, n, why, constants, 0, known);
}

extern "C" int lavcc_compile_program_c(struct lavcc_lib* lv, const char* vname, const struct lavcc_const* vc, unsigned nvc,
                                       struct lavcc_lib* lf, const char* fname, const struct lavcc_const* fc, unsigned nfc,
                                       struct lav5_stage* vs, struct lav5_stage* fs, struct lav5_link* link, char* why, size_t whysize);

extern "C" int lavcc_compile_program(struct lavcc_lib* lv, const char* vname, struct lavcc_lib* lf, const char* fname,
                                     struct lav5_stage* vs, struct lav5_stage* fs, struct lav5_link* link, char* why, size_t whysize)
{
    return lavcc_compile_program_c(lv, vname, nullptr, 0, lf, fname, nullptr, 0, vs, fs, link, why, whysize);
}

static int compile_program(struct lavcc_lib* lv, const char* vname, const struct lavcc_const* vc, unsigned nvc,
                           struct lavcc_lib* lf, const char* fname, const struct lavcc_const* fc, unsigned nfc, const KnownBuffer* known,
                           struct lav5_stage* vs, struct lav5_stage* fs, struct lav5_link* link, char* why, size_t whysize);

extern "C" int lavcc_compile_program_c(struct lavcc_lib* lv, const char* vname, const struct lavcc_const* vc, unsigned nvc,
                                       struct lavcc_lib* lf, const char* fname, const struct lavcc_const* fc, unsigned nfc,
                                       struct lav5_stage* vs, struct lav5_stage* fs, struct lav5_link* link, char* why, size_t whysize)
{
    return compile_program(lv, vname, vc, nvc, lf, fname, fc, nfc, nullptr, vs, fs, link, why, whysize);
}

extern "C" int lavcc_compile_program_known(struct lavcc_lib* lv, const char* vname, const struct lavcc_const* vc, unsigned nvc,
                                           struct lavcc_lib* lf, const char* fname, const struct lavcc_const* fc, unsigned nfc,
                                           unsigned binding, const uint8_t* bytes, const uint8_t* fold, size_t length,
                                           struct lav5_stage* vs, struct lav5_stage* fs, struct lav5_link* link, char* why, size_t whysize)
{
    KnownBuffer kb;
    kb.binding = binding;
    kb.bytes.assign(bytes, bytes + length);
    kb.fold.assign(fold, fold + length);
    return compile_program(lv, vname, vc, nvc, lf, fname, fc, nfc, &kb, vs, fs, link, why, whysize);
}

extern "C" int lavcc_control_bytes(struct lavcc_lib* lf, const char* fname, const struct lavcc_const* fc, unsigned nfc, unsigned binding,
                                   size_t length, uint8_t* mask, char* why, size_t whysize)
{
    const MetalFunction* fn = lf->lib.find(fname);
    const uint8_t* bc = nullptr;
    uint64_t n = 0;
    if (!fn || !lf->lib.bitcode(*fn, &bc, &n)) { snprintf(why, whysize, "no bitcode for %s", fname); return 0; }
    FunctionConstants consts;
    for (unsigned i = 0; i < nfc; i++) consts.push_back({fc[i].index, fc[i].bits});
    std::vector<uint8_t> m;
    std::string w;
    if (!knownBufferControlBytes(bc, n, consts, binding, length, &m, &w)) { snprintf(why, whysize, "%s", w.c_str()); return 0; }
    memset(mask, 0, length);
    memcpy(mask, m.data(), m.size() < length ? m.size() : length);
    return 1;
}

static int compile_program(struct lavcc_lib* lv, const char* vname, const struct lavcc_const* vc, unsigned nvc,
                           struct lavcc_lib* lf, const char* fname, const struct lavcc_const* fc, unsigned nfc, const KnownBuffer* known,
                           struct lav5_stage* vs, struct lav5_stage* fs, struct lav5_link* link, char* why, size_t whysize)
{
    std::string w;
    FunctionConstants vconsts, fconsts;
    for (unsigned i = 0; i < nvc; i++) vconsts.push_back({vc[i].index, vc[i].bits});
    for (unsigned i = 0; i < nfc; i++) fconsts.push_back({fc[i].index, fc[i].bits});
    auto v = stage_of(lv, vname, MetalFunction::kVertex, &w, vconsts);
    if (!v) { snprintf(why, whysize, "vertex function: %s", w.c_str()); return 0; }
    auto f = stage_of(lf, fname, MetalFunction::kFragment, &w, fconsts, known);
    if (!f) { snprintf(why, whysize, "fragment function: %s", w.c_str()); return 0; }
    Linkage lk{};
    if (!render::link(v->stage, f->stage, &lk, &w)) { snprintf(why, whysize, "link: %s", w.c_str()); return 0; }
    memset(vs, 0, sizeof *vs);
    memset(fs, 0, sizeof *fs);
    memset(link, 0, sizeof *link);
    flatten_stage(v->stage, v->resources, vs);
    flatten_stage(f->stage, f->resources, fs);
    if (const char* dir = access("/var/tmp/lavlist", F_OK) == 0 ? "/var/tmp/lst" : nullptr) {
        mkdir(dir, 0777);
        // the ir3 disassembly of the fragment stage, for the eye (one file per compile)
        static unsigned seq;
        char path[300];
        snprintf(path, sizeof path, "%s/fs-%s-%s-%u.txt", dir, fname, known ? "known" : "plain", seq++);
        if (FILE* fp = fopen(path, "w")) {
            fwrite(f->listing.data(), 1, f->listing.size(), fp);
            fclose(fp);
        }
    }
    link->cnt = lk.cnt;
    link->max_loc = lk.maxLoc;
    link->position_loc = lk.positionLoc;
    link->psize_loc = lk.psizeLoc;
    for (int i = 0; i < 4; i++) link->varmask[i] = lk.varmask[i];
    for (int i = 0; i < 8; i++) link->interp[i] = lk.interpMode[i];
    for (uint32_t i = 0; i < 32; i++) {
        link->var[i].slot = lk.var[i].slot;
        link->var[i].regid = lk.var[i].regid;
        link->var[i].compmask = lk.var[i].compmask;
        link->var[i].loc = lk.var[i].loc;
    }
    return 1;
}

// the names of a function's constants: air.function_constants holds one node each, strings (the name among them) then the index
extern "C" unsigned lavcc_function_constants(struct lavcc_lib* l, const char* fname, struct lavcc_fcname* out, unsigned max)
{
    const MetalFunction* fn = l->lib.find(fname);
    const uint8_t* bc = nullptr;
    uint64_t bcSize = 0;
    unsigned num = 0;
    if (!fn || !l->lib.bitcode(*fn, &bc, &bcSize)) return 0;
    void* ctx = ralloc_context(nullptr);
    const char* err = nullptr;
    struct air_module* m = air_module_read(ctx, bc, bcSize, &err);
    if (m) {
        for (unsigned i = 0; i < m->num_named_md; i++) {
            if (strcmp(m->named_md[i].name, "air.function_constants") != 0) continue;
            for (unsigned j = 0; j < m->named_md[i].num_ops && num < max; j++) {
                const struct air_md* node = m->named_md[i].ops[j];
                if (!node || node->kind != AIR_MD_NODE) continue;
                const char* names[4];
                unsigned nn = 0;
                for (unsigned k = 0; k < node->num_ops; k++) {
                    const struct air_md* op = node->ops[k];
                    if (!op) continue;
                    if (op->kind == AIR_MD_STRING) {
                        const char* s = air_md_string(op);
                        if (s && nn < 4) names[nn++] = s;
                        continue;
                    }
                    uint64_t idx;
                    if (op->kind == AIR_MD_VALUE && air_const_uint(op->value, &idx)) {
                        // every string of the node answers to the index: the constant's name is one of them
                        for (unsigned q = 0; q < nn && num < max; q++) {
                            snprintf(out[num].name, sizeof out[num].name, "%s", names[q]);
                            out[num].index = uint32_t(idx);
                            num++;
                        }
                        break;
                    }
                }
            }
        }
    }
    ralloc_free(ctx);
    return num;
}

extern "C" void lavcc_stage_free(struct lav5_stage* s) { lavcc_kernel_free(&s->k); }

// ---- a survey of a library: what each function needs, and whether this compiler serves it (lavsurvey) ----

extern "C" int lavcc_probe(struct lavcc_lib* l, unsigned i, char* summary, size_t ssize, char* why, size_t whysize)
{
    if (i >= l->lib.functions().size()) return -1;
    const MetalFunction& fn = l->lib.functions()[i];
    const uint8_t* bc = nullptr;
    uint64_t n = 0;
    summary[0] = 0;
    if (!l->lib.bitcode(fn, &bc, &n)) { snprintf(why, whysize, "no bitcode"); return 0; }
    std::string w;
    if (fn.type == MetalFunction::kKernel) {
        auto k = compileAir(bc, n, &w);
        if (!k) { snprintf(why, whysize, "%s", w.c_str()); return 0; }
        snprintf(summary, ssize, "kernel %u words", k->kernel.numWords);
        return 1;
    }
    auto s = compileStage(bc, n, &w);
    if (!s) { snprintf(why, whysize, "%s", w.c_str()); return 0; }
    unsigned buf = 0, tex = 0, smp = 0, img = 0;
    for (const auto& r : s->resources) {
        if (r.kind == Resource::kTexture) tex++;
        else if (r.kind == Resource::kSampler) smp++;
        else if (r.kind == Resource::kImage) img++;
        else buf++;
    }
    lav5_stage ls;
    memset(&ls, 0, sizeof ls);
    flatten_stage(s->stage, s->resources, &ls);
    const char* refuse = lav5_refuse_stage(&ls);
    snprintf(summary, ssize, "%s %u words, %u buffers %u textures %u samplers %u images; ssbos %u ubos %u ncolor %u%s%s", fn.type == MetalFunction::kVertex ? "vertex" : "fragment",
             s->stage.k.numWords, buf, tex, smp, img, ls.k.num_ssbos, ls.k.num_ubos, ls.ncolor, refuse ? " -- DRIVER REFUSES: " : "", refuse ? refuse : "");
    lavcc_stage_free(&ls);
    return 1;
}

// every vertex function with every fragment function of a library, each stage compiled once: which pairs link and which of those the
// driver refuses (lavsurvey pairs). Prints to stdout.
extern "C" void lavcc_survey_pairs(struct lavcc_lib* l)
{
    const auto& fns = l->lib.functions();
    std::vector<std::unique_ptr<CompiledStage>> st(fns.size());
    std::vector<std::string> whyc(fns.size());
    for (size_t i = 0; i < fns.size(); i++) {
        if (fns[i].type != MetalFunction::kVertex && fns[i].type != MetalFunction::kFragment) continue;
        std::string w;
        st[i] = stage_of(l, fns[i].name.c_str(), fns[i].type, &w);
        if (!st[i]) printf("NOCOMPILE %s: %s\n", fns[i].name.c_str(), w.c_str());
    }
    unsigned tried = 0, usable = 0, refused = 0, nolink = 0;
    std::map<std::string, unsigned> reasons, refusedBy;
    for (size_t i = 0; i < fns.size(); i++) {
        if (fns[i].type != MetalFunction::kVertex || !st[i]) continue;
        unsigned vusable = 0;
        for (size_t j = 0; j < fns.size(); j++) {
            if (fns[j].type != MetalFunction::kFragment || !st[j]) continue;
            tried++;
            Linkage lk{};
            std::string w;
            if (!render::link(st[i]->stage, st[j]->stage, &lk, &w)) {
                nolink++;
                for (auto& c : w) if (c >= '0' && c <= '9') c = '#';
                // drop the function-specific part before the first colon-free tail
                reasons[w.substr(0, 110)]++;
                continue;
            }
            struct lav5_stage vs, fs;
            memset(&vs, 0, sizeof vs);
            memset(&fs, 0, sizeof fs);
            flatten_stage(st[i]->stage, st[i]->resources, &vs);
            flatten_stage(st[j]->stage, st[j]->resources, &fs);
            const char* e = lav5_refuse_stage(&vs);
            const char* who = fns[i].name.c_str();
            if (!e) { e = lav5_refuse_stage(&fs); who = fns[j].name.c_str(); }
            if (e) {
                refused++;
                refusedBy[std::string(who) + ": " + e]++;
                reasons[std::string("REFUSED: ") + e]++;
            } else {
                usable++;
                vusable++;
            }
            lavcc_stage_free(&vs);
            lavcc_stage_free(&fs);
        }
        printf("VS %-44s usable with %u fragment functions\n", fns[i].name.c_str(), vusable);
        fflush(stdout);
    }
    printf("pairs: %u tried, %u usable, %u refused by the driver, %u do not link\n", tried, usable, refused, nolink);
    for (auto& kv : reasons) printf("  %5u x %s\n", kv.second, kv.first.c_str());
    for (auto& kv : refusedBy) printf("  refused %5u x %s\n", kv.second, kv.first.c_str());
}
