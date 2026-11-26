# lavender-metal

GPU bring-up for the **Redmi Note 7 (lavender, Snapdragon 660, Adreno 512)** running iOS 16 through the
[lavender-ios16](https://github.com/pcorezzola/lavender-ios16) port: an Adreno 512 kernel driver for XNU and a
Metal-compatible layer on top of it, so that iOS stops drawing everything on the CPU.

## Why

The port boots to the iOS 16.7.16 lock screen, but there is no GPU: `MTLCreateSystemDefaultDevice()` returns nil and
QuartzCore falls back to its software OpenGL renderer (`CA::OGL::SW`). On the device `backboardd` uses ~150-170 % of
the four A73 cores with the lock screen idle, frames come out at ~13 fps on average (30 Hz panel), and wallpapers that
need Metal (Astronomy, ...) fail. See `lavender-ios16/README.md`, section *Performance*.

## Starting point

[`oosexgang/hakios-xnu-kernel`](https://github.com/oosexgang/hakios-xnu-kernel) (`hakios-metal/`, MIT, and the kernel
side in `drivers/gpu/`, APSL) is a working Metal driver for **Adreno A6xx** (Adreno 642L, GMU): the Metal command model,
compute, textures, render passes, an AIR -> NIR -> ir3 shader compiler built on Mesa, and the `MTLDevice` plug-in. It is
the architecture we follow, not code that runs on this GPU:

| | hakios (A6xx) | this device (A5xx) |
| --- | --- | --- |
| Power-up | GMU | no GMU: GPMU, CP/PFP microcode, zap shader through TrustZone |
| Registers / packets | `a6xx.hpp`, `CP_LOAD_STATE6` | `a5xx.xml`, `CP_LOAD_STATE4` |
| Reference for sequences | turnip (Vulkan) | none (turnip never supported A5xx): Gallium `fd5` |
| SoC GPU rails | SC7280 | SDM660: `gpucc-sdm660`, CPR `gfx` corner, `gfx_ldo` |

The reference tree is kept (sparse, not committed) under `refs/hakios` at the commit recorded in `docs/REFERENCES.md`.

## Roadmap

Details and "done when" criteria per milestone are in [`docs/PLAN.md`](docs/PLAN.md). Struck-through items are done and
verified (on the device unless noted).

**M0 - Reconnaissance** (done 2026-11-21)

- ~~Extract the GPU firmware from the stock ROM~~
- ~~Decode the GPU node of the DTB (registers, IRQ, clocks, GDSCs, SMMU, zap pas-id)~~
- ~~Work out what Mesa and hakios already cover (chip id, ir3/fd5, chip-independent IOKit service)~~

**M1 - Power and identify** (done 2026-11-21)

- ~~`lavender.gpu` XNU driver behind the `lavgpu=N` boot-arg~~
- ~~GPU clocks and GX rail up without hanging the SoC~~
- ~~Read the GPU register block (chip id, CP version, RBBM status)~~
- ~~TrustZone confirms PAS id 13~~

**M2 - Ring, microcode, fence** (done 2026-11-21)

- ~~Zap shader loaded through SCM~~
- ~~GPU SMMU stage 1 identity table~~
- ~~PM4/PFP microcode loaded, ring buffer set up, `ME_INIT`~~
- ~~Secure mode off, CP-written fence read back by the CPU~~
- ~~Deliberate hang recovered with a GPU reset, no reboot~~

**M3 - A5xx register layer and compute** (done 2026-11-21)

- ~~`a5xx` register and PM4 packet layer, checked against Mesa's `a5xx.xml`~~
- ~~Fake CP / GPU interpreter and host tests~~
- ~~First compute dispatch on the device (hand-assembled shader wrote `0xc0de0001`)~~
- ~~Buffers in and out (`out = in + 0x11111111`)~~
- ~~64-thread workgroup: 64 of 64 outputs equal the host reference~~

**M4 - Render** (done 2026-11-22, depth parked)

- ~~2D blit through the A5xx 2D core: buffer copy, then RGBA8 rectangle copy, pixel exact~~
- ~~First sysmem 3D draw: hand-assembled VS/FS, triangle into a 64x64 RGBA8 target, 351/351 pixels~~
- ~~Texture sampling (4x4 RGBA8, nearest)~~
- ~~Interpolated texture coordinates through the VPC, 213/213 pixels~~
- ~~Colour blending (SRC_ALPHA / ONE_MINUS_SRC_ALPHA, ONE/ZERO, ONE/ONE), 351/351 pixels~~
- ~~GPU output on the real panel (the kernel copies the GPU-rendered targets 4x into the scanout; the RB cannot target it directly because its stride is not a multiple of 64)~~
- 2D solid fill: parked, no variant wrote anything
- [ ] Depth (optional; only if CoreAnimation needs it)

**M5 - The Metal device** (in progress; `backboardd` renders on the GPU since 2026-11-24)

- ~~M5.0: the GPU as an IOKit service, `LavenderGPU`, with a user client (info, buffer new/free mapped into the client, submit of PM4 words, submit by reference). Verified on the device from the kernel (selftest) and from a userland program: the GPU writes into a mapped buffer and the program reads it back (patch 0319 in the port)~~
- ~~M5.1: a device that exists: the `LavenderMetal` bundle, and `MTLCreateSystemDefaultDevice()` returns it on the phone (the plug-in properties are put on `IOAcceleratorES` on demand by the service)~~
- ~~M5.2: buffers, command queue, command buffer and blit encoder (fill by the command processor, copy by the 2D core) through Metal's own objects, verified byte by byte on the phone~~
- ~~M5.2b: textures (1D/2D, A8/R8/R16/RG8/R32/RG16F/RGBA16F/RGB10A2/sRGB, mip level 0) and samplers; the 16 texture/sampler units are always emitted (1x1 descriptors in the empty ones), otherwise QuartzCore's uber fragment shader hangs the CP~~
- ~~M5.3: libraries and the shader compiler (AIR -> NIR -> ir3 for A5xx), function constants by index and by name, compute encoder~~
- ~~M5.4: render pipeline and render encoder~~
- ~~M5.5: CAMetalLayer drawable over an IOSurface, presented~~
- ~~M5.6: CoreAnimation over Metal: the lock screen (wallpaper, date, clock, camera button, bottom bar) is drawn by `backboardd` through the plug-in; Control Centre and the home screen with the plug-in on still show glitches, see *Open problems*~~
- [ ] Idle lock screen costs a fraction of the software renderer's `backboardd` CPU (software: ~150-170 % of 400 %, ~13 fps); animations are still slow, see *Findings*
- [ ] Astronomy wallpaper (`AegirPoster`, Metal): not working, the extension aborts on launch (see *Open problems*)

## Findings (2026-11-25)

Measured on the phone; the numbers are what to expect, not what to hope for.

- **The GPU cannot touch memory at or above 4 GB.** Any write (`CP_BLIT`, `CP_MEM_WRITE`) to a physical address >= 4 GB hangs
  the GPU with *no* SMMU fault (GFSR/FSR/FAR/FSYNR0 all zero) and every later submit fails until the phone is rebooted.
  Extending the CB4 identity table to 8 GB or setting `TCR.IPS` made no difference. About 65 % of the pages of the 1080x2340
  scanout IOSurface (622 scattered 16 KB runs) are above 4 GB, so GPU-to-IOSurface copies are not possible and the GPU
  can only work on its own buffers below 4 GB. Open hypothesis: only 2 of the 12 `*_ADDR_MODE_CNTL` registers are written
  in the service path (the rest only in the compute self-test); `lavaddr64=1` (patch 0366) writes them all, untested.
- **Cost of a frame is the copy, not the GPU.** Textures are mapped write-combined and the CPU reads them very slowly;
  the IOSurface copy took 25-90 ms per commit. Cacheable (write-back) textures with an explicit data-cache flush of whatever
  the CPU touched (`lavexp=32`, patch 0365) bring it to ~3-15 ms (mean 6 ms); typical commits 7-20 ms.
- **Fixed cost per GPU submit** was ~11 ms (10 ms sleeps in the service's ring wait); polling every 50 us first brought
  commits of ~80 submits from ~1 s to 77-163 ms (patch 0359).
- **Fill rate** ~0.45-0.7 Gpix/s (160 MHz: ~0.45; 266 MHz: ~0.66). `lavgpuclk=266` boots and is stable. Neither the RB/CCU
  bits nor `VBIF` QoS (patch 0360) changed it; fewer pixels per frame (damage, GMEM, batching) is what is left.
- **Compile cost.** "Control bytes" analysis of the uber shader (~566 ms each, 2084 per boot) is now done once per
  pipeline and cached in memory and on disk (`/var/tmp/lavmetal-cache/*.ctl`, `*.spec`): a warm boot does 0 analyses and 0
  shader compiles, the first commit takes ~2 s instead of 12-47 s.
- **Debug switches** (all cost nothing when the marker file is absent): `/var/tmp/lavexp` (bit 0 DP2, 1 CCU, 2 GPU copy -
  do not use, 3 blit self-test, 4 surface self-test, 5 cacheable textures), `/var/tmp/lavtrace`, `/var/tmp/lavlist`,
  `/var/tmp/lavshot` (dumps the surface to `/var/tmp/lavsurf.bgra`; needs `lavtrace`), `/var/tmp/lavprof`.

## Open problems

- **Wallpaper previews, widgets and blurs.** In the poster gallery the previews are blank white (text and complications
  draw), the category icons are empty circles, the Calendar widget is a white square, and Control Centre shows a flat grey
  backdrop with a black band at the top and black dashes near the bottom. It looked the same with the plug-in off, so
  it is probably the materials/blur path, not the compositor; not yet isolated.
- **Astronomy (`AegirPoster`)** aborts a fraction of a second after launch inside `NUNIAegirRenderer
  getOrCreateUniformBufferforFrameBufferIndex:` (a Metal buffer comes back nil). It respawns every ~10 s. It is a second GPU
  client next to `backboardd`, which hung the GPU before: **only one process may use the plug-in at a time.**
- **After a reboot the plug-in is off:** run `lavmetal_switch on`, then restart `backboardd` and SpringBoard. (A boot-arg
  `lavmetal=1` that does it at start is the first thing worth taking from the `aLpHa-Git-69` fork.)
- Hardware limits: HS400 eMMC does not boot; CPU Gold is capped at 1113.6 MHz (voltage is not programmable).

## Ideas worth taking from forks

`aLpHa-Git-69/lavender-metal` (9 commits on top of `main`): the `lavmetal=1` boot-arg, deferred compile of the uber
fragment shader, specialisation over several UBOs/SSBOs, no ir3 listing by default, and logging submits only when
they fail. The pool, GPU clock and fast-poll parts duplicate what is already here.

## Related repositories

- [pcorezzola/lavender-ios16](https://github.com/pcorezzola/lavender-ios16): the iOS 16 port for the Redmi Note 7 that 
  this GPU work serves. The kernel driver (`kernel/lavender_gpu.c` here) is built into that port as `drivers/gpu/lavender_gpu.c` 
  and the patches `xnu/patches/0300`-`0399`, and is switched on with the `lavgpu=N` boot-arg. Boot images and CI live there.
- this repository: research, host-side tests and the plan; no firmware or Qualcomm blobs are committed.

## Layout

```
docs/       plan, hardware notes, references, measurements
kernel/     the kernel driver (lavender_gpu.c) and the IOKit service (LavenderGPU.cpp), as built into the port
userland/   programs that run on the phone (lavgpu_test: exercises the IOKit service)
refs/       reference trees (not committed; see docs/REFERENCES.md)
firmware/   GPU firmware taken from the phone's own ROM (not committed)
tools/      scripts (firmware extraction, DTB reading, probes)
notes/      working notes per milestone
```

## Legal

Apple binaries and the phone's firmware are not distributed. Firmware is read from the user's own stock ROM. Kernel
code derived from XNU stays under the APSL; user-space code is MIT. See `docs/REFERENCES.md` for upstream licenses.
