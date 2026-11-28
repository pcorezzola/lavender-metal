#!/bin/bash
# Build the M5.1 - M5.3 pieces on the Mac (Xcode, iPhoneOS SDK, and the iOS Mesa archives tools/mesa-ios/build-mac.sh made):
#   out/LavenderMetal.bundle   the plug-in, to be copied to /System/Library/Extensions/ on the phone
#   out/lavmetal_test          the program that uses Metal through it
# The plug-in carries the shader compiler (hakios-metal's AIR front end and Mesa's ir3, in hk/ and the archives).
set -euo pipefail
cd "$(dirname "$0")"
. ../tools/mesa-ios/flags.sh
mkdir -p out/LavenderMetal.bundle
CC="xcrun -sdk iphoneos clang -arch arm64 -target arm64-apple-ios16.0 -O2 -Wall"

(cd hk && ./build-ios.sh)
$CC -std=gnu11 -c lava5.c -o out/lava5.o
$CC -std=gnu11 -c lava5r.c -o out/lava5r.o
$CXX $CXXF -Ihk/include -c lavcc.cpp -o out/lavcc.o
$CC -fno-objc-arc -c LavenderMetal.m -o out/LavenderMetal.o

LIBS="$D/src/compiler/air/libair.a $(for a in $(cat $B/archives.txt); do [ $a = src/compiler/air/libair.a ] || echo -n "$D/$a "; done)"
$CXX -arch arm64 -isysroot "$SDK" -target arm64-apple-ios16.0 -bundle -o out/LavenderMetal.bundle/LavenderMetal \
	out/LavenderMetal.o out/lava5.o out/lava5r.o out/lavcc.o hk/out/*.o $LIBS -framework Foundation -framework IOKit -framework IOSurface -lc++ -lz -lm
cp LavenderMetal-Info.plist out/LavenderMetal.bundle/Info.plist
# the plug-in fails to load (Metal returns no device) once the unstripped binary grows past ~8.27 MB: strip it
strip -x out/LavenderMetal.bundle/LavenderMetal 2>/dev/null
codesign -s - -f out/LavenderMetal.bundle
$CC -fobjc-arc -framework Foundation -framework Metal -framework QuartzCore -framework IOSurface -framework IOKit -framework CoreFoundation -o out/lavmetal_test lavmetal_test.m
codesign -s - -f out/lavmetal_test
$CC -fobjc-arc -framework Foundation -framework Metal -framework IOKit -o out/lavqc_test lavqc_test.m
codesign -s - -f out/lavqc_test
$CC -fobjc-arc -framework Foundation -framework Metal -framework QuartzCore -framework CoreGraphics -framework IOKit -framework IOSurface -o out/lavca_test lavca_test.m
$CC -framework IOKit -o out/lavmetal_switch lavmetal_switch.c
codesign -s - -f out/lavmetal_switch
codesign -s - -f out/lavca_test
echo built
