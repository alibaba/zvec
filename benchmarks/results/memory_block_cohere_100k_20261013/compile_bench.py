import pathlib,subprocess,sys
R=pathlib.Path('/Users/ali.yzf/zvec');O=pathlib.Path(__file__).resolve().parent
cc='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++'
for name in sys.argv[1:] or ['main','current']:
 d=O/name;s=d/'source'
 args=[cc,'-isysroot','/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.1.sdk','-std=c++17','-O3','-DNDEBUG','-march=armv8','-I'+str(s/'src/include'),'-I'+str(s/'src'),str(O/'bench.cc'),'-Wl,-force_load,'+str(d/'libcore.a'),'-Wl,-force_load,'+str(R/'build/lib/libzvec_turbo.a'),str(R/'build/lib/libzvec_ailego.a'),str(R/'build/external/usr/local/lib/libroaring.a'),'-o',str(d/'bench')]
 subprocess.run(args,check=True)
 subprocess.run(['otool','-L',str(d/'bench')],check=True)
