import concurrent.futures,json,pathlib,shlex,subprocess,tarfile,time
R=pathlib.Path('/Users/ali.yzf/zvec'); O=R/'benchmarks/results/memory_block_cohere_100k_20261013'; DB=json.loads((R/'build/compile_commands.json').read_text())
entries=[x for x in DB if 'CMakeFiles/zvec_core.dir/' in x['command']]
cc='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++'
for name,ref in [('main','13fd088b'),('current','ffabef5c')]:
 d=O/name; src=d/'source'; obj=d/'objects'; src.mkdir(parents=True,exist_ok=True);obj.mkdir(exist_ok=True)
 if not (src/'src').exists():
  p=subprocess.Popen(['git','archive',ref,'src','tools'],cwd=R,stdout=subprocess.PIPE)
  with tarfile.open(fileobj=p.stdout,mode='r|') as tar:tar.extractall(src)
  assert p.wait()==0
 def compile(e):
  rel=pathlib.Path(e['file']).relative_to(R); out=obj/(str(rel).replace('/','_')+'.o'); log=out.with_suffix('.log')
  if out.exists():return out
  a=shlex.split(e['command']); j=a.index('-o');a[j+1]=str(out)
  a=[v.replace(str(R/'src'),str(src/'src')) if str(R/'src') in v else v for v in a]
  a=[('-Dzvec_core_VERSION="'+ref+'"') if v.startswith('-Dzvec_core_VERSION=') else v for v in a]
  with log.open('w') as f: result=subprocess.run(a,cwd=e['directory'],stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(str(log)+'\n'+log.read_text()[-5000:])
  print(name,'compiled',rel,flush=True);return out
 t=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:objects=list(pool.map(compile,entries))
 lib=d/'libcore.a';subprocess.run(['ar','rcs',str(lib)]+list(map(str,objects)),check=True)
 e=next(x for x in DB if x['file'].endswith('/tools/core/local_builder_original.cc'))
 a=shlex.split(e['command']);j=a.index('-o');a[j+1]=str(d/'builder.o');a=[v.replace(str(R/'src'),str(src/'src')).replace(str(R/'tools'),str(src/'tools')) for v in a]
 subprocess.run(a,cwd=e['directory'],check=True)
 link=[cc,'-isysroot','/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.1.sdk','-O3','-march=armv8',str(d/'builder.o'),'-Wl,-force_load,'+str(lib),'-Wl,-force_load,'+str(R/'build/lib/libzvec_turbo.a'),str(R/'build/lib/libzvec_ailego.a'),str(R/'build/external/usr/local/lib/libroaring.a'),str(R/'build/external/usr/local/lib/libyaml-cpp.a'),str(R/'build/external/usr/local/lib/libgflags_nothreads.a'),'-o',str(d/'builder')]
 subprocess.run(link,check=True)
 print('DONE',name,time.monotonic()-t,flush=True)
