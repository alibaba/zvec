# zvec benchmark runner
# Usage: do doscript\benchmark.do [hnsw|ivf|flat] [sift|random] [topk]

global_variable = index_type, dataset, topk, root, cmake_check

// Auto-detect project root: check for CMakeLists.txt in CWD
root = capture "cd"
cmake_check = capture 'if exist "{root}\CMakeLists.txt" (echo yes) else (echo no)'

if cmake_check == "no"
  say "CMakeLists.txt not found in current directory."
  ask root "Enter full path to zvec project root (e.g. C:\Users\User\zvec-0.2.0):"
end_if

say 'Using project root: {root}'

if arg1 == ""
  ask index_type "Index type? (hnsw/ivf/flat):"
else
  index_type = arg1
end_if

if arg2 == ""
  ask dataset "Dataset? (sift/random):"
else
  dataset = arg2
end_if

if arg3 == ""
  ask topk "Top-K results? (e.g. 10):"
else
  topk = arg3
end_if

if index_type == ""
  index_type = "hnsw"
end_if

if dataset == ""
  dataset = "sift"
end_if

if topk == ""
  topk = "10"
end_if

run 'mkdir "{root}\data" 2>nul & mkdir "{root}\bench_results" 2>nul'

if dataset == "sift"
  say "Downloading SIFT-128 dataset..."
  download "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" to '{root}\data\sift.tar.gz'
  run 'tar -xzf "{root}\data\sift.tar.gz" -C "{root}\data"'
end_if

if dataset == "random"
  say "Generating random test vectors..."
  run 'py -c "import numpy as np; np.random.rand(10000,128).astype(\"float32\").tofile(r\"{root}\data\\random_base.fvecs\")"'
end_if

say 'Running {index_type} benchmark (top-{topk})...'
run '"{root}\build\tools\bench" --index {index_type} --data "{root}\data" --topk {topk} --output "{root}\bench_results\result_{index_type}.json"'

say 'Benchmark complete! Results in {root}\bench_results\result_{index_type}.json'
