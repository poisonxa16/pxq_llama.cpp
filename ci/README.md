# CI

`ci/run.sh` runs the full build-and-test sweep on your own machine. It is much
heavier than a plain `cmake --build`, and it is good practice to run it before
publishing changes:

```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with CUDA support
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with SYCL support
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

The first argument is the results directory, the second is a scratch directory
for models. The first run downloads the public reference models and the
wikitext-2 corpus into that scratch directory, so point it at a disk with room
and expect it to take a while; later runs reuse what is already there.

Release artifacts are built by GitHub Actions from
`.github/workflows/release-binaries.yml`.

`ci/run.sh` is inherited from upstream llama.cpp, which also runs it on its own
hosted CI fleet. That infrastructure is not part of this repository.
