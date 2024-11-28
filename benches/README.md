Benchmarking is hard... (c)

## Environment

- x86-64 test were run on Gentoo Linux with AMD 3700X.
- ARM test were run on Apple M1.
- tiny-skia SSE2 is built with `-Ctarget-cpu=x86-64`
- tiny-skia AVX is built with `-Ctarget-cpu=haswell`
- Skia v90.
- Skia SSE2 is built using clang with `-march=x86-64`
- Skia AVX is built using clang with `-march=haswell`
- cairo v1.16.0.
- cairo is built using gcc with `-march=native`, because it's the system one
- raqote version can be found at `Cargo.lock`
- raqote is built with `-Ctarget-cpu=x86-64`.
  Testing with `haswell` doesn't change the results much.
- Rust 1.62
- clang 13

## Results

[x86-64 Results](https://linebender.github.io/tiny-skia/x86_64.html)

[ARM Results](https://linebender.github.io/tiny-skia/arm.html)

## Running benchmarks

Benchmarks are using nightly Rust, so you have to install it first:

```sh
rustup toolchain install nightly
```

And to run benchmarks:

```sh
rustup run nightly cargo bench
```

By default, only tiny-skia is tested. To enable other libraries use `--features`:

```sh
# those are skia-rs specific exports
export SKIA_DIR="/path/to/skia"
export SKIA_LIB_DIR="/path/to/skia/out/Shared"
export LD_LIBRARY_PATH="/path/to/skia/out/Shared"

rustup run nightly cargo bench --features skia-rs,raqote,cairo-rs
```

You have to install cairo first and built Skia from sources (see below).

### Building Skia

You will need `git`, `clang`, `ninja` and Python.

On Windows, use `clang-cl` and `clang-cl++` for `cc` and `cxx` instead.

When building on macOS, you can remove `cc` and `cxx`. On M1 also remove `"-march=haswell"`

```sh
git clone https://skia.googlesource.com/skia.git
cd skia
git fetch --all
git checkout -b m90 origin/chrome/m90
python3 tools/git-sync-deps # this will download about 3 GiB of code
bin/gn gen out/Shared --args='
    is_official_build=false
    is_component_build=true
    is_debug=false
    cc="clang"
    cxx="clang++"
    extra_cflags_cc=["-march=haswell", "-DSK_FORCE_RASTER_PIPELINE_BLITTER"]
    werror=false
    paragraph_gms_enabled=false
    paragraph_tests_enabled=false
    skia_enable_android_utils=false
    skia_enable_discrete_gpu=false
    skia_enable_gpu=false
    skia_enable_nvpr=false
    skia_enable_pdf=false
    skia_enable_skottie=false
    skia_enable_skrive=false
    skia_enable_skshaper=false
    skia_enable_tools=false
    skia_use_expat=false
    skia_use_gl=false
    skia_use_harfbuzz=false
    skia_use_icu=false
    skia_use_libgifcodec=false
    skia_use_libheif=false
    skia_use_libjpeg_turbo_decode=false
    skia_use_libjpeg_turbo_encode=false
    skia_use_libwebp_decode=false
    skia_use_libwebp_encode=false
    skia_use_lua=false
    skia_use_piex=false'
ninja -C out/Shared
```
