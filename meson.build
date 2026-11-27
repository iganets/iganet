project(
  'iganet',
  'cpp',
  version: '25.06.0',
  meson_version: '>=1.3.0',
  default_options: [
    'buildtype=release',
    'cpp_std=c++20',
    'warning_level=1',
    'b_ndebug=if-release',
  ],
  license: 'MPL-2.0',
)

fs = import('fs')
cpp = meson.get_compiler('cpp')
iganet_version_parts = meson.project_version().split('.')

# ---------------------------------------------------------------------
# LibTorch
# ---------------------------------------------------------------------
# LibTorch ships CMake package files and no pkg-config file. The default path
# runs find_package(Torch) through Meson; the directory is given with
# -Dcmake_prefix_path=<libtorch>. The prefix path builds the dependency by hand.
torch_modules = ['torch', 'torch_cpu', 'c10']
torch_defs = ['USE_C10D_GLOO', 'USE_DISTRIBUTED', 'USE_RPC', 'USE_TENSORPIPE']
torch_def_args = []
foreach d : torch_defs
  torch_def_args += '-D' + d
endforeach

if get_option('torch_method') == 'cmake'
  torch_dep = dependency('Torch', method: 'cmake', modules: torch_modules)
  torch_includes = torch_dep.get_variable(cmake: 'TORCH_INCLUDE_DIRS').split()
  torch_abi = torch_dep.get_variable(cmake: 'TORCH_CXX_FLAGS')
  torch_root = fs.parent(torch_includes[0])
else
  torch_root = get_option('torch_prefix')
  if torch_root == ''
    error('torch_method=prefix needs -Dtorch_prefix=<libtorch directory>')
  endif
  torch_config = torch_root / 'share' / 'cmake' / 'Torch' / 'TorchConfig.cmake'
  if not fs.is_file(torch_config)
    error('no TorchConfig.cmake below torch_prefix: ' + torch_config)
  endif
  # The archive records the C++ ABI it was built for in TorchConfig.cmake.
  torch_config_text = fs.read(torch_config)
  if torch_config_text.contains('_GLIBCXX_USE_CXX11_ABI=1')
    torch_abi = '-D_GLIBCXX_USE_CXX11_ABI=1'
  elif torch_config_text.contains('_GLIBCXX_USE_CXX11_ABI=0')
    torch_abi = '-D_GLIBCXX_USE_CXX11_ABI=0'
  else
    error('TorchConfig.cmake does not state _GLIBCXX_USE_CXX11_ABI')
  endif
  torch_includes = [
    torch_root / 'include',
    torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include',
  ]
  torch_libdir = torch_root / 'lib'
  torch_libs = []
  foreach lib : torch_modules
    torch_libs += cpp.find_library(lib, dirs: [torch_libdir], required: true)
  endforeach
  torch_dep = declare_dependency(
    include_directories: include_directories(torch_includes, is_system: true),
    compile_args: [torch_abi] + torch_def_args,
    dependencies: torch_libs,
    link_args: ['-Wl,-rpath,' + torch_libdir],
  )
endif

torch_libdir = torch_root / 'lib'
iganet_with_cuda = fs.exists(torch_libdir / 'libtorch_cuda.so')
iganet_with_hip = fs.exists(torch_libdir / 'libtorch_hip.so')
iganet_with_xpu = fs.exists(torch_libdir / 'libtorch_xpu.so')
if get_option('cpuonly')
  iganet_with_cuda = false
  iganet_with_hip = false
  iganet_with_xpu = false
elif iganet_with_cuda or iganet_with_hip or iganet_with_xpu
  error('This LibTorch has a GPU backend. The Meson build links the CPU libraries only; configure with -Dcpuonly=true.')
endif

# ---------------------------------------------------------------------
# Third-party dependencies (WrapDB or subprojects/*.wrap fallbacks)
# ---------------------------------------------------------------------
pugixml_dep = dependency('pugixml', fallback: ['pugixml', 'pugixml_dep'])
json_dep = dependency('nlohmann_json', fallback: ['nlohmann_json', 'nlohmann_json_dep'])
boost_pp_dep = dependency(
  'boost_preprocessor',
  fallback: ['boost_preprocessor', 'boost_preprocessor_dep'],
)

iganet_deps = [torch_dep, pugixml_dep, json_dep, boost_pp_dep]
iganet_jit_flags_extra = []

if get_option('with_openmp')
  openmp_dep = dependency('openmp', language: 'cpp', required: true)
  iganet_deps += openmp_dep
  iganet_jit_flags_extra += ['-fopenmp']
endif

if get_option('with_mpi')
  iganet_deps += dependency('mpi', language: 'cpp', required: true)
endif

if get_option('with_gismo')
  iganet_deps += dependency('gismo', required: true)
endif

if get_option('with_matplot')
  iganet_deps += dependency(
    'Matplot++',
    method: 'cmake',
    modules: ['Matplot++::matplot'],
    required: true,
  )
endif

iganet_defs = ['-DJSON_HAS_RANGES=0']

# ---------------------------------------------------------------------
# Just-in-time compiler configuration (utils/jit.hpp)
# ---------------------------------------------------------------------
_jit_parts = ['-std=' + get_option('cpp_std'), '-fPIC', torch_abi]
# buildtype plain has optimization 'plain', which is no -O level
if get_option('optimization') != 'plain'
  _jit_parts += '-O' + get_option('optimization')
endif
_jit_parts += torch_def_args
_jit_parts += iganet_defs
_jit_parts += ['-I' + meson.current_source_dir() / 'include']
_jit_parts += ['-I' + meson.current_build_dir() / 'include']
torch_include_args = []
foreach inc : torch_includes
  torch_include_args += '-I' + inc
endforeach
_jit_parts += torch_include_args
# The just-in-time compiled code includes pugixml, nlohmann_json and
# Boost.Preprocessor headers too. A system dependency names its include
# directory through pkg-config; a subproject fallback has its sources in the
# directory its wrap names (subprojects/*.wrap, key `directory`).
_jit_dep_dirs = [
  [pugixml_dep, 'pugixml-1.15' / 'src'],
  [json_dep, 'nlohmann_json-3.12.0' / 'include'],
  [boost_pp_dep, 'preprocessor-boost-1.88.0' / 'include'],
]
foreach d : _jit_dep_dirs
  if d[0].type_name() == 'internal'
    # Unpacked below this project's subprojects/ or, when the wrap is
    # promoted, below the top project's; the subproject ran already, so the
    # directory exists in one of them.
    _found = false
    foreach _root : [meson.current_source_dir(), meson.global_source_root()]
      _dir = _root / 'subprojects' / d[1]
      if not _found and fs.is_dir(_dir)
        _jit_parts += '-I' + _dir
        _found = true
      endif
    endforeach
    if not _found
      error('JIT include directory of a subproject not found: subprojects/' + d[1])
    endif
  elif d[0].type_name() == 'pkgconfig'
    _inc = d[0].get_variable(pkgconfig: 'includedir', default_value: '')
    if _inc != ''
      _jit_parts += '-I' + _inc
    endif
  endif
endforeach
_jit_parts += iganet_jit_flags_extra
_jit_parts += [
  '-L' + torch_libdir,
  '-Wl,-rpath,' + torch_libdir,
  '-ltorch',
  '-ltorch_cpu',
  '-lc10',
]
if get_option('jit_extra_flags') != ''
  _jit_parts += get_option('jit_extra_flags')
endif
iganet_jit_flags = ' '.join(_jit_parts)
iganet_jit_compiler = ' '.join(cpp.cmd_array())

_py = find_program('python3', 'python')
iganet_jit_fingerprint = run_command(
  _py,
  '-c',
  'import hashlib, pathlib, sys; r = pathlib.Path(sys.argv[1]); h = hashlib.sha256(); [h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode()) for p in sorted(r.rglob("*")) if p.is_file() and p.suffix in (".h", ".hpp", ".in")]; print(h.hexdigest())',
  meson.current_source_dir() / 'include' / 'iganet',
  check: true,
).stdout().strip()

subdir('include/iganet/core')
subdir('include/iganet/utils')

# ---------------------------------------------------------------------
# The dependency consumers use
# ---------------------------------------------------------------------
iganet_inc = include_directories('include')

iganet_dep = declare_dependency(
  include_directories: iganet_inc,
  compile_args: iganet_defs,
  dependencies: iganet_deps,
  version: meson.project_version(),
  variables: {'torch_libdir': torch_libdir},
)
meson.override_dependency('iganet', iganet_dep)
meson.override_dependency('iganet-core', iganet_dep)

install_subdir(
  'include/iganet',
  install_dir: get_option('includedir'),
  exclude_files: [
    'core/config.hpp.in',
    'core/meson.build',
    'utils/jit.hpp.in',
    'utils/meson.build',
  ],
)

pkg = import('pkgconfig')
pkg.generate(
  name: 'iganet',
  description: 'Isogeometric analysis networks (header-only core)',
  version: meson.project_version(),
  url: 'https://github.com/iganets/iganet',
  requires: ['pugixml', 'nlohmann_json'],
  variables: ['torch_libdir=' + torch_libdir],
  extra_cflags: [torch_abi, '-DJSON_HAS_RANGES=0'] + torch_def_args + torch_include_args,
  libraries: ['-L' + torch_libdir, '-Wl,-rpath,' + torch_libdir, '-ltorch', '-ltorch_cpu', '-lc10'],
)
