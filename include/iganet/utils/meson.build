jit_conf = configuration_data()
jit_conf.set('JIT_CXX_COMPILER', iganet_jit_compiler)
jit_conf.set('JIT_CONFIGURED_FLAGS', iganet_jit_flags)
jit_conf.set('JIT_ALTERNATE_FLAGS', '')
jit_conf.set('JIT_CXX_OUTPUT_FLAG', '-o ')
jit_conf.set('IGANET_VERSION', meson.project_version())
jit_conf.set('JIT_CACHE_FINGERPRINT', iganet_jit_fingerprint)
jit_conf.set('IGANET_INSTALL_INCLUDEDIR', get_option('includedir') / 'iganet')
jit_conf.set('JIT_INSTALL_FALLBACK_PREFIX', get_option('prefix'))
# <prefix>/<includedir>/iganet/utils/jit.hpp: two directories for the file and
# utils, plus one per component of <includedir>/iganet.
jit_conf.set(
  'JIT_INSTALL_PREFIX_DEPTH',
  (get_option('includedir') / 'iganet').split('/').length() + 2,
)

configure_file(
  input: 'jit.hpp.in',
  output: 'jit.hpp',
  configuration: jit_conf,
  format: 'cmake@',
  install: true,
  install_dir: get_option('includedir') / 'iganet' / 'utils',
)
