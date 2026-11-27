iganet_conf = configuration_data()
iganet_conf.set('iganet_VERSION', meson.project_version())
iganet_conf.set('iganet_VERSION_MAJOR', iganet_version_parts[0])
iganet_conf.set('iganet_VERSION_MINOR', iganet_version_parts[1])
iganet_conf.set('iganet_VERSION_PATCH', iganet_version_parts[2])
iganet_conf.set10('IGANET_WITH_GISMO', get_option('with_gismo'))
iganet_conf.set10('IGANET_WITH_MATPLOT', get_option('with_matplot'))
iganet_conf.set10('IGANET_WITH_OPENMP', get_option('with_openmp'))
iganet_conf.set10('IGANET_WITH_MPI', get_option('with_mpi'))
iganet_conf.set10('IGANET_WITH_CUDA', iganet_with_cuda)
iganet_conf.set10('IGANET_WITH_HIP', iganet_with_hip)
iganet_conf.set10('IGANET_WITH_XPU', iganet_with_xpu)

# #cmakedefine X expands to "#define X" for 1 and "/* #undef X */" for 0.
configure_file(
  input: 'config.hpp.in',
  output: 'config.hpp',
  configuration: iganet_conf,
  format: 'cmake@',
  install: true,
  install_dir: get_option('includedir') / 'iganet' / 'core',
)
