project(
  'boost_preprocessor',
  'cpp',
  version: '1.88.0',
  meson_version: '>=1.3.0',
  license: 'BSL-1.0',
)

boost_preprocessor_dep = declare_dependency(
  include_directories: include_directories('include'),
  version: meson.project_version(),
)
meson.override_dependency('boost_preprocessor', boost_preprocessor_dep)

install_subdir('include/boost', install_dir: get_option('includedir'))
