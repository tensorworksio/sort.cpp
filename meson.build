project('sort.cpp', 'cpp')

dlib_dep = dependency('dlib-1', method : 'pkg-config')
boost_dep = dependency('boost', modules : ['filesystem', 'program_options', 'json'])
opencv_dep = dependency('opencv4', modules : ['core', 'highgui', 'imgproc', 'imgcodecs', 'video', 'videoio'])

cxx_flags = ['-Isrc', '-Wall', '-Wextra', '-g', '-std=c++17']

inc_dir = include_directories('src')
src_dir = meson.current_source_dir() / 'src'
src_files = run_command('find', src_dir, '-name', '*.cpp', check : true).stdout().strip().split()

main_file = meson.current_source_dir() / 'main.cpp'
src_files += main_file

executable('main', src_files,
  dependencies : [dlib_dep, boost_dep, opencv_dep],
  include_directories : inc_dir,
  cpp_args : cxx_flags,
  install : true
)
