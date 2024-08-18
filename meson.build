project('sort.cpp', 'cpp')

# Include directory
inc_dir = include_directories('include')

# Dependencies
dlib_dep = dependency('dlib-1', method : 'pkg-config')
boost_dep = dependency('boost', modules : ['filesystem', 'program_options', 'json'])
opencv_dep = dependency('opencv4', modules : ['core', 'highgui', 'imgproc', 'imgcodecs', 'video', 'videoio'])

# Source files
src_files = files(
  'src/tracker.cpp',
  'src/sort.cpp',
  'src/main.cpp'
)

# Build shared library
library('sort', src_files,
  dependencies : [dlib_dep, boost_dep, opencv_dep],
  include_directories : inc_dir
)

# Main executable
executable('sort', src_files,
  dependencies : [dlib_dep, boost_dep, opencv_dep],
  include_directories : inc_dir,
  install : true
)