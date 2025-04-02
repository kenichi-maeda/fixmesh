# PyMesh Installation Guide

## 0. Prerequisite

### Install CMake (if you don't have one)
```bash
sudo apt update
sudo apt install build-essential checkinstall zlib1g-dev libssl-dev -y
wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2.tar.gz
tar -zxvf cmake-3.23.2.tar.gz
cd cmake-3.23.2
./bootstrap
sudo make install
cmake --version
```

### Add Extra Swap File (Optional)
If your system has insufficient memory, the `cc1plus` process is likely to get killed. To avoid that, add a swap file.

#### 1. Check Current Memory Status
```bash
free -h
```
Example output (numbers will vary by machine):
```
total        used        free      shared  buff/cache   available
Mem:            16G         12G        2.0G         1G         2G        2.5G
Swap:           2G          1G         3G
```

#### 2. Add More Swap Space (Example: 7GB)
```bash
sudo fallocate -l 7G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Verify the New Swap Space
After adding, it should look like this (example):
```bash
free -h
```
```
total        used        free      shared  buff/cache   available
Mem:            16G         12G        2.0G         1G         2G        2.5G
Swap:           9G          1G         3G
```

## 1. Download the Source
```bash
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
export PYMESH_PATH=`pwd`
```

## 2. Install System Dependencies
```bash
sudo apt-get install \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
    python3-dev
```

Install Python dependencies:
```bash
pip install -r $PYMESH_PATH/python/requirements.txt
```

## 3. Build PyMesh with Setup Tools
```bash
./setup.py build
```

## 4. Install PyMesh
```bash
./setup.py install
```

---

## Errors and Solutions (As of Jan 2025)


### Error 1
#### Error
```
In file included from `/home/USER/PyMesh/third_party/draco/src/draco/core/hash_utils.cc:15`:
/home/USER/PyMesh/third_party/draco/src/draco/core/hash_utils.h:26:1: error: ‘size_t’ does not name a type
   26 | size_t HashCombine(T1 a, T2 b) {
      | ^~~~~~
/home/USER/PyMesh/third_party/draco/src/draco/core/hash_utils.h:20:1: note: ‘size_t’ is defined in header ‘<cstddef>’; did you forget to ‘#include <cstddef>’?
```

#### Solution
Navigate to the file `./third_party/draco/src/draco/core/hash_utils.h` and add the following line:
```cpp
#include <cstddef>
```
Save the file and retry building PyMesh.

### Error2
#### Error
```/home/USER/PyMesh/third_party/draco/src/draco/io/parser_utils.cc: In function ‘bool draco::parser::ParseFloat(draco::DecoderBuffer*, float*)’:
/home/USER/PyMesh/third_party/draco/src/draco/io/parser_utils.cc:113:16: error: ‘numeric_limits’ is not a member of ‘std’
  113 |       v = std::numeric_limits<double>::infinity();
      |                ^~~~~~~~~~~~~~
/home/USER/PyMesh/third_party/draco/src/draco/io/parser_utils.cc:113:31: error: expected primary-expression before ‘double’
  113 |       v = std::numeric_limits<double>::infinity();
      |                               ^~~~~~
```

#### Solution
Navigate to the file `./third_party/draco/src/draco/io/parser_utils.cc` and add the following line:
```cpp
#include <limits>
```
Save the file and retry building PyMesh.


### Error3
#### Error
When you see a bunch of:
```
/usr/bin/ld: lib/libmmg2d.a(boulep.c.o):(.bss+0x30): multiple definition of `MMG5_bezierCP'; CMakeFiles/mmg2d.dir/src/mmg2d/mmg2d.c.o:(.bss+0xba8): first defined here
/usr/bin/ld: lib/libmmg2d.a(boulep.c.o):(.bss+0x38): multiple definition of `MMG5_chkmsh'; CMakeFiles/mmg2d.dir/src/mmg2d/mmg2d.c.o:(.bss+0xbb0): first defined here
/usr/bin/ld: lib/libmmg2d.a(cenrad_2d.c.o):(.bss+0x0): multiple definition of `MMG2D_defsiz'; CMakeFiles/mmg2d.dir/src/mmg2d/mmg2d.c.o:(.bss+0xb40): first defined here
/usr/bin/ld: lib/libmmg2d.a(cenrad_2d.c.o):(.bss+0x8): multiple definition of `MMG2D_gradsizreq'; CMakeFiles/mmg2d.dir/src/mmg2d/mmg2d.c.o:(.bss+0xb48): first defined here
/usr/bin/ld: lib/libmmg2d.a(cenrad_2d.c.o):(.bss+0x10): multiple definition of `MMG2D_gradsiz'; CMakeFiles/mmg2d.dir/src/mmg2d/mmg2d.c.o:(.bss+0xb50): first defined here
/usr/bin/ld: lib/libmmg2d.a(cenrad_2d.c.o):(.bss+0x18): multiple definition of `MMG2D_intmet'; CMakeFiles/mmg2d.dir/src/mmg2d/mmg2d.c.o:(.bss+0xb58): first defined here

```

#### Solution
Open `./third_party/mmg/CMakeLists.txt`, then add `SET(CMAKE_C_FLAGS "-fcommon ${CMAKE_C_FLAGS}")` right below `PROJECT(mmg)`.

```
PROJECT(mmg) 

SET(CMAKE_C_FLAGS "-fcommon ${CMAKE_C_FLAGS}")
``` 
Save the file and retry building PyMesh.

### Error4
#### Error
When you see this kind of error:
```
/home/USER/PyMesh/tools/CGAL/SelfIntersection.cpp: In member function ‘void PyMesh::SelfIntersection::detect_self_intersection()’:
/home/USER/PyMesh/tools/CGAL/SelfIntersection.cpp:113:23: error: ‘_1’ was not declared in this scope
  113 |                 this, _1, _2);

```

#### Solution
Replace `<boostrap/bind.h>` with
```
#include <boost/bind/bind.hpp>
using namespace boost::placeholders;
```

##### Which file?
For `SelfIntersection.cpp`, modify `/home/USER/PyMesh/tools/CGAL/SelfIntersection.cpp`.<br><br>

For others do the following. <br>
Suppose `Straight_skeleton_2_impl.h` is problematic. 
##### 1. Run `find`
```
find /home/USER/PyMesh/ -name "Straight_skeleton_builder_2_impl.h"
```
Then you will see something like:
```
/home/USER/PyMesh/python/pymesh/third_party/include/CGAL/Straight_skeleton_2/Straight_skeleton_builder_2_impl.h
/home/USER/PyMesh/third_party/cgal/Straight_skeleton_2/include/CGAL/Straight_skeleton_2/Straight_skeleton_builder_2_impl.h
```

##### 2. Modify .h in `PyMesh/third_party`
In this particular example, navigate to
`/home/USER/PyMesh/third_party/cgal/Straight_skeleton_2/include/CGAL/Straight_skeleton_2/Straight_skeleton_builder_2_impl.h`.

You might need to change a couple of .h files as you move on ( `convexity_check_2_impl.h`, `AABB_traits.h`, `Triangulation_3.h`).


### Error5
#### Error
```
c++: fatal error: Killed signal terminated program cc1plus
compilation terminated.
```

#### Solution
Increase the size of swap as mentioned above. 9GB would be enough for this installation.

### Error6
#### Error
```
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h: In function ‘std::string pybind11::detail::error_string()’:
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:442:36: error: invalid use of incomplete type ‘PyFrameObject’ {aka ‘struct _frame’}
  442 |                 "  " + handle(frame->f_code->co_filename).cast<std::string>() +
      |                                    ^~
In file included from /home/USER/anaconda3/include/python3.11/Python.h:42,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/detail/common.h:112,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/pytypes.h:12,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:13,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/attr.h:13,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/pybind11.h:44,
                 from /home/USER/PyMesh/python/module.cpp:3:
/home/USER/anaconda3/include/python3.11/pytypedefs.h:22:16: note: forward declaration of ‘PyFrameObject’ {aka ‘struct _frame’}
   22 | typedef struct _frame PyFrameObject;
      |                ^~~~~~
In file included from /home/USER/PyMesh/third_party/pybind11/include/pybind11/attr.h:13,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/pybind11.h:44,
                 from /home/USER/PyMesh/python/module.cpp:3:
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:442:75: error: expected primary-expression before ‘>’ token
  442 |                 "  " + handle(frame->f_code->co_filename).cast<std::string>() +
      |                                                                           ^
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:442:77: error: expected primary-expression before ‘)’ token
  442 |                 "  " + handle(frame->f_code->co_filename).cast<std::string>() +
      |                                                                             ^
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:444:29: error: invalid use of incomplete type ‘PyFrameObject’ {aka ‘struct _frame’}
  444 |                 handle(frame->f_code->co_name).cast<std::string>() + "\n";
      |                             ^~
In file included from /home/USER/anaconda3/include/python3.11/Python.h:42,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/detail/common.h:112,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/pytypes.h:12,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:13,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/attr.h:13,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/pybind11.h:44,
                 from /home/USER/PyMesh/python/module.cpp:3:
/home/USER/anaconda3/include/python3.11/pytypedefs.h:22:16: note: forward declaration of ‘PyFrameObject’ {aka ‘struct _frame’}
   22 | typedef struct _frame PyFrameObject;
      |                ^~~~~~
In file included from /home/USER/PyMesh/third_party/pybind11/include/pybind11/attr.h:13,
                 from /home/USER/PyMesh/third_party/pybind11/include/pybind11/pybind11.h:44,
                 from /home/USER/PyMesh/python/module.cpp:3:
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:444:64: error: expected primary-expression before ‘>’ token
  444 |                 handle(frame->f_code->co_name).cast<std::string>() + "\n";
      |                                                                ^
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:444:66: error: expected primary-expression before ‘)’ token
  444 |                 handle(frame->f_code->co_name).cast<std::string>() + "\n";
      |                                                                  ^
/home/USER/PyMesh/third_party/pybind11/include/pybind11/cast.h:445:26: error: invalid use of incomplete type ‘PyFrameObject’ {aka ‘struct _frame’}
  445 |             frame = frame->f_back;
      |                          ^~
```

#### Solution
If feasible, try using Python 3.10 or an earlier version to avoid compatibility issues.
```
conda create -n pymesh_env python=3.10
conda activate pymesh_env
```