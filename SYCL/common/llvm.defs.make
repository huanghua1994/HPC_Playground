CXX         = clang++
CFLAGS      = -O3 -g -fPIC
CXXFLAGS    = $(CFLAGS) -fsycl -fsycl-unnamed-lambda -std=c++17
LIBS        = -lsycl