import os
import numpy
pid = os.getpid();
ret = os.system('lsof -p ' + str(pid) + ' | grep -E "libmkl_rt|libopenblas" > /tmp/numpy_load.txt')
file1 = open("/tmp/numpy_load.txt", "r")
lsof_line = file1.readline()
file1.close()
for s1 in lsof_line.split(" "):
    substr = s1.strip()
    if os.path.isfile(substr):
        lib_path = substr
        break

lib_dir = os.path.dirname(lib_path)
lib_basename_noext = os.path.splitext(os.path.basename(lib_path))[0]
lib_name = lib_basename_noext[3:]
print('-L ' + lib_dir)
print('-l' + lib_name)
