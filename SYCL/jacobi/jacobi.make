EXES    = jacobi-2d-usm.exe
CPPSRCS = $(wildcard *.cpp)
OBJS    = $(CPPSRCS:.cpp=.cpp.o)

all: $(EXES)

.SECONDARY: $(OBJS)

%.exe: %.cpp.o
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXES)