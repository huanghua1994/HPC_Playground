EXES    = atomic-add.exe
CPPSRCS = $(wildcard *.cpp)
OBJS    = $(CPPSRCS:.cpp=.cpp.o)

all: $(EXES)

%.exe: %.cpp.o
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXES)