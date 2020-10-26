EXES    = stream-triad-accessor.exe stream-triad-usm.exe
CPPSRCS = $(wildcard *.cpp)
OBJS    = $(CPPSRCS:.cpp=.cpp.o)

all: $(EXES)

%.exe: %.cpp.o
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.cpp.o: %.cpp
	$(CXX) $(CFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXES)