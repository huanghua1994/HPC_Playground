EXES    = msg-channel.exe
CPPSRCS = $(wildcard *.cpp)
OBJS    = $(CPPSRCS:.cpp=.cpp.o)

LIBS += -lpthread

all: $(EXES)

msg-channel.exe: $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm $(OBJS) $(EXES)