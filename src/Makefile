# ----------------------------------------------- #
# ** Makefile for mbCO2CO2 routine with autodiff **
# ----------------------------------------------- #

2BRG = CO2CO2
2B = mbCO2CO2

CXX = c++
CXXFLAGS = -O3 -fPIC -std=c++17

# Detect OS
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# macOS: build universal binary if possible
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -arch x86_64 -arch arm64
    LDFLAGS  += -dynamiclib -arch x86_64 -arch arm64
else
    LDFLAGS  += -shared
endif

# Include paths (adjust if needed)
INCLUDES = -I. -I/usr/local/include -I/usr/local/include/eigen3

SRCS = mbCO2CO2.cpp poly-1b.cpp poly-2b.cpp x1b.cpp x2b.cpp sapt-s.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = libCO2CO2.so

all: $(TARGET)
	cp $(TARGET) ../co2_potential/$(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
