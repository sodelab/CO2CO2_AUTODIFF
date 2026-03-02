# ----------------------------------------------- #
# ** Makefile for mbCO2CO2 routine with autodiff **
# ----------------------------------------------- #

2BRG = CO2CO2
2B = mbCO2CO2

CXX = c++
CXXFLAGS = -O0 -fPIC -std=c++17

# Detect OS
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -dynamiclib
    # Only build universal binary if UNIVERSAL=1 is set (for local builds)
    ifeq ($(UNIVERSAL),1)
        CXXFLAGS += -arch x86_64 -arch arm64
        LDFLAGS  += -arch x86_64 -arch arm64
    endif
else
    LDFLAGS += -shared
endif

# Include paths
INCLUDES = -I. -I/usr/local/include -I/usr/local/include/eigen3

CONDA_PREFIX ?= $(shell echo $$CONDA_PREFIX)
ifneq ($(CONDA_PREFIX),)
    INCLUDES += -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/eigen3
endif

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
