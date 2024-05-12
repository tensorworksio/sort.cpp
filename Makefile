CXX = g++
CXXFLAGS = $(shell pkg-config --cflags dlib-1) -I/usr/local/include/opencv4 -Isrc -Wall -Wextra -g
LDFLAGS = $(shell pkg-config --libs dlib-1) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)

.PHONY: all clean main

all: main

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) main