CXX = g++
UNSAFECXXFLAGS = -Wall -Werror -O2 -std=c++14
SAFECXXFLAGS = $(UNSAFECXXFLAGS) -fsanitize=address
CXXFLAGS = $(UNSAFECXXFLAGS)

SOURCES = $(wildcard *.cpp)
OBJECTS = $(subst .cpp,.o,$(SOURCES)) tests main

default: all
all: $(OBJECTS)

tests: tests.cpp NeuralNet.o Matrix.o MatrixExceptions.o
	$(CXX) $(CXXFLAGS) -o tests tests.cpp NeuralNet.o Matrix.o MatrixExceptions.o

main: main.cpp NeuralNet.o Matrix.o MatrixExceptions.o
	$(CXX) $(CXXFLAGS) -o main main.cpp NeuralNet.o Matrix.o MatrixExceptions.o

NeuralNet.o: NeuralNet.cpp NeuralNet.hpp Matrix.hpp
	$(CXX) $(CXXFLAGS) -o NeuralNet.o -c NeuralNet.cpp

Matrix.o: Matrix.cpp Matrix.hpp MatrixExceptions.hpp
	$(CXX) $(CXXFLAGS) -o Matrix.o -c Matrix.cpp

MatrixExceptions.o: MatrixExceptions.cpp MatrixExceptions.hpp
	$(CXX) $(CXXFLAGS) -o MatrixExceptions.o -c MatrixExceptions.cpp
clean:
	rm -f a.out *.gch $(OBJECTS)
