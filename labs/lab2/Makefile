# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -O2

# OpenCV include and library flags (using pkg-config)
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Source file
SRC = main.c

# Output binary
TARGET = main

# Build the application
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(OPENCV_FLAGS)

# Clean the build
clean:
	rm -f $(TARGET)
