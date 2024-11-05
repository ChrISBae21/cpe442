# Compiler
# CC = g++
CC = aarch64-linux-gnu-g++

# Compiler flags
CFLAGS = -O0

LFLAGS = -lpthread

# OpenCV include and library flags (using pkg-config)
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Source file
# SRC = main.c
SRC = lab5.c

# Output binary
# TARGET = main
TARGET = lab5

# Build the application
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(OPENCV_FLAGS) $(LFLAGS)

# Clean the build
clean:
	rm -f $(TARGET)
