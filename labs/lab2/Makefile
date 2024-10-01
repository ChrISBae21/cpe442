# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O2

# Include directories for OpenCV
INCLUDE_DIRS = `pkg-config --cflags opencv4`

# Libraries for OpenCV
LIBS = `pkg-config --libs opencv4`

# Source files
SRCS = main.c

# Output binary
TARGET = main

# Build the application
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -o $(TARGET) $(SRCS) $(LIBS)

# Clean the build
clean:
	rm -f $(TARGET)
