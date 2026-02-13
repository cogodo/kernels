NVCC := nvcc
CUDA_ARCH := sm_100
CFLAGS := -O3 -arch=$(CUDA_ARCH)

BUILD_DIR := build
TARGET := $(BUILD_DIR)/libgemm.so

SRCS := $(wildcard *.cu)
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -shared -o $@ $^

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -Xcompiler -fPIC -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Syntax check only (no GPU required)
check: $(SRCS)
	$(NVCC) $(CFLAGS) -Xcompiler -fPIC -fsyntax-only $^

clean:
	rm -rf $(BUILD_DIR)
