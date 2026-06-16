# Cross-compile the board wake-word app (kws) for the Luckfox Pico
# (RV1103/RV1106, ARM uClibc target).
#
# TOOLCHAIN / RKNN_RT are read via ?= so the dev container can inject them as
# env vars (see .devcontainer/devcontainer.json) with no edits here. For a local
# build, either export them, pass them on the command line
# (`make TOOLCHAIN=... RKNN_RT=...`), or edit the two defaults below.

TOOLCHAIN ?= $(HOME)/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
RKNN_RT   ?= $(HOME)/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api

CC := $(TOOLCHAIN)gcc

# librknnmrt.so (the RV1103/RV1106 "mini" runtime) + rknn_api.h live here.
RKNN_INC := $(RKNN_RT)/include
RKNN_LIB := $(RKNN_RT)/armhf-uclibc

CFLAGS  := -O2 -Wall -Wextra -I$(RKNN_INC)
LDFLAGS := -L$(RKNN_LIB) -lrknnmrt -lm -lpthread -ldl

TARGET := kws
SRC    := kws.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
