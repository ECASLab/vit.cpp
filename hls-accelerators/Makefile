# Setting relevant variables
ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TARGET := hw
PLATFORM := $(ROOT_DIR)/../kv260_custom.xpfm


HLS_FILE_NAME = Operations_Vit
# Setting input file
HLS_FILES := $(wildcard *.cpp)
HLS_FILES_NAMES := $(HLS_FILES:.cpp=)


# Setting directories
TEMP_DIR := ./tmp.$(TARGET)
BUILD_DIR := ./build.$(TARGET)
PACKAGE_OUT = ./package.$(TARGET)
LINK_OUTPUT := $(BUILD_DIR)/$(HLS_FILE_NAME).link.xclbin
EMCONFIG_DIR = $(TEMP_DIR)

# Packaging
FPGA_BIN := $(PACKAGE_OUT)/$(HLS_FILE_NAME).bit.bin
XCL_BIN := $(PACKAGE_OUT)/$(HLS_FILE_NAME).xclbin
SHELL_JSON := $(PACKAGE_OUT)/shell.json
DTB_FILE := $(PACKAGE_OUT)/$(HLS_FILE_NAME).dtbo


# Output files
HLS_KERNEL_FILES := $(addprefix $(TEMP_DIR)/,$(HLS_FILES:.cpp=.xo))

# Setting V++ Flags
VPP_PFLAGS :=
VPP_LDFLAGS := --vivado.impl.strategies=Area_Explore
VPP_FLAGS += --save-temps --jobs 6

.PHONY: all clean cleanall fpgabin

all: build

.PHONY: build
build: check_platform emconfig $(LINK_OUTPUT) $(FPGA_BIN) $(XCL_BIN)

# Rules for creating the HW
check_platform:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

$(TEMP_DIR)/%.xo: %.cpp
	mkdir -p $(TEMP_DIR)
	v++ -c $(VPP_FLAGS) $(VPP_PFLAGS) -t $(TARGET) --platform $(PLATFORM) --hls.clock=300000000:$(<:.cpp=) -k $(<:.cpp=) --temp_dir $(TEMP_DIR) -I'$(<D)' -o '$@' '$<'

$(LINK_OUTPUT): $(HLS_KERNEL_FILES)
	mkdir -p $(BUILD_DIR)
	v++ -l $(VPP_FLAGS) $(VPP_LDFLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR) --freqhz=200000000:matmul_f16_1.ap_clk -o'$(LINK_OUTPUT)' $^

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

system.bit: $(LINK_OUTPUT)
	cp $(TEMP_DIR)/link/int/$@ .

bootgen.bif:
	echo "all:{system.bit}" > bootgen.bif

$(FPGA_BIN): system.bit bootgen.bif $(SHELL_JSON) $(DTB_FILE)
	mkdir -p $(PACKAGE_OUT)
	bootgen -w -arch zynqmp -process_bitstream bin -image bootgen.bif
	cp system.bit.bin $@

$(XCL_BIN): $(LINK_OUTPUT)
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

$(SHELL_JSON): $(ROOT_DIR)/../shell.json
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

$(DTB_FILE): $(ROOT_DIR)/../pl.dtbo
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

# Cleaning rules
clean:
	$(RM) -rf $(EXECUTABLE) *.xclbin/{*sw_emu*,*hw_emu*}
	$(RM) -rf profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	$(RM) -rf src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	$(RM) bootgen.bif system.bit system.bit.bin

cleanall: clean
	$(RM) -rf build_dir* sd_card*
	$(RM) -rf package.* tmp.*
	$(RM) -rf _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin
