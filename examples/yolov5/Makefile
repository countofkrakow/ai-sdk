TARGET = yolov5

ifdef AI_SDK_PLATFORM
include ../../machinfo/$(AI_SDK_PLATFORM)/config.mk
endif

# Normalize trailing slash so INSTALL_PREFIX=./ does not expand to .//...
INSTALL_PREFIX_NORM := $(if $(filter /,$(INSTALL_PREFIX)),/,$(patsubst %/,%,$(INSTALL_PREFIX)))

INCLUDE += -I$(INSTALL_PREFIX)/usr/include -I$(STAGING_DIR)/usr/include/opencv4 -I../libawnn_viplite -I../libawutils -I../../
LIBS    += -L $(INSTALL_PREFIX)/usr/lib -L $(INSTALL_PREFIX)/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,$(INSTALL_PREFIX)/usr/lib -Wl,-rpath-link,$(INSTALL_PREFIX)/usr/lib/aarch64-linux-gnu -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_highgui -ljpeg -lperiphery
SRCS    += main.c app_config.cpp app_init.cpp app_loop.cpp app_shutdown.cpp debug_trace.cpp play_engine.cpp tracking_utils.cpp servo_control.cpp play_algorithms.cpp play_tuning.cpp play_director_policy.cpp yolov5_post_process.cpp yolov5_pre_process.cpp ../libawnn_viplite/awnn_lib.c ../libawnn_viplite/awnn_quantize.c ../libawutils/image_utils.c

# Prefer the system OpenCL SONAME to avoid accidentally picking an incompatible
# vendor libOpenCL from other -L search paths (can cause OPENCL_1.x unresolved
# symbols from transitive libs such as libavutil via OpenCV videoio).
OPENCL_LIB ?= $(INSTALL_PREFIX)/usr/lib/aarch64-linux-gnu/libOpenCL.so.1

CFLAGS += -g -O0 -DDEBUG -D_DEBUG

ifeq ($(SAVE_OUTPUT_TXT_FILE),1)
  CFLAGS += -DSAVE_OUTPUT_TXT_FILE
endif

ifeq ("$(NPU_SW_VERSION)", "v2.0")
  INCLUDE += -I../../viplite-tina/lib/aarch64-none-linux-gnu/v2.0/inc
  LIBS_VIP += -L../../viplite-tina/lib/aarch64-none-linux-gnu/v2.0/ -lNBGlinker -lVIPhal
  MYCFLAGS += -DNPU_SW_VERSION=2
else
  INCLUDE += -I../../viplite-tina/lib/aarch64-none-linux-gnu/v1.13/inc
  LIBS_VIP += -L../../viplite-tina/lib/aarch64-none-linux-gnu/v1.13/ -lVIPlite -lVIPuser
  MYCFLAGS += -DNPU_SW_VERSION=1
endif


$(TARGET): $(SRCS)
	$(CXX) $(CFLAGS) -g -o $@ $^ $(INCLUDE) $(LIBS) $(LIBS_VIP) -Wl,--no-as-needed $(OPENCL_LIB) -Wl,--as-needed -lm -pthread

all:$(TARGET) install

install: $(TARGET)
	-@mkdir -p $(INSTALL_PREFIX_NORM)/usr/bin
	@install -m 0755 $(TARGET) $(INSTALL_PREFIX_NORM)/usr/bin/$(TARGET)
	-@mkdir -p $(INSTALL_PREFIX_NORM)/etc/npu/yolov5
	-@mkdir -p $(INSTALL_PREFIX_NORM)/etc/npu/yolov5/model
	@cp -f model/$(NPU_VERSION)/* $(INSTALL_PREFIX_NORM)/etc/npu/yolov5/model
	-@mkdir -p $(INSTALL_PREFIX_NORM)/etc/npu/yolov5/input_data
	@cp -f input_data/* $(INSTALL_PREFIX_NORM)/etc/npu/yolov5/input_data

clean:
	rm -rf $(TARGET) *.o *~
