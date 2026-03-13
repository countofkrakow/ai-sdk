#ifndef YOLOV5_DEBUG_TRACE_H_
#define YOLOV5_DEBUG_TRACE_H_

#include <stdio.h>

enum DebugLogLevel {
    DEBUG_LOG_ERROR = 0,
    DEBUG_LOG_WARN = 1,
    DEBUG_LOG_INFO = 2,
    DEBUG_LOG_TRACE = 3,
};

struct DebugTrace {
    DebugLogLevel level;
    FILE *csv_file;
};

struct FrameTraceRow {
    unsigned long frame_seq;
    double timestamp_sec;
    int has_cat;
    float confidence;
    const char *algorithm_name;
    float engagement_score;
    float intensity_scale;
    float target_x;
    float target_y;
    float pan_deg;
    float tilt_deg;
    int laser_on;
};

void debug_trace_init(DebugTrace *trace, DebugLogLevel level, const char *csv_path_or_null);
void debug_trace_close(DebugTrace *trace);
void debug_trace_log(const DebugTrace *trace, DebugLogLevel level, const char *category, const char *fmt, ...);
void debug_trace_write_row(DebugTrace *trace, const FrameTraceRow *row);

#endif
