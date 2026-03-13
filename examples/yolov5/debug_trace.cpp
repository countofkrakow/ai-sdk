#include "debug_trace.h"

#include <stdarg.h>
#include <string.h>
#include <time.h>

static const char *level_name(DebugLogLevel level) {
    switch (level) {
        case DEBUG_LOG_ERROR: return "ERROR";
        case DEBUG_LOG_WARN: return "WARN";
        case DEBUG_LOG_INFO: return "INFO";
        case DEBUG_LOG_TRACE: return "TRACE";
        default: return "UNK";
    }
}

void debug_trace_init(DebugTrace *trace, DebugLogLevel level, const char *csv_path_or_null) {
    if (trace == NULL) return;
    trace->level = level;
    trace->csv_file = NULL;
    if (csv_path_or_null != NULL && csv_path_or_null[0] != '\0') {
        trace->csv_file = fopen(csv_path_or_null, "w");
        if (trace->csv_file != NULL) {
            fprintf(trace->csv_file,
                    "frame_seq,timestamp_sec,has_cat,confidence,algorithm,engagement,intensity,target_x,target_y,pan_deg,tilt_deg,laser_on\n");
            fflush(trace->csv_file);
        }
    }
}

void debug_trace_close(DebugTrace *trace) {
    if (trace == NULL) return;
    if (trace->csv_file != NULL) {
        fclose(trace->csv_file);
        trace->csv_file = NULL;
    }
}

void debug_trace_log(const DebugTrace *trace, DebugLogLevel level, const char *category, const char *fmt, ...) {
    if (trace == NULL || level > trace->level) return;
    const char *cat = (category != NULL) ? category : "GEN";

    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[%s][%s] ", level_name(level), cat);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

void debug_trace_write_row(DebugTrace *trace, const FrameTraceRow *row) {
    if (trace == NULL || row == NULL || trace->csv_file == NULL) return;
    fprintf(trace->csv_file,
            "%lu,%.6f,%d,%.4f,%s,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%d\n",
            row->frame_seq,
            row->timestamp_sec,
            row->has_cat,
            row->confidence,
            (row->algorithm_name != NULL) ? row->algorithm_name : "none",
            row->engagement_score,
            row->intensity_scale,
            row->target_x,
            row->target_y,
            row->pan_deg,
            row->tilt_deg,
            row->laser_on);
    fflush(trace->csv_file);
}
