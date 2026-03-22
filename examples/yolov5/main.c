#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
#include <execinfo.h>
#endif

#include "app_config.h"
#include "app_init.h"
#include "app_loop.h"
#include "app_shutdown.h"

static volatile sig_atomic_t g_sigint_received = 0;

static void handle_sigint(int signum) {
    (void)signum;
    g_sigint_received = 1;
}

static void handle_fatal_signal(int signum, siginfo_t *info, void *ucontext) {
    (void)ucontext;
    const void *fault_addr = (info != NULL) ? info->si_addr : NULL;
    fprintf(stderr, "\n[FATAL] Caught signal %d", signum);
    if (fault_addr != NULL) {
        fprintf(stderr, " at address %p", fault_addr);
    }
    fprintf(stderr, "\n");
#if defined(__linux__)
    void *frames[64];
    int frame_count = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
    if (frame_count > 0) {
        fprintf(stderr, "[FATAL] Backtrace (%d frames):\n", frame_count);
        backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    }
#endif
    _Exit(128 + signum);
}

static void install_fatal_signal_handlers(void) {
    const int fatal_signals[] = {SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE};
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = handle_fatal_signal;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);

    for (unsigned int i = 0; i < sizeof(fatal_signals) / sizeof(fatal_signals[0]); ++i) {
        sigaction(fatal_signals[i], &sa, NULL);
    }
}

int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IOLBF, 0);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    install_fatal_signal_handlers();
    fprintf(stderr, "[INFO][MAIN] Starting YOLOv5 app with %d argument(s)\n", argc);

    AppConfig cfg;
    if (app_parse_config(argc, argv, &cfg) != 0) {
        fprintf(stderr, "[ERROR][MAIN] Failed to parse CLI arguments\n");
        app_print_usage(argv[0]);
        return -1;
    }
    fprintf(stderr, "[INFO][MAIN] Parsed config: model=%s camera=%s dry_run=%d test_mode=%d replay=%s brightness=%u\n",
            cfg.nbg_path,
            cfg.camera_device.c_str(),
            cfg.dry_run ? 1 : 0,
            cfg.test_mode ? 1 : 0,
            cfg.replay_frames_dir.empty() ? "<none>" : cfg.replay_frames_dir.c_str(),
            cfg.laser_brightness_percent);

    AppRuntime rt;
    if (app_runtime_init(&cfg, &rt) != 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        app_runtime_shutdown(&rt, 1);
        return -1;
    }

    fprintf(stderr, "[INFO][MAIN] Entering app_run_loop\n");
    const int rc = app_run_loop(&rt, &g_sigint_received);
    fprintf(stderr, "[INFO][MAIN] app_run_loop returned rc=%d sigint=%d\n", rc, g_sigint_received ? 1 : 0);
    app_runtime_shutdown(&rt, g_sigint_received ? 1 : 0);
    fprintf(stderr, "[INFO][MAIN] Shutdown complete\n");
    return (rc == 0) ? 0 : -1;
}
