#include <signal.h>
#include <stdio.h>
#include <string.h>

#include "app_config.h"
#include "app_init.h"
#include "app_loop.h"
#include "app_shutdown.h"

static volatile sig_atomic_t g_sigint_received = 0;

static void handle_sigint(int signum) {
    (void)signum;
    g_sigint_received = 1;
}

int main(int argc, char **argv) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

    AppConfig cfg;
    if (app_parse_config(argc, argv, &cfg) != 0) {
        app_print_usage(argv[0]);
        return -1;
    }

    AppRuntime rt;
    if (app_runtime_init(&cfg, &rt) != 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        return -1;
    }

    const int rc = app_run_loop(&rt, &g_sigint_received);
    app_runtime_shutdown(&rt, g_sigint_received ? 1 : 0);
    return (rc == 0) ? 0 : -1;
}
