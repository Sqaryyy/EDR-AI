#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <unistd.h>
#include <stdio.h>

const char *syscall_name(long num) {
    switch(num) {
        case 0:   return "read";
        case 1:   return "write";
        case 2:   return "open";
        case 3:   return "close";
        case 4:   return "stat";
        case 9:   return "mmap";
        case 11:  return "munmap";
        case 41:  return "socket";
        case 42:  return "connect";
        case 56:  return "clone";
        case 57:  return "fork";
        case 59:  return "execve";
        case 82:  return "rename";
        case 257: return "openat";
        case 260: return "openat2";
        case 316: return "renameat2";
        default:  return "other";
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: ./tracer <program> [args]\n");
        return 1;
    }

    pid_t child = fork();

    if (child == 0) {
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        execvp(argv[1], &argv[1]);
    } else {
    int status;
    struct user_regs_struct regs;

    // first wait for child to stop after PTRACE_TRACEME
    wait(&status);

    // tell ptrace to follow forks, clones, and execs automatically
    ptrace(PTRACE_SETOPTIONS, child, NULL,
           PTRACE_O_TRACEFORK | PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXEC);

    // start tracing
    ptrace(PTRACE_SYSCALL, child, NULL, NULL);

    while (1) {
        pid_t traced = waitpid(-1, &status, 0);
        if (traced == -1) break;

        if (WIFEXITED(status) || WIFSIGNALED(status)) {
            // a child exited, keep looping in case others are still running
            continue;
        }

        if (WIFSTOPPED(status)) {
            int sig = WSTOPSIG(status);
            if (sig == SIGTRAP || sig == (SIGTRAP | 0x80)) {
                if (ptrace(PTRACE_GETREGS, traced, NULL, &regs) == 0) {
                    printf("%d,%lld,%s\n", traced, regs.orig_rax, syscall_name(regs.orig_rax));
                }
            }
            ptrace(PTRACE_SYSCALL, traced, NULL, NULL);
        }
    }
}
    return 0;
}