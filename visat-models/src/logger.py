import header
import type

def log_trace(*args, **kwargs):
    if header.log_level >= type.LogLevel.trace:
        print("[TRACE]: ", end = "")
        print(*args, **kwargs)

    return

def log_debug(*args, **kwargs):
    if header.log_level >= type.LogLevel.debug:
        print("[DEBUG]: ", end = "")
        print(*args, **kwargs)

    return

def log_info(*args, **kwargs):
    if header.log_level >= type.LogLevel.info:
        print("[INFO]: ", end = "")
        print(*args, **kwargs)

    return

def log_warn(*args, **kwargs):
    if header.log_level >= type.LogLevel.warn:
        print("[WARN]: ", end = "")
        print(*args, **kwargs)

    return

def log_error(*args, **kwargs):
    if header.log_level >= type.LogLevel.error:
        print("[ERROR]: ", end = "")
        print(*args, **kwargs)

    return

def log_fatal(*args, **kwargs):
    if header.log_level >= type.LogLevel.fatal:
        print("[FATAL]: ", end = "")
        print(*args, **kwargs)

    return

def log_trace_raw(*args, **kwargs):
    if header.log_level >= type.LogLevel.trace:
        print(*args, **kwargs)

    return

def log_debug_raw(*args, **kwargs):
    if header.log_level >= type.LogLevel.debug:
        print(*args, **kwargs)

    return

def log_info_raw(*args, **kwargs):
    if header.log_level >= type.LogLevel.info:
        print(*args, **kwargs)

    return

def log_warn_raw(*args, **kwargs):
    if header.log_level >= type.LogLevel.warn:
        print(*args, **kwargs)

    return

def log_error_raw(*args, **kwargs):
    if header.log_level >= type.LogLevel.error:
        print(*args, **kwargs)

    return

def log_fatal_raw(*args, **kwargs):
    if header.log_level >= type.LogLevel.fatal:
        print(*args, **kwargs)

    return