"""IndentLogger: thread-safe indented logging with timing context manager."""

import threading
import logging
import builtins
import time


class IndentLogger:
    _local = threading.local()
    _logger = logging.getLogger("ray")

    _direct_print_mode = True
    _timing_threshold = 0.1

    def __init__(self, label=None, *args):
        self._label = label
        self._args = args
        self._start_time = None

    def __call__(self, label, *args):
        return IndentLogger(label, *args)

    @classmethod
    def set_direct_print(cls, enabled: bool):
        cls._direct_print_mode = enabled

    @classmethod
    def set_timing_threshold(cls, seconds: float):
        cls._timing_threshold = seconds

    @classmethod
    def _get_level(cls):
        if not hasattr(cls._local, 'level'):
            cls._local.level = 0
        return cls._local.level

    @classmethod
    def _get_indent(cls):
        return "  " * cls._get_level()

    @classmethod
    def _format_msg(cls, *args):
        if not args:
            return ""
        if len(args) > 1 and isinstance(args[0], str) and '%' in args[0]:
            try:
                return args[0] % args[1:]
            except TypeError:
                return " ".join(map(str, args))
        return " ".join(map(str, args))

    @classmethod
    def _output(cls, level, msg, direct_print=None):
        use_print = direct_print if direct_print is not None else cls._direct_print_mode
        if not use_print and not cls._logger.isEnabledFor(level):
            return
        indented_msg = f"{cls._get_indent()}{msg}"
        if use_print:
            builtins.print(indented_msg)
        else:
            cls._logger.log(level, indented_msg)

    @classmethod
    def log(cls, level, *args, direct_print=None):
        use_print = direct_print if direct_print is not None else cls._direct_print_mode
        if not use_print and not cls._logger.isEnabledFor(level):
            return
        msg = cls._format_msg(*args)
        cls._output(level, msg, direct_print=direct_print)

    @classmethod
    def info(cls, *args, **kwargs): cls.log(logging.INFO, *args, **kwargs)
    @classmethod
    def debug(cls, *args, **kwargs): cls.log(logging.DEBUG, *args, **kwargs)
    @classmethod
    def error(cls, *args, **kwargs): cls.log(logging.ERROR, *args, **kwargs)

    @classmethod
    def debug_items(cls, header, items, **kwargs):
        max_items = kwargs.get('max_items', 50)
        cls.debug(header)
        items_list = list(items) if not isinstance(items, list) else items
        for i, item in enumerate(items_list):
            if i >= max_items:
                cls.debug("  ... and %d more", len(items_list) - max_items)
                break
            cls.debug("  - %r", item)

    def __enter__(self):
        if self._label:
            msg = f"[{self._label}]"
            if self._args:
                msg += " " + self._format_msg(*self._args)
            self._output(logging.DEBUG, f"▶ {msg}")
        if not hasattr(self._local, 'level'):
            self._local.level = 0
        self._local.level += 1
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._local.level -= 1
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self._label:
            status = "✗" if exc_type else "✓"
            timing = ""
            if elapsed >= self._timing_threshold:
                if elapsed >= 60:
                    timing = f" ({elapsed/60:.1f}m)"
                elif elapsed >= 1:
                    timing = f" ({elapsed:.1f}s)"
                else:
                    timing = f" ({elapsed*1000:.0f}ms)"
            self._output(logging.DEBUG, f"{status} [{self._label}]{timing}")


# Global instance — works both as bare context manager and as callable
indent_log = IndentLogger()
