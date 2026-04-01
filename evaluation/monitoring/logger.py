"""
Structured logging for LLM Meta-Analysis Framework.

Provides JSON-formatted logging with context tracking.
"""

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional
from uuid import uuid4

# Thread-local context storage
class Context:
    """Thread-local context for logging."""
    _context = {}

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context value."""
        cls._context[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return cls._context.get(key, default)

    @classmethod
    def update(cls, data: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        cls._context.update(data)

    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._context.clear()

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all context."""
        return cls._context.copy()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add thread and process info
        log_entry['thread_id'] = record.thread
        log_entry['process_id'] = record.process

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }

        # Add context
        context = Context.get_all()
        if context:
            log_entry['context'] = context

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'exc_info', 'exc_text',
                'stack_info', 'getMessage'
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def configure_logging(
    level: str = 'INFO',
    handler: Optional[logging.Handler] = None
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        handler: Optional custom handler (defaults to stdout)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    # Set JSON formatter
    handler.setFormatter(JSONFormatter())

    # Add handler
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


@contextmanager
def log_context(**kwargs):
    """
    Context manager for adding logging context.

    Usage:
        with log_context(request_id='123', user='test'):
            logger.info('Processing request')
    """
    # Store previous context
    previous_context = Context.get_all().copy()

    # Update with new context
    Context.update(kwargs)

    try:
        yield
    finally:
        # Restore previous context
        Context.clear()
        Context.update(previous_context)


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with timing.

    Args:
        logger: Logger instance (uses __name__ if not provided)

    Usage:
        @log_function_call()
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)

            # Generate call ID
            call_id = str(uuid4())[:8]
            Context.set('call_id', call_id)

            # Log start
            _logger.debug(
                'function_call_start',
                extra={
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args)[:1000],
                    'kwargs': str(kwargs)[:1000]
                }
            )

            # Track timing
            start_time = time.time()

            try:
                # Call function
                result = func(*args, **kwargs)

                # Log success
                duration = time.time() - start_time
                _logger.debug(
                    'function_call_success',
                    extra={
                        'function': func.__name__,
                        'duration_ms': round(duration * 1000, 2)
                    }
                )

                return result

            except Exception as e:
                # Log error
                duration = time.time() - start_time
                _logger.error(
                    'function_call_error',
                    extra={
                        'function': func.__name__,
                        'duration_ms': round(duration * 1000, 2),
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


def log_async_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log async function calls with timing.

    Usage:
        @log_async_function_call()
        async def my_async_function(x):
            return await x
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)

            # Generate call ID
            call_id = str(uuid4())[:8]
            Context.set('call_id', call_id)

            # Log start
            _logger.debug(
                'async_function_call_start',
                extra={
                    'function': func.__name__,
                    'module': func.__module__
                }
            )

            # Track timing
            start_time = time.time()

            try:
                # Call function
                result = await func(*args, **kwargs)

                # Log success
                duration = time.time() - start_time
                _logger.debug(
                    'async_function_call_success',
                    extra={
                        'function': func.__name__,
                        'duration_ms': round(duration * 1000, 2)
                    }
                )

                return result

            except Exception as e:
                # Log error
                duration = time.time() - start_time
                _logger.error(
                    'async_function_call_error',
                    extra={
                        'function': func.__name__,
                        'duration_ms': round(duration * 1000, 2),
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                raise

        return wrapper
    return decorator
