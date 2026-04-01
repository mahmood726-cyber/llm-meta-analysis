"""
Prometheus metrics for LLM Meta-Analysis Framework.

Tracks key performance indicators for monitoring.
"""

import time
from functools import wraps
from typing import Callable, Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Create registry
if PROMETHEUS_AVAILABLE:
    registry = CollectorRegistry()

    # Counters
    extraction_requests_total = Counter(
        'extraction_requests_total',
        'Total number of extraction requests',
        ['model', 'task', 'status'],
        registry=registry
    )

    llm_calls_total = Counter(
        'llm_calls_total',
        'Total number of LLM API calls',
        ['provider', 'model', 'status'],
        registry=registry
    )

    analysis_requests_total = Counter(
        'analysis_requests_total',
        'Total number of analysis requests',
        ['method', 'status'],
        registry=registry
    )

    db_queries_total = Counter(
        'db_queries_total',
        'Total number of database queries',
        ['operation', 'table', 'status'],
        registry=registry
    )

    # Histograms for timing
    extraction_duration_seconds = Histogram(
        'extraction_duration_seconds',
        'Extraction request duration in seconds',
        ['model', 'task'],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
        registry=registry
    )

    llm_call_duration_seconds = Histogram(
        'llm_call_duration_seconds',
        'LLM API call duration in seconds',
        ['provider', 'model'],
        buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
        registry=registry
    )

    analysis_duration_seconds = Histogram(
        'analysis_duration_seconds',
        'Analysis request duration in seconds',
        ['method'],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
        registry=registry
    )

    db_query_duration_seconds = Histogram(
        'db_query_duration_seconds',
        'Database query duration in seconds',
        ['operation', 'table'],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
        registry=registry
    )

    # Gauges for current state
    active_requests_gauge = Gauge(
        'active_requests',
        'Number of currently active requests',
        ['type'],
        registry=registry
    )

    queue_size_gauge = Gauge(
        'queue_size',
        'Current size of processing queue',
        registry=registry
    )

    cache_size_gauge = Gauge(
        'cache_size',
        'Current size of cache in bytes',
        registry=registry
    )

    # Summary for token usage
    tokens_used_summary = Summary(
        'tokens_used',
        'Summary of tokens used in LLM calls',
        ['provider', 'model', 'type'],
        registry=registry
    )

else:
    # Create no-op versions if prometheus_client is not available
    class NoOpMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def observe(self, amount):
            pass

        def set(self, value):
            pass

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    extraction_requests_total = NoOpMetric()
    llm_calls_total = NoOpMetric()
    analysis_requests_total = NoOpMetric()
    db_queries_total = NoOpMetric()
    extraction_duration_seconds = NoOpMetric()
    llm_call_duration_seconds = NoOpMetric()
    analysis_duration_seconds = NoOpMetric()
    db_query_duration_seconds = NoOpMetric()
    active_requests_gauge = NoOpMetric()
    queue_size_gauge = NoOpMetric()
    cache_size_gauge = NoOpMetric()
    tokens_used_summary = NoOpMetric()


def init_metrics() -> None:
    """Initialize metrics (call this at application startup)."""
    if PROMETHEUS_AVAILABLE:
        import atexit
        # Ensure metrics are registered properly
        atexit.register(lambda: None)


def record_extraction_request(
    model: str,
    task: str,
    status: str,
    duration: float
) -> None:
    """Record an extraction request metric."""
    extraction_requests_total.labels(model=model, task=task, status=status).inc()
    extraction_duration_seconds.labels(model=model, task=task).observe(duration)


def record_llm_call(
    provider: str,
    model: str,
    status: str,
    duration: float,
    tokens: Optional[int] = None
) -> None:
    """Record an LLM API call metric."""
    llm_calls_total.labels(provider=provider, model=model, status=status).inc()
    llm_call_duration_seconds.labels(provider=provider, model=model).observe(duration)
    if tokens is not None:
        tokens_used_summary.labels(provider=provider, model=model, type='total').observe(tokens)


def record_analysis_request(
    method: str,
    status: str,
    duration: float
) -> None:
    """Record an analysis request metric."""
    analysis_requests_total.labels(method=method, status=status).inc()
    analysis_duration_seconds.labels(method=method).observe(duration)


def record_db_query(
    operation: str,
    table: str,
    status: str,
    duration: float
) -> None:
    """Record a database query metric."""
    db_queries_total.labels(operation=operation, table=table, status=status).inc()
    db_query_duration_seconds.labels(operation=operation, table=table).observe(duration)


def track_active_requests(request_type: str, count: int) -> None:
    """Update active requests gauge."""
    active_requests_gauge.labels(type=request_type).set(count)


def track_queue_size(size: int) -> None:
    """Update queue size gauge."""
    queue_size_gauge.set(size)


def track_cache_size(size_bytes: int) -> None:
    """Update cache size gauge."""
    cache_size_gauge.set(size_bytes)


def get_metrics_text() -> bytes:
    """Get Prometheus metrics text format."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(registry)
    return b''


def get_metrics_content_type() -> str:
    """Get Prometheus metrics content type."""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return 'text/plain'


# Decorators for automatic metric recording

def track_extraction(model: str, task: str):
    """Decorator to track extraction requests."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                record_extraction_request(model, task, status, duration)

        return wrapper
    return decorator


def track_llm_call(provider: str, model: str):
    """Decorator to track LLM API calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            tokens = None

            try:
                result = func(*args, **kwargs)
                # Try to extract token count from result
                if isinstance(result, dict) and 'tokens' in result:
                    tokens = result['tokens']
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                record_llm_call(provider, model, status, duration, tokens)

        return wrapper
    return decorator


def track_analysis(method: str):
    """Decorator to track analysis requests."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                record_analysis_request(method, status, duration)

        return wrapper
    return decorator


def track_db_query(operation: str, table: str):
    """Decorator to track database queries."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                record_db_query(operation, table, status, duration)

        return wrapper
    return decorator
