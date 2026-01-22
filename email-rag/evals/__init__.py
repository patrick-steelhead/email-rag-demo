# Email RAG Pipeline - Evaluation Module

from .precision import (
    calculate_precision,
    run_evaluation as run_precision_evaluation,
    TEST_CASES as PRECISION_TEST_CASES,
)

from .latency import (
    run_latency_evaluation,
    run_single_test as run_latency_test,
    LATENCY_TEST_CASES,
)

from .precision_delta import (
    run_evaluation as run_precision_delta_evaluation,
    evaluate_precision_delta,
    PRECISION_DELTA_TEST_CASES,
)

__all__ = [
    # Precision evaluation
    "calculate_precision",
    "run_precision_evaluation",
    "PRECISION_TEST_CASES",
    # Latency evaluation
    "run_latency_evaluation",
    "run_latency_test",
    "LATENCY_TEST_CASES",
    # Precision delta evaluation
    "run_precision_delta_evaluation",
    "evaluate_precision_delta",
    "PRECISION_DELTA_TEST_CASES",
]
