"""Optional independent reference for fresh metrics; retain original via unwrap."""
from functools import wraps
from verified_reference import enabled, reference_values


def verified_reference(n):
    def decorate(function):
        @wraps(function)
        def wrapped(x):
            return reference_values(n, x) if enabled() else function(x)
        return wrapped
    return decorate
