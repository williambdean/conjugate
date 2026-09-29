from typing import Annotated, Protocol, runtime_checkable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@runtime_checkable
class NUMERIC(Protocol):
    """Protocol for numeric types that support arithmetic operations.

    This protocol defines the interface for types that behave like numbers,
    supporting addition, subtraction, multiplication, division, and comparison
    operations. It is compatible with int, float, numpy arrays, and other
    numeric types.

    The protocol uses `object` for parameter types to support Python's duck
    typing behavior (e.g., int + float works). Return types use `Self` to
    maintain the type of the implementing class.

    Note: Complex numbers do not conform to this protocol because they don't
    support floor division (`//`) and modulo (`%`) operations. If you need to
    work with complex numbers, you may need to use a more specific type hint
    or create a separate protocol.
    """

    # Arithmetic operations
    def __add__(self, other: object) -> Self: ...  # pragma: no cover

    def __radd__(self, other: object) -> Self: ...  # pragma: no cover

    def __sub__(self, other: object) -> Self: ...  # pragma: no cover

    def __rsub__(self, other: object) -> Self: ...  # pragma: no cover

    def __mul__(self, other: object) -> Self: ...  # pragma: no cover

    def __rmul__(self, other: object) -> Self: ...  # pragma: no cover

    def __truediv__(self, other: object) -> Self: ...  # pragma: no cover

    def __rtruediv__(self, other: object) -> Self: ...  # pragma: no cover

    def __floordiv__(self, other: object) -> Self: ...  # pragma: no cover

    def __rfloordiv__(self, other: object) -> Self: ...  # pragma: no cover

    def __mod__(self, other: object) -> Self: ...  # pragma: no cover

    def __rmod__(self, other: object) -> Self: ...  # pragma: no cover

    def __pow__(self, other: object) -> Self: ...  # pragma: no cover

    def __rpow__(self, other: object) -> Self: ...  # pragma: no cover

    # Unary operations
    def __neg__(self) -> Self: ...  # pragma: no cover

    def __pos__(self) -> Self: ...  # pragma: no cover

    def __abs__(self) -> Self: ...  # pragma: no cover

    # Comparison operations
    def __lt__(self, other: object) -> bool: ...  # pragma: no cover

    def __le__(self, other: object) -> bool: ...  # pragma: no cover

    def __gt__(self, other: object) -> bool: ...  # pragma: no cover

    def __ge__(self, other: object) -> bool: ...  # pragma: no cover

    # Note: __eq__, __ne__, and __hash__ are intentionally omitted.
    # Equality comparison is inherited from object, and __hash__ is omitted
    # to support unhashable types like numpy arrays (which have __hash__ = None).


Real = Annotated[NUMERIC, "Real"]
Natural = Annotated[NUMERIC, "Natural"]
PositiveReal = Annotated[NUMERIC, "Positive", "Real"]
Probability = Annotated[NUMERIC, "Probability"]
