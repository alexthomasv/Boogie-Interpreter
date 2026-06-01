"""Shared execution error types for the Rust interpreter facade."""


class AssertViolation(AssertionError):
    def __init__(self, stmt, pc, block, expr_str):
        self.stmt = stmt
        self.pc = pc
        self.block = block
        self.expr_str = expr_str
        super().__init__(
            f"Assert failed at pc={pc} block={block!r}: {expr_str}"
        )


class AssumeViolation(AssertionError):
    def __init__(self, pc, block, expr_str, reason="assume"):
        self.pc = pc
        self.block = block
        self.expr_str = expr_str
        self.reason = reason
        super().__init__(
            f"Invalid input at pc={pc} block={block!r} "
            f"reason={reason}: {expr_str}"
        )
