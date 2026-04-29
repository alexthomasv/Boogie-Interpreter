"""Coverage-guided input generation for the Boogie interpreter.

Hybrid greybox mutation (Phase 1) + constraint-guided generation (Phase 2),
following the QSYM/Driller pattern: fast mutation until coverage stalls,
then targeted constraint solving for hard-to-reach blocks.

Entry point: python3 -m interpreter.coverage_gen.driver <pkg_dir>
"""
