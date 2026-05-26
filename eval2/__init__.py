"""Second-generation TACO benchmark: persistent cognition & longitudinal continuity.

Distinct from `eval/` (gen-1, retrieval efficiency). This suite measures whether
TACO's added cognition layers — identity abstraction, reconsolidation, predictive
continuity, memory-conditioned reasoning, state-conditioned retrieval — produce
measurable *continuity* gains over a realistic naive-RAG memory stack, using the
same base model. Outputs are written to `Results/` with the suffix `2`; gen-1
artifacts in `eval/out/` are never touched.
"""
