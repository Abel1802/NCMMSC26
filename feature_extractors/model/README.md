# Collaborative-gate model

`collaborative_gate.py` adapts `Speaker_Independent_Triple_Mode_without_Context`
from `mlt_sarcasm/src/SVM_DNN/run_dnn.py`. Each selected modality is projected
to a shared dimension. For multimodal runs, each anchor receives a sum of
pairwise softmax weights from the other available modalities, followed by a
second softmax gate. The gated vectors are concatenated in T,V,A order and
classified by an MLP. For unimodal runs the pair gate is omitted.

The default shared and gate projection dimensions are 256 and 128. LayerNorm
replaces BatchNorm so a final batch of size one is valid. The gate is multiplied
by the shared dimension after softmax to maintain its activation scale. Video's
eight frame vectors are mean pooled in the data loader, before projection.
