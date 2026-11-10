from __future__ import annotations

from torch import Tensor


def session_window_starts(
    sessions: int = 2, trials_per_session: int = 36,
    length: int = 12, stride: int = 6,
) -> list[int]:
    return [
        session * trials_per_session + offset
        for session in range(sessions)
        for offset in range(0, trials_per_session - length + 1, stride)
    ]



def aggregate_subject_probabilities(sequence_logits: Tensor) -> Tensor:
    """Average sequence logits, then apply softmax for a subject prediction."""
    if sequence_logits.ndim != 2 or sequence_logits.shape[-1] != 4:
        raise ValueError('sequence_logits must have shape [sequences, 4]')
    return sequence_logits.mean(dim=0).softmax(dim=-1)
