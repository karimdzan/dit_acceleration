from dataclasses import dataclass
import torch

from .descriptors import Descriptor

@dataclass
class PFDResult:
    pfd_mean: float
    pfd_std: float
    n: int

@torch.no_grad()
def estimate_pfd(
    *,
    teacher_images_01: torch.Tensor,
    student_images_01: torch.Tensor,
    descriptor: Descriptor,
) -> PFDResult:
    '''
    Empirical PFD estimate (Eq. (5) in the paper): mean L2 distance between descriptor
    embeddings of teacher and student outputs produced from *shared noise*.
    '''
    assert teacher_images_01.shape == student_images_01.shape
    emb_t = descriptor.encode(teacher_images_01)
    emb_s = descriptor.encode(student_images_01)
    d = torch.norm(emb_t - emb_s, dim=-1)
    return PFDResult(
        pfd_mean=float(d.mean().item()),
        pfd_std=float(d.std(unbiased=False).item()),
        n=int(d.numel()),
    )

@torch.no_grad()
def m_distance_to_trainset(
    *,
    student_images_01: torch.Tensor,
    trainset_emb: torch.Tensor,
    descriptor: Descriptor,
    chunk: int = 256,
) -> float:
    '''
    Practical memorization proxy: for each generated sample, compute its minimum distance
    to the training set in descriptor space (Appendix D discusses this family of metrics)
    Returns: mean min-distance across student_images_01
    '''
    emb = descriptor.encode(student_images_01).float()
    train = trainset_emb.float()

    mins = []
    for i in range(0, emb.shape[0], chunk):
        q = emb[i:i+chunk]
        sim = q @ train.T  # cosine similarity
        max_sim = sim.max(dim=-1).values.clamp(-1, 1)
        min_l2 = torch.sqrt((2 - 2 * max_sim).clamp(min=0))
        mins.append(min_l2)
    return float(torch.cat(mins, dim=0).mean().item())
