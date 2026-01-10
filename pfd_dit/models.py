from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DDPMScheduler

@dataclass
class LoadedPipelines:
    teacher: DiTPipeline
    student: DiTPipeline

def _make_deterministic_scheduler(pipe: DiTPipeline) -> DPMSolverMultistepScheduler:
    '''
    Deterministic ODE-like solver so the pipeline induces a deterministic noise->image mapping
    given the initial noise (needed for PFD)
    '''
    if hasattr(pipe.scheduler.config, "use_karras_sigmas"):
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def load_teacher_student(
    teacher_id: str,
    *,
    torch_dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    student_from_teacher: bool = True,
) -> LoadedPipelines:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = DiTPipeline.from_pretrained(teacher_id, torch_dtype=torch_dtype)
    teacher.to(device)
    
    train_sched = DDPMScheduler.from_config(teacher.scheduler.config)

    teacher.scheduler = _make_deterministic_scheduler(teacher)

    if student_from_teacher:
        student = DiTPipeline.from_pretrained(teacher_id, torch_dtype=torch_dtype)
    else:
        from diffusers.models import DiTTransformer2DModel
        student_transformer = DiTTransformer2DModel.from_config(teacher.transformer.config)
        student = DiTPipeline(transformer=student_transformer, vae=teacher.vae, scheduler=teacher.scheduler)

    student.to(device)

    student._train_scheduler = DDPMScheduler.from_config(train_sched.config)
    teacher._train_scheduler = DDPMScheduler.from_config(train_sched.config)

    student.scheduler = _make_deterministic_scheduler(student)
    return LoadedPipelines(teacher=teacher, student=student)
