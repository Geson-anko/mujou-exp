import os

# mujou runs the hooks on worker threads, where OpenMP starts with as many
# threads as cores: `torch.set_num_threads(1)` of the host reaches a worker only
# once torch initializes its threads there, and MKL does not, so a matrix
# product may split over the cores and round differently from run to run.
# OpenMP reads this when torch loads it: set it before any test imports torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
