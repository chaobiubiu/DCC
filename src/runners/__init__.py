REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .general_runner import GeneralEpisodeRunner
REGISTRY["general"] = GeneralEpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .maven_runner import MavenParallelRunner
REGISTRY["maven"] = MavenParallelRunner
