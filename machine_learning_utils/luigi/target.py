import random
from pathlib import Path
from contextlib import contextmanager
from luigi.local_target import LocalTarget, atomic_file


class suffix_preserving_atomic_file(atomic_file):
    """Class that preserves the extension of a Temp file"""

    def generate_tmp_path(self, path):
        return (
            path
            + "-luigi-tmp-%09d" % random.randrange(0, 1e10)
            + "".join(Path(path).suffixes)
        )


class BaseAtomicProviderLocalTarget(LocalTarget):
    """Class that gives access to file system operations"""

    atomic_provider = atomic_file

    def open(self, mode="r"):
        rwmode = mode.replace("b", "").replace("t", "")
        if rwmode == "w":
            self.makedirs()
            return self.format.pipe_writer(self.atomic_provider(self.path))

        else:
            super().open()

    @contextmanager
    def temporary_path(self):
        # NB: unclear why LocalTarget doesn't use atomic_file in its implementation
        self.makedirs()
        with self.atomic_provider(self.path) as af:
            yield af.tmp_path


class SuffixPreservingLocalTarget(BaseAtomicProviderLocalTarget):
    """Class that preserves suffix and gives access to file systems operations"""

    atomic_provider = suffix_preserving_atomic_file
