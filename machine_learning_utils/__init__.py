from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass
#    from setuptools_scm import get_version
#    __version__ = get_version(root='..', relative_to=__file__)
