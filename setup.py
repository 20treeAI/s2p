import subprocess

from setuptools import setup
from setuptools.command import build_py, develop


def readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


class CustomDevelop(develop.develop, object):
    """
    Class needed for "pip install -e ."
    """
    def run(self):
        subprocess.check_call("make", shell=True)
        super(CustomDevelop, self).run()


class CustomBuildPy(build_py.build_py, object):
    """
    Class needed for "pip install s2p"
    """
    def run(self):
        super(CustomBuildPy, self).run()
        subprocess.check_call("make", shell=True)
        subprocess.check_call("cp -r bin lib build/lib/", shell=True)


try:
    from wheel.bdist_wheel import bdist_wheel
    class BdistWheel(bdist_wheel):
        """
        Class needed to build platform dependent binary wheels
        """
        def finalize_options(self):
            bdist_wheel.finalize_options(self)
            self.root_is_pure = False

except ImportError:
    BdistWheel = None

requirements = ['numpy',
                'scipy',
                'rasterio[s3]>=1.3.8',
                'utm',
                'pyproj>=3.0.0',
                'beautifulsoup4[lxml]',
                'plyfile',
                'plyflatten>=0.2.0',
                'ransac',
                'rpcm @ git+https://github.com/20treeAI/rpcm.git@v1.4.11',
                'srtm4',
                'requests',
                'opencv-python',
                'geopandas',
                'geopy']

extras_require = {
    "test": ["pytest", "pytest-cov", "psutil"],
}

setup(name="s2p",
      version="1.6.10",
      description="Satellite Stereo Pipeline.",
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/cmla/s2p',
      packages=['s2p'],
      install_requires=requirements,
      extras_require=extras_require,
      cmdclass={'develop': CustomDevelop,
                'build_py': CustomBuildPy,
                'bdist_wheel': BdistWheel},
      python_requires=">=3",
      entry_points="""
          [console_scripts]
          s2p=s2p.cli:main
      """)
