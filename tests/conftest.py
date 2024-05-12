import os

def pytest_sessionstart():
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + "/opt/homebrew/Caskroom/miniconda/base/envs/s2p/lib"