from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='youtube_tts_data_generator',
    version='0.1.3',
    description='A python library that generates speech data with transcriptions by collecting data from YouTube.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Het Pandya',
    url='http://github.com/thehetpandya/youtube_tts_data_generator',
    author_email='hetpandya6797@gmail.com',
    license='MIT',
    install_requires=[
       "webvtt-py",
       "librosa>=0.5.1",
       "youtube-dl",
       "tqdm",
       "pandas",
       "pydub",
       "scikit-learn==0.19.2",
       "webrtcvad",
       "scipy>=1.0.0",
       "numba==0.48",
       "inflect",
       "numpy>=1.14.0",
   ],
   packages=['youtube_tts_data_generator'],
)
