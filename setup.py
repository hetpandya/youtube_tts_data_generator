from setuptools import setup

setup(
    name='youtube_tts_data_generator',
    version='0.1.0',
    description='A python library that generates speech data with transcriptions by collecting data from YouTube.',
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
