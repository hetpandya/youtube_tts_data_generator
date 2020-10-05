# Youtube Speech Data Generator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A python library to generate speech dataset. Youtube Speech Data Generator also takes care of almost all your speech data preprocessing needed to build a speech dataset along with their transcriptions.

## Installation
Make sure [ffmpeg](https://ffmpeg.org/download.html#get-packages) is installed and is set to the system path.

```bash
$ pip install youtube-tts-data-generator
```

## Minimal start for creating the dataset

```python
from youtube_tts_data_generator import YTSpeechDataGenerator

# First create a YTSpeechDataGenerator instance:

generator = YTSpeechDataGenerator(dataset_name='elon')

# Now create a '.txt' file that contains a list of YouTube videos that contains speeches.
# NOTE - Make sure you choose videos with subtitles.

generator.prepare_dataset('links.txt')
# The above will take care about creating your dataset, creating a metadata file and trimming silence from the audios.

```

## Usage
<!--ts-->
- Initializing the generator:
  ```generator = YTSpeechDataGenerator(dataset_name='your_dataset')```
  - Parameters:
    - *dataset_name*: 
      - The name of the dataset you'd like to give. 
      - A directory structure like this will be created:
        ```
        ├───your_dataset
        │   ├───txts
        │   └───wavs
        └───your_dataset_prep
            ├───concatenated
            ├───downloaded
            └───split
        ```
    - *output_type*: 
      - The type of the metadata to be created after the dataset has been generated.
      - Supported types: csv/json
      - Default output type is set to *csv*
    - *keep_audio_extension*:
      - Whether to keep the audio file extension in the metadata file
      - Default value is set to *False*
 
- Methods:
  - download():
    - Downloads video files from YouTube along with their subtitles and saves them as wav files.
    - Parameters:
      - *links_txt*:
        - Path to the '.txt' file that contains the urls for the videos.
    - Usage of this method is optional. If you do not use this method, make sure to place all the audio and subtitle files in 'your_dataset_prep/downloaded' directory. 
    - Then, create a file called 'files.txt' and again place it under 'your_dataset_prep/downloaded'.
      'files.txt' should follow the following format:
      ```
      filename,subtitle,trim_min_begin,trim_min_end
      audio.wav,subtitle.srt,0,0
      audio2.wav,subtitle.vtt,5,6
      ```
    - Create a '.txt' file that contains a list of YouTube videos that contains speeches.
    - Example - ```generator.download('links.txt')```
  - split_audios():
    - This method splits all the wav files into smaller chunks according to the duration of the text in the subtitles.
    - Saves transcriptions as '.txt' file for each of the chunks.
    - Example - ```generator.split_audios()```
  - concat_audios():
    - Since the split audios are based on the duration of their subtitles, they might not be so long. This method joins the split files into recognizable ones.
    - Example - ```generator.concat_audios()```
  - finalize_dataset():
    - Trims silence the joined audios since the data has been collected from YouTube and generates the final dataset after finishing all the preprocessing.
    - Parameters:
      - *min_audio_length*:
        - The minumum length of the speech that should be kept. The rest will be ignored.
        - The default value is set set to *7*.
    - Example - ```generator.finalize_dataset(min_audio_length=6)```
  - get_total_audio_length():
    - Returns the total amount of preprocessed speech data collected by the generator.
    - Example - ```generator.get_total_audio_length()```
  - prepare_dataset():
    - A wrapper method for *download()*,*split_audios()*,*concat_audios()* and *finalize_dataset()*.
    - If you do not wish to use the above methods, you can directly call *prepare_dataset()*. It will handle all your data generation.
    - Parameters:
      - *links_txt*:
        - Path to the '.txt' file that contains the urls for the videos.
      - *download_youtube_data*:
        - Whether to download audios from YouTube.
        - Default value is *True*
      - *min_audio_length*:
        - The minumum length of the speech that should be kept. The rest will be ignored.
    - Example - ```generator.prepare_dataset(links_txt='links.txt',download_youtube_data=True,min_audio_length=6)```
<!--te-->

## Final dataset structure
Once the dataset has been created, the structure under 'your_dataset' directory should look like:
```
your_dataset
├───txts
│   ├───your_dataset1.txt
│   └───your_dataset2.txt
├───wavs
│    ├───your_dataset1.wav
│    └───your_dataset2.wav
└───metadata.csv/alignment.json
```

NOTE - `audio.py` is highly based on [Real Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/master/encoder/audio.py)
