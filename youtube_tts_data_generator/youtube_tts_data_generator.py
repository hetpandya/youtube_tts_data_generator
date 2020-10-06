import youtube_dl
import os
import errno
import warnings
import webvtt
import subprocess
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import librosa
from .audio import preprocess_wav
import shutil
import json
from pydub import AudioSegment


class NoSubtitleWarning(UserWarning):
    pass


class YTLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


class YTSpeechDataGenerator(object):
    """
    YTSpeechDataGenerator makes it easier to
    generate data for Text to Speech/Speech to Text.  .

    Parameters:

    dataset_name:           Name of the dataset

    output_type:            Format of the metadata file.
                            Supported formats:(csv/json)

    keep_audio_extension:   Whether to keep the audio file name
                            extensions in the metadata file.

    Available methods:

    download:               Download wavs from YouTube from a .txt file.

    split_audios:           Split the downloaded single wav files into
                            multiple.

    concat_audios:          Merge multiple smaller audios into a bit
                            longer ones.

    finalize_dataset:       Generate final dataset from the processes
                            audios.
    """

    def __init__(self, dataset_name, output_type="csv", keep_audio_extension=False):
        self.ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "writeautomaticsub": True,
            "logger": YTLogger(),
        }
        self.wav_counter = 0
        self.wav_filenames = []
        self.name = dataset_name
        self.root = os.getcwd()
        self.prep_dir = os.path.join(self.root, self.name + "_prep")
        self.dest_dir = os.path.join(self.root, self.name)
        self.download_dir = os.path.join(self.prep_dir, "downloaded")
        self.split_dir = os.path.join(self.prep_dir, "split")
        self.concat_dir = os.path.join(self.prep_dir, "concatenated")
        self.filenames_txt = os.path.join(self.download_dir, "files.txt")
        self.split_audios_csv = os.path.join(self.split_dir, "split.csv")
        self.len_dataset = 0
        self.len_shortest_audio = 0
        self.len_longest_audio = 0
        self.keep_audio_extension = keep_audio_extension
        if output_type not in ["csv", "json"]:
            raise Exception(
                "Invalid output type. Supported output files are 'csv'/'json'"
            )
        else:
            self.output_type = output_type

        if not os.path.exists(self.prep_dir):
            print(f"Creating directory '{self.name}_prep'..")
            print(f"Creating directory '{self.name}_prep/downloaded'")
            print(f"Creating directory '{self.name}_prep/split'")
            print(f"Creating directory '{self.name}_prep/concatenated'")
            os.mkdir(self.prep_dir)
            os.mkdir(self.download_dir)
            os.mkdir(self.split_dir)
            os.mkdir(self.concat_dir)

        if not os.path.exists(self.dest_dir):
            print(f"Creating directory '{self.name}'..")
            print(f"Creating directory '{self.name}/wavs'")
            print(f"Creating directory '{self.name}/txts'")
            os.mkdir(self.dest_dir)
            os.mkdir(os.path.join(self.dest_dir, "wavs"))
            os.mkdir(os.path.join(self.dest_dir, "txts"))

    def download(self, links_txt):
        """
        Downloads YouTube Videos as wav files.

        Parameters:
              links_txt: A .txt file that contains list of
                         youtube video urls separated by new line.
        """
        self.text_path = os.path.join(self.root, links_txt)
        if os.path.exists(self.text_path) and os.path.isfile(self.text_path):

            links = open(os.path.join(self.text_path)).read().strip().split("\n")

            if os.path.getsize(self.text_path) > 0:
                for ix in range(len(links)):
                    filename = f"{self.name}{ix+1}.mp4"
                    wav_file = filename.replace(".mp4", ".wav")
                    self.ydl_opts["outtmpl"] = os.path.join(self.download_dir, filename)
                    link = links[ix]

                    with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                        caption_info = ydl.extract_info(link, download=False)[
                            "automatic_captions"
                        ]
                        if caption_info == {}:
                            warnings.warn(
                                f"WARNING - video {link} does not have subtitles. Skipping..",
                                NoSubtitleWarning,
                            )
                        else:
                            ydl.download([link])
                            print("Completed downloading " + wav_file + " from " + link)
                            self.wav_counter += 1
                            self.wav_filenames.append(wav_file)

                    del self.ydl_opts["outtmpl"]

                if self.wav_filenames != []:
                    with open(self.filenames_txt, "w") as f:
                        lines = "filename,subtitle,trim_min_begin,trim_min_end\n"
                        for wav in self.wav_filenames:
                            lines += f"{wav},{wav.replace('.wav','.mp4')}.en.vtt,0,0\n"
                        f.write(lines)
                    print(f"Completed downloading audios to '{self.download_dir}'")
                    print(f"You can find files data in '{self.filenames_txt}'")
                else:
                    warnings.warn(
                        f"WARNING - No video with subtitles found to create dataset.",
                        NoSubtitleWarning,
                    )

            else:
                raise Exception(f"ERROR - File '{links_txt}' is empty")
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), links_txt)

    def split_audios(self):
        """
        Split the downloaded videos into smaller chunks.
        """
        if os.path.exists(self.filenames_txt) and os.path.isfile(self.filenames_txt):
            files_list = open(self.filenames_txt).read().strip().split("\n")
            files_list = files_list[1:]
            try:
                check_ffmpeg = subprocess.run(
                    ["ffmpeg"], stderr=subprocess.STDOUT, stdout=subprocess.PIPE
                )

                files_pbar = tqdm(files_list)
                for line in files_pbar:
                    filename, subtitle, trim_min_begin, trim_min_end = line.split(",")
                    files_pbar.set_description("Processing %s" % filename)
                    if not subtitle.endswith(".vtt") or subtitle.endswith(".srt"):
                        raise Exception(
                            "Invalid subtitle type. Supported subtitle types are 'vtt'/'srt'"
                        )
                    trim_min_end = int(trim_min_end)
                    trim_min_begin = int(trim_min_begin)
                    filename = filename[:-4]
                    cnt = 0
                    if subtitle.endswith(".vtt"):
                        captions = webvtt.read(
                            os.path.join(self.download_dir, subtitle)
                        ).captions
                    elif subtitle.endswith(".srt"):
                        captions = webvtt.from_srt(
                            os.path.join(self.download_dir, subtitle)
                        ).captions
                    for ix in range(len(captions)):
                        cap = captions[ix]
                        text = cap.text.strip()
                        t = datetime.strptime(cap.start, "%H:%M:%S.%f").time()

                        if trim_min_end > 0:

                            t2 = datetime.strptime(cap.end, "%H:%M:%S.%f").time()

                            if (
                                t.minute >= trim_min_begin
                                and t2.minute <= trim_min_end
                                and text != "\n"
                            ):
                                text = " ".join(text.split("\n"))

                                new_name = filename + "-" + str(cnt)

                                cmd = [
                                    "ffmpeg",
                                    "-i",
                                    f"{os.path.join(self.download_dir,filename)}.wav",
                                    "-ss",
                                    cap.start,
                                    "-to",
                                    cap.end,
                                    "-c",
                                    "copy",
                                    f"{os.path.join(self.split_dir,new_name)}.wav",
                                ]

                                call = subprocess.run(cmd, stderr=subprocess.STDOUT)

                                with open(
                                    os.path.join(self.split_dir, new_name + ".txt"), "w"
                                ) as f:
                                    f.write(text)
                        else:
                            if t.minute >= trim_min_begin and text != "\n":
                                text = " ".join(text.split("\n"))
                                new_name = filename + "-" + str(cnt)

                                cmd = [
                                    "ffmpeg",
                                    "-i",
                                    f"{os.path.join(self.download_dir,filename)}.wav",
                                    "-ss",
                                    cap.start,
                                    "-to",
                                    cap.end,
                                    "-c",
                                    "copy",
                                    f"{os.path.join(self.split_dir,new_name)}.wav",
                                ]

                                call = subprocess.run(cmd, stderr=subprocess.STDOUT)

                                with open(
                                    os.path.join(self.split_dir, new_name + ".txt"), "w"
                                ) as f:
                                    f.write(text)
                        cnt += 1

                tqdm.write(
                    f"Completed splitting audios and texts to '{self.split_dir}'"
                )

                files_pbar = tqdm(files_list)

                tqdm.write(f"Verifying split audios and their transcriptions.")

                df = []
                for name in files_pbar:
                    filename, subtitle, trim_min_begin, trim_min_end = line.split(",")
                    files_pbar.set_description("Processing %s" % filename)
                    fname = filename[:-4]
                    files = os.listdir(self.split_dir)
                    for ix in range(len(files)):
                        current_file = fname + "-" + str(ix) + ".txt"
                        current_wav = current_file.replace(".txt", ".wav")
                        try:
                            current_text = (
                                open(os.path.join(self.split_dir, current_file))
                                .read()
                                .strip()
                            )

                            if ix != 0:
                                prev_file = fname + "-" + str(ix - 1) + ".txt"

                                prev_text = (
                                    open(os.path.join(self.split_dir, prev_file))
                                    .read()
                                    .strip()
                                )

                                current_text = current_text.replace(
                                    prev_text, ""
                                ).strip()

                            wav, sr = librosa.load(
                                os.path.join(self.split_dir, current_wav)
                            )
                            length = wav.shape[0] / sr

                            if current_text != "" and length > 1:
                                df.append([current_wav, current_text])
                        except:
                            pass

                df = pd.DataFrame(df, columns=["wav_file_name", "transcription"])
                df.to_csv(self.split_audios_csv, sep="|", index=None)

                tqdm.write(
                    f"Completed verifying audios and their transcriptions in '{self.split_dir}'."
                )
                tqdm.write(f"You can find files data in '{self.split_audios_csv}'")

            except FileNotFoundError:
                print(
                    "ERROR - Could not locate ffmpeg. Please install ffmpeg and add it to the environment."
                )
        else:
            print(
                f"ERROR - Couldn't find file 'files.txt'. Make sure it is placed in {self.download_dir}"
            )
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "files.txt"
            )

    def concat_audios(self):
        """
        Joins the chunk of audio files into
        audios of recognizable length.
        """
        if os.path.exists(self.split_audios_csv) and os.path.isfile(
            self.split_audios_csv
        ):
            tqdm.write(f"Reading audio data from 'split.csv'.")
            df = pd.read_csv(self.split_audios_csv, sep="|")
            for ix in tqdm(range(0, df.shape[0], 2)):
                try:
                    combined_sounds = 0
                    text = ""
                    for i in range(ix, ix + 2):
                        audio = df.iloc[i][0]
                        sound1 = AudioSegment.from_wav(
                            os.path.join(self.split_dir, audio)
                        )
                        combined_sounds += sound1
                        text += " " + df.iloc[i][1]
                    text = text.strip()
                    new_name = f"{self.name}-{ix}"
                    combined_sounds.export(
                        os.path.join(self.concat_dir, new_name + ".wav"), format="wav"
                    )
                    with open(
                        os.path.join(self.concat_dir, new_name + ".txt"), "w"
                    ) as f:
                        f.write(text)
                except:
                    pass
            tqdm.write(
                f"Completed concatenating audios and their transcriptions in '{self.concat_dir}'."
            )
        else:
            tqdm.write(
                f"ERROR - Couldn't find file 'split.csv'. Make sure it is placed in {self.split_dir}"
            )
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "split.csv"
            )

    def get_total_audio_length(self):
        """
        Returns the total number of preprocessed audio
        in seconds.
        """

        tqdm.write(
            f"Collected {round(self.len_dataset/3600, 2)}hours ({int(self.len_dataset)} seconds) of audio."
        )
        return int(self.len_dataset)

    def finalize_dataset(self, min_audio_length=7,max_audio_length=14):
        """
        Trims silence from audio files
        and creates a medatada file in csv/json format.

        Parameters:
            min_audio_length: The minimum length of audio files.

            max_audio_length: The maximum length of audio files.
        """

        tqdm.write(f"Trimming silence from audios in '{self.concat_dir}'.")

        concat_audios = [
            wav for wav in os.listdir(self.concat_dir) if wav.endswith(".wav")
        ]
        concat_txt = [wav.replace(".wav", ".txt") for wav in concat_audios]

        filtered_audios = []
        filtered_txts = []
        audio_lens = []

        for ix in tqdm(range(len(concat_audios))):
            audio = concat_audios[ix]
            wav, sr = librosa.load(os.path.join(self.concat_dir, audio))
            silence_removed = preprocess_wav(wav)
            trimmed_length = silence_removed.shape[0] / sr
            audio_lens.append(trimmed_length)
            
            if trimmed_length >= min_audio_length and trimmed_length <= max_audio_length:
                self.len_dataset += trimmed_length
                librosa.output.write_wav(
                    os.path.join(self.dest_dir, "wavs", audio), silence_removed, sr
                )
                filtered_audios.append(audio)
                filtered_txts.append(audio.replace(".wav", ".txt"))

        self.len_shortest_audio = min(audio_lens)
        self.len_longest_audio = max(audio_lens)

        for text in filtered_txts:
            shutil.copyfile(
                os.path.join(self.concat_dir, text),
                os.path.join(self.dest_dir, "txts", text),
            )

        trimmed = []

        for wav, trans in zip(filtered_audios, filtered_txts):
            with open(os.path.join(self.concat_dir, trans)) as f:
                text = f.read().strip()
            trimmed.append([wav, text])

        trimmed = pd.DataFrame(trimmed, columns=["wav_file_name", "transcription"])

        if not self.keep_audio_extension:
            trimmed["wav_file_name"] = trimmed["wav_file_name"].apply(
                lambda x: x.replace(".wav", "")
            )

        if self.output_type == "csv":
            trimmed["transcription_utf"] = trimmed["transcription"]
            trimmed.to_csv(
                os.path.join(self.dest_dir, "metadata.csv"),
                sep="|",
                index=None,
                header=None,
            )
            tqdm.write(
                f"Dataset '{self.name}' has been generated. Wav files are placed in '{self.dest_dir}/wavs'. Transcription files are placed in '{self.dest_dir}/txts'."
            )
            tqdm.write(f"Metadata is placed in '{self.dest_dir}' as 'metadata.csv'.")
        elif self.output_type == "json":
            data = {}
            for ix in range(trimmed.shape[0]):
                name = trimmed.iloc[ix][0]
                text = trimmed.iloc[ix][1]
                data[name] = text
            with open(os.path.join(self.dest_dir, "alignment.json"), "w") as f:
                json.dump(data, f)
            tqdm.write(
                f"Dataset '{self.name}' has been generated. Wav files are placed in '{self.dest_dir}/wavs'. Transcription files are placed in '{self.dest_dir}/txts'."
            )
            tqdm.write(f"Metadata is placed in '{self.dest_dir}' as 'alignment.json'.")

        self.get_total_audio_length()

    def prepare_dataset(
        self, links_txt, download_youtube_data=True, min_audio_length=7,max_audio_length=14
    ):
        """
        A wrapper method for:
          download
          split_audios
          concat_audios
          finalize_dataset

        Downloads YouTube Videos as wav files(optional),
        splits the audios into chunks, joins the
        junks into reasonable audios and trims silence
        from the audios. Creates a metadata file as csv/json
        after the dataset has been generated.

        Parameters:
              links_txt: A .txt file that contains list of
                         youtube video urls separated by new line.

              download_youtube_data: Weather to download data from
                                     Youtube.

              min_audio_length: The minimum length of audio files.

              max_audio_length: The maximum length of audio files.
        """
        if download_youtube_data:
            self.download(links_txt)
        self.split_audios()
        self.concat_audios()
        self.finalize_dataset(min_audio_length)
