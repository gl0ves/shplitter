import os
import yt_dlp
import librosa
import numpy as np
import torch
from demucs.api import Separator
from demucs.audio import save_audio


def download_audio(url, download_folder):
    output_template = os.path.join(download_folder, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        downloaded_file = (
            ydl.prepare_filename(result)
            .replace(".webm", ".mp3")
            .replace(".m4a", ".mp3")
        )

    return downloaded_file


def detect_key_and_bpm(input_path):
    # Load the audio file
    y, sr = librosa.load(input_path)

    # Detect BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)  # Ensure tempo is a scalar

    # Detect key
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Major and minor template profiles for comparison
    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    # List of keys (C, C#, D, D#,..., B)
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Compare chroma profile with major and minor templates
    major_scores = [np.dot(np.roll(major_profile, i), chroma_mean) for i in range(12)]
    minor_scores = [np.dot(np.roll(minor_profile, i), chroma_mean) for i in range(12)]

    # Find the key with the highest score (either major or minor)
    best_major = np.argmax(major_scores)
    best_minor = np.argmax(minor_scores)

    if max(major_scores) > max(minor_scores):
        key = keys[best_major] + " major"
    else:
        key = keys[best_minor] + " minor"

    return key, tempo


def split_audio(input_path, output_folder):
    print(f"Splitting audio: {input_path}")

    # Set the device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a Separator instance
    separator = Separator(model="htdemucs_ft", device=device)

    # Separate the audio
    wav, sources = separator.separate_audio_file(input_path)

    # Ensure the output directory exists with proper permissions
    os.makedirs(output_folder, exist_ok=True)

    # Save the separated tracks
    for source, audio in sources.items():
        out = os.path.join(output_folder, f"{source}.wav")
        try:
            save_audio(audio, out, separator.samplerate)
            print(f"Saved {source} to {out}")
        except Exception as e:
            print(f"Error saving {source}: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Output folder: {output_folder}")
            print(f"File path: {out}")
            print(f"File path exists: {os.path.exists(os.path.dirname(out))}")
            print(f"File path writable: {os.access(os.path.dirname(out), os.W_OK)}")

    print("Audio splitting completed.")


def main():
    download_folder = "inputs"
    output_folder = "outputs"

    os.makedirs(download_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    while True:
        url = input("Enter YouTube URL (or type 'exit' to quit): ")
        if url.lower() == "exit":
            break

        # Download the audio from the provided URL
        input_path = download_audio(url, download_folder)
        if not os.path.exists(input_path):
            print("Error downloading the file.")
            continue

        # Detect key and BPM
        key, bpm = detect_key_and_bpm(input_path)
        print(f"Detected Key: {key}, BPM: {bpm}")

        # Extract song title without the file extension
        song_title = os.path.splitext(os.path.basename(input_path))[0]

        # Create a new output folder name based on song title, key, and BPM
        new_output_folder = os.path.join(
            output_folder, f"{song_title}_{key}_{int(bpm)}"
        )
        os.makedirs(new_output_folder, exist_ok=True)

        # Split the audio
        split_audio(input_path, new_output_folder)


if __name__ == "__main__":
    main()
