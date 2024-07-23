# utils.py
import os
import PyPDF2
import re

def extract_speaker_info(text):
    speaker_info = {}
    lines = text.split('\n')
    for line in lines:
        match = re.match(r'(\w+ \w+ \(\w+ at \w+\)): (\w+ \d+:\d+)', line)
        if match:
            speaker, timestamp = match.groups()
            speaker_info[timestamp] = speaker
    return speaker_info

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def read_text(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_file(file_path):
    if file_path.endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.endswith(".txt"):
        return read_text(file_path)
    else:
        raise ValueError("Unsupported file format")

def read_files_in_directory(directory_path, max_files=2):
    combined_text = ""
    files_processed = 0
    for filename in os.listdir(directory_path):
        if files_processed >= max_files:
            break
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".pdf") or filename.endswith(".txt"):
            combined_text += read_file(file_path) + "\n"
            files_processed += 1
    return combined_text

def load_transcripts(data_dir):
    transcripts = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r') as file:
                transcripts[filename] = file.read()
    return transcripts

def identify_speakers(transcripts, speaker_info):
    labeled_segments = []
    for filename, transcript in transcripts.items():
        for line in transcript.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                speaker_identifier = parts[0].strip()
                text = parts[1].strip()

                # Check for known speaker labels in the speaker_info dictionary
                matched_speaker = next((key for key in speaker_info.keys() if key in speaker_identifier), None)

                if matched_speaker:
                    speaker_label = speaker_info[matched_speaker]
                else:
                    speaker_label = {'label': 'Unknown', 'organization': 'Unknown'}

                labeled_segments.append({
                    'speaker': speaker_identifier,
                    'label': speaker_label['label'],
                    'organization': speaker_label['organization'],
                    'role': 'sales' if speaker_label['label'] == 'Adam Rencher' else 'product feedback',
                    'text': text,
                    'source': filename
                })
    return labeled_segments
