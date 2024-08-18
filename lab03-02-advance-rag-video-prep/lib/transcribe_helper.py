import os
import boto3
import time
from pathlib import Path
#from urllib.request import urlretrieve
from termcolor import colored
import requests

def url_retrieve(url: str, outfile: Path):
    
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    with open(outfile,'wb') as f:
      f.write(r.content)

def transcribe(bucket, path, file, media_format="mp4", language_code="en-US", verbose=True):

    # check to see if transcript already exists
    video_dir = Path(file).stem
    if os.path.exists(os.path.join(video_dir, 'transcript.vtt')):
        print(colored(f"Transcript already exists for {file}", 'yellow'))
        return None

    # start transcription job
    transcribe_response = start_transcription_job(
        bucket, 
        path,
        file, media_format, language_code)

    # wait for completion
    transcribe_response = wait_for_transcription_job(
        transcribe_response['TranscriptionJob']['TranscriptionJobName'], 
        verbose)

    return transcribe_response

def start_transcription_job(bucket, path, file, media_format="mp4", language_code="en-US"):

    # create a random job name
    job_name = '-'.join([
        Path(file).stem,
        os.urandom(4).hex(),
    ])

    key = path+'/'+file

    transcribe_client = boto3.client('transcribe')

    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode=language_code,
        MediaFormat=media_format,
        Media={
            'MediaFileUri': f"s3://{bucket}/{key}",
        },
        Subtitles={
            'Formats': [
                'vtt',
            ],
        },
    )

    return response

def wait_for_transcription_job(job_name, verbose=True):
    transcribe_client = boto3.client('transcribe')

    while True:
        try:
            response = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            transcription_job_status = response['TranscriptionJob']['TranscriptionJobStatus']
            if verbose: 
                print(f"wait_for_transcription_job: status = {transcription_job_status}")
            if transcription_job_status in ['COMPLETED', 'FAILED']:
                return response
            # Sleep for polling loop
            # nosemgrep Rule ID: arbitrary-sleep Message: time.sleep() call; did you mean to leave this in?
            time.sleep(4)
        except Exception as e:
            print(f"Error fetching transcription job status: {e}")
            raise

def estimate_transcribe_cost(duration_ms):
    transcribe_batch_per_min = 0.02400

    transcribe_cost = round(transcribe_batch_per_min * (duration_ms / 60000), 4)


    return {
        'cost_per_min': transcribe_batch_per_min,
        'duration': round(duration_ms / 1000, 2),
        'estimated_cost': transcribe_cost,
    }

def display_transcription_cost(duration_ms, display=True):
    transcribe_cost = estimate_transcribe_cost(duration_ms)

    if display:
        print('\nEstimated cost to Transcribe video:', colored(f"${transcribe_cost['estimated_cost']}", 'green'), f"in us-east-1 region with duration: {transcribe_cost['duration']}s")
    
    return transcribe_cost

def download_vtt(response, output_dir = ''):

    output_file = os.path.join(output_dir, 'transcript.vtt')
    if os.path.exists(output_file):
        return output_file

    subtitle_uri = response['TranscriptionJob']['Subtitles']['SubtitleFileUris'][0]
    url_retrieve(subtitle_uri, output_file)

    return output_file

def download_transcript(response, output_dir = ''):

    output_file = os.path.join(output_dir, 'transcript.json')
    if os.path.exists(output_file):
        return output_file

    transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
    url_retrieve(transcript_uri, output_file)

    return output_file

