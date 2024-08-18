import os
import time
import json
import glob
import subprocess
import shlex
from pathlib import Path
from urllib.parse import urlparse
from lib import util

def probe_stream(video_url):
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    stream_info_file = os.path.join(video_dir, 'stream_info.json')

    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')
        
    util.mkdir(video_dir)
    

    # check to see if wav file already exists
    if os.path.exists(stream_info_file):
        with open(stream_info_file, 'r', encoding="utf-8") as f:
            stream_info = json.loads(f.read())
        print(f"  probe_stream: found stream_info.json. SKIPPING...")
        return stream_info

    command_string = f'ffprobe -v quiet -print_format json -show_format -show_streams {shlex.quote(video_url)}'
    
    # shlex.quote will place harmful input in quotes so it can't be executed by the shell
    # nosemgrep Rule ID: dangerous-subprocess-use-audit
    child_process = subprocess.Popen(
        shlex.split(command_string),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = child_process.communicate()

    stream_info = json.loads(str(stdout, 'utf-8'))
    stream_info['format']['filename'] = video_file

    # parse video_info
    video_stream = list(filter(lambda x: (x['codec_type'] == 'video'), stream_info['streams']))[0]

    progressive = bool('field_order' not in video_stream or video_stream['field_order'] == 'progressive')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration_ms = int(float(video_stream['duration']) * 1000)
    num_frames = int(video_stream['nb_frames'])
    framerate = util.to_fraction(video_stream['r_frame_rate'])
    sample_aspect_ratio = util.to_fraction(video_stream['sample_aspect_ratio'])
    display_aspect_ratio = util.to_fraction(video_stream['display_aspect_ratio'])
    display_width = int((width * sample_aspect_ratio.numerator) / sample_aspect_ratio.denominator)

    stream_info['video_stream'] = {
        'duration_ms': duration_ms,
        'num_frames': num_frames,
        'framerate': (framerate.numerator, framerate.denominator),
        'progressive': progressive,
        'sample_aspect_ratio': (sample_aspect_ratio.numerator, sample_aspect_ratio.denominator),
        'display_aspect_ratio': (display_aspect_ratio.numerator, display_aspect_ratio.denominator),
        'encoded_resolution': (width, height),
        'display_resolution': (display_width, height),
    }

    util.save_to_file(stream_info_file, stream_info)

    return stream_info

def extract_frames(video_url, stream_info, max_res = (750, 500)):
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    
    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')

    frame_dir = os.path.join(video_dir, 'frames')
    if os.path.exists(frame_dir):
        jpeg_frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
        print(f"  extract_frames: found {len(jpeg_frames)} frames. SKIPPING...")
        return jpeg_frames

    util.mkdir(frame_dir)

    t0 = time.time()
    video_filters = []
    video_stream = stream_info['video_stream']

    # need deinterlacing
    progressive = video_stream['progressive']
    if not progressive:
        video_filters.append('yadif')

    # downscale image
    dw, dh = video_stream['display_resolution']
    factor = max((max_res[0] / dw), (max_res[1] / dh))
    w = round((dw * factor) / 2) * 2
    h = round((dh * factor) / 2) * 2
    video_filters.append(f"scale={w}x{h}")

    # ffmpeg -ss 588 -i f"{video_url}" -vf "yadif,scale=iw*sar:ih" -frames:v 1 test2.jpg
    
    command = [
        'ffmpeg',
        '-v',
        'quiet',
        '-i',
        shlex.quote(video_url),
        # '-t',
        # str(60),
        '-vf',
        f"{','.join(video_filters)}",
        '-r',
        str(1),
        f"{shlex.quote(frame_dir)}/frames.%07d.jpg"
    ]

    print(f"  Resizing: {dw}x{dh} -> {w}x{h} (Progressive? {progressive})")
    print(f"  Command: {command}")
    
    # shlex.quote will place harmful input in quotes so it can't be executed by the shell
    # nosemgrep Rule ID: dangerous-subprocess-use-audit
    subprocess.run(
        command,
        shell=False,
        stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )

    t1 = time.time()
    print(f"  extract_frames: elapsed {round(t1 - t0, 2)}s")

    # return jpeg files
    jpeg_frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
    return jpeg_frames

def extract_audio(video_url):
    t0 = time.time()
    
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    
    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')

    audio_dir = Path(urlparse(video_url).path).stem
    wav_file = os.path.join(audio_dir, 'audio.wav')

    # check to see if wav file already exists
    if os.path.exists(wav_file):
        print(f"  extract_audio: found audio.wav. SKIPPING...")
        return wav_file

    # run ffmpeg to extract audio
    bitrate = '96k'
    sampling_rate = 16000
    channel = 1
    command = [
        'ffmpeg',
        '-i',
         shlex.quote(video_url),
        '-vn',
        '-c:a',
        'pcm_s16le',
        '-ab',
        bitrate,
        '-ar',
        str(sampling_rate),
        '-ac',
        str(channel),
        wav_file
    ]
    print(command)
    
    # shlex.quote will place harmful input in quotes so it can't be executed by the shell
    # nosemgrep Rule ID: dangerous-subprocess-use-audit
    subprocess.run(
        command,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    t1 = time.time()
    print(f"  extract_audio: elapsed {round(t1 - t0, 2)}s")
    return wav_file

def create_lowres_video(video_url, stream_info, max_res = (360, 202)):
    video = urlparse(video_url)
    video_file = video.path
    video_dir = Path(video_file).stem
    
    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(video_file):
        raise Exception('input video does not exist')
    
    low_res_video_file = os.path.join(video_dir, 'lowres_video.mp4')

    # check to see if video file already exists
    if os.path.exists(low_res_video_file):
        print(f"  create_lowres_video: found lowres_video.mp4. SKIPPING...")
        return low_res_video_file

    util.mkdir(video_dir)

    video_stream = stream_info['video_stream']

    video_filters = []
    # need deinterlacing
    progressive = video_stream['progressive']
    if not progressive:
        video_filters.append('yadif')

    # downscale image
    dw, dh = video_stream['display_resolution']
    factor = max((max_res[0] / dw), (max_res[1] / dh))
    w = round((dw * factor) / 2) * 2
    h = round((dh * factor) / 2) * 2
    video_filters.append(f"scale={w}x{h}")

    # ffmpeg -ss 588 -i f"{video_url}" -vf "yadif,scale=iw*sar:ih" -frames:v 1 test2.jpg
    command = [
        'ffmpeg',
        '-v',
        'quiet',
        '-i',
        shlex.quote(video_url),
        '-vf',
        f"{','.join(video_filters)}",
        '-ac',
        str(2),
        '-ab',
        '64k',
        '-ar',
        str(44100),
        f"{low_res_video_file}"
    ]

    print(f"  Downscaling: {dw}x{dh} -> {w}x{h} (Progressive? {progressive})")
    print(f"  Command: {command}")

    t0 = time.time()
    
    # shlex.quote will place harmful input in quotes so it can't be executed by the shell
    # nosemgrep Rule ID: dangerous-subprocess-use-audit
    subprocess.run(
        command,
        shell=False,
        stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )

    t1 = time.time()
    print(f"  downscale_video: elapsed {round(t1 - t0, 2)}s")

    return low_res_video_file
