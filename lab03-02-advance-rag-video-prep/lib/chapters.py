import math
import webvtt
from functools import cmp_to_key
import json



def to_milliseconds(timestamp):
    hh, mm, ss = timestamp.split(':')
    ss, ms = ss.split('.')
    hh, mm, ss, ms = map(int, (hh, mm, ss, ms))
    return (((hh * 3600) + (mm * 60) + ss) * 1000) + ms

def to_hhmmssms(milliseconds):
    hh = math.floor(milliseconds / 3600000)
    mm = math.floor((milliseconds % 3600000) / 60000)
    ss = math.floor((milliseconds % 60000) / 1000)
    ms = math.ceil(milliseconds % 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def parse_webvtt(file):
    captions = webvtt.read(file)
    captions = [{
        'text': caption.text,
        'start': caption.start,
        'end': caption.end,
        'start_ms': to_milliseconds(caption.start),
        'end_ms': to_milliseconds(caption.end)
    } for caption in captions]

    return captions

# merge chapters just in case there are overlapped timestamps of the chapters
# sort by start time and by end time
def cmp_timestamps(a, b):
    if a['start_ms'] < b['start_ms']:
        return -1
    if a['start_ms'] > b['start_ms']:
        return 1
    return b['end_ms'] - a['end_ms']

def merge_chapters(chapters):
    # convert timestamp to milliseconds
    for chapter in chapters:
        start = chapter['start']
        end = chapter['end']

        start_ms = to_milliseconds(start)
        end_ms = to_milliseconds(end)

        chapter['start_ms'] = start_ms
        chapter['end_ms'] = end_ms

    chapters = sorted(chapters, key=cmp_to_key(cmp_timestamps))

    # merge chapters if overlap
    merged = [chapters[0]]
    for i in range(1, len(chapters)):
        prev = merged[-1]
        cur = chapters[i]

        prev_start_ms = prev['start_ms']
        prev_end_ms = prev['end_ms']
        cur_start_ms = cur['start_ms']
        cur_end_ms = cur['end_ms']

        if cur_end_ms < prev_start_ms:
            raise Exception('end_ms < start_ms? SHOULD NOT HAPPEN!')

        if cur_start_ms >= prev_end_ms:
            merged.append(cur)
            continue

        # totally overlapped, skip the chapter
        if cur_start_ms > prev_start_ms and cur_end_ms < prev_end_ms:
            continue

        # overlapped, merge the chapters
        start_ms = prev_start_ms
        if start_ms > cur_start_ms:
            start_ms = cur_start_ms
        
        end_ms = prev_end_ms
        if end_ms < cur_end_ms:
            end_ms = cur_end_ms

        prev_duration = prev_end_ms - prev_start_ms
        cur_duration = cur_end_ms - cur_start_ms

        reason = prev['reason']
        if cur_duration > prev_duration:
            reason = cur['reason']

        new_chapter = {
            'reason': reason,
            'start': to_hhmmssms(start_ms),
            'end': to_hhmmssms(end_ms),
            'start_ms': start_ms,
            'end_ms': end_ms
        }

        merged.pop()
        merged.append(new_chapter)

    return chapters


## Validating the timestamp boundaries of the conversations against the WebVtt timestamps
def validate_timestamps(chapters, captions):
    ## collect caption timestamps per chapter
    for chapter in chapters:
        chapter_start = chapter['start_ms']
        chapter_end = chapter['end_ms']

        while len(captions) > 0:
            caption = captions[0]

            caption_start = caption['start_ms']
            caption_end = caption['end_ms']

            if caption_start >= chapter_end:
                break

            if caption_end <= chapter_start:
                captions.pop(0)
                continue

            if abs(chapter_end - caption_start) < abs(caption_end - chapter_end):
                break

            if 'timestamps' not in chapter:
                chapter['timestamps'] = []
            chapter['timestamps'].append([caption_start, caption_end])

            captions.pop(0)

    ## align the chapter boundary timestamps with the caption timestamps
    for chapter in chapters:
        if 'timestamps' not in chapter:
            continue
        
        chapter_start = chapter['start_ms']
        chapter_end = chapter['end_ms']

        caption_start = chapter['timestamps'][0][0]
        caption_end = chapter['timestamps'][-1][1]

        if chapter_start != caption_start:
            chapter['start_ms'] = caption_start
            chapter['start'] = to_hhmmssms(caption_start)

        if chapter_end != caption_end:
            chapter['end_ms'] = caption_end
            chapter['end'] = to_hhmmssms(caption_end)

        del chapter['timestamps']

    return chapters