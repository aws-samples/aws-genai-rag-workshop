from lib import transcribe_helper as trh
from lib import bedrock_helper as brh
from lib import chapters as chpt
from lib import util
from lib import embeddings
from lib import frames
from lib import ffmpeg_helper as ffh
from pathlib import Path
import re
import os
import json


def load_iab_taxonomies(file):
    with open(file) as f:
        iab_taxonomies = json.load(f)
    return iab_taxonomies


def get_chapter_frames(frame_embeddings, scenes_in_chapters):
    num_chapters = len(scenes_in_chapters)
    chapters_frames = [{
        'chapter_id': i,
        'text': '',
        'frames': [],
    } for i in range(num_chapters)]

    for frame in frame_embeddings:
        chapter_id = frame['chapter_id']
        file = frame['file']
        chapters_frames[chapter_id]['frames'].append(file)
        chapters_frames[chapter_id]['text'] = scenes_in_chapters[chapter_id]['text']
        
    return chapters_frames


def generate_chapeter_segements(mp4_file, video_dir, bucket, duration_ms):
    
    transcribe_response = trh.transcribe(bucket, "contextual_ad", mp4_file)

    transcript_filename = trh.download_transcript(transcribe_response, output_dir = video_dir)

    vtt_filename = trh.download_vtt(transcribe_response, output_dir = video_dir)

    transcribe_cost = trh.display_transcription_cost(duration_ms, display=False)

    conversation_response = brh.analyze_conversations(vtt_filename)

    # show the conversation cost
    conversation_cost = brh.display_conversation_cost(conversation_response, display=False)

    ## parse the conversation
    conversations = conversation_response['content'][0]['json']
    
    ## merge overlapped conversation timestamps
    chapters = chpt.merge_chapters(conversations['chapters'])
    
    ## validate the conversation timestamps against the caption timestamps
    captions = chpt.parse_webvtt(vtt_filename)
    chapters = chpt.validate_timestamps(chapters, captions)
    
    conversations['chapters'] = chapters
    
    ## save the conversations
    util.save_to_file(os.path.join(video_dir, 'conversations.json'), conversations)

    return conversations, transcribe_cost, conversation_cost


def group_scene_segements(file_name, video_dir, stream_info):

    jpeg_files = ffh.extract_frames(file_name, stream_info, (392, 220))

    print(f"Frame extracted: {len(jpeg_files)}")

    # generate embeddings =================================
    
    frame_embeddings = embeddings.batch_generate_embeddings(jpeg_files, output_dir = video_dir)

    frame_embeddings_cost = embeddings.display_embedding_cost(frame_embeddings, display=False)

    # group frames into shots ================================
    frames_in_shots = frames.group_frames_to_shots(frame_embeddings)

    print(f"Number of shots: {len(frames_in_shots)} from {len(frame_embeddings)} frames")

    # update shot_id in frame_embeddings dict
    for idx, frames_in_shot in enumerate(frames_in_shots):
        for frame_id in frames_in_shot['frame_ids']:
            frame_embeddings[frame_id]['shot_id'] = idx
    
    # save to json file
    for file, data in [
        ('frames_in_shots.json', frames_in_shots),
        ('frame_embeddings.json', frame_embeddings)
    ]:
        output_file = os.path.join(video_dir, file)
        util.save_to_file(output_file, data)

    ## create an index ===============
    dimension = len(frame_embeddings[0]['embedding'])
    vector_store = embeddings.create_index(dimension)

    ## indexing all the frames ====================
    embeddings.index_frames(vector_store, frame_embeddings)
    print(f"Total indexed = {vector_store.ntotal}")
    
    ## find similar frames for each of the frames and store in the frame_embeddings
    for frame in frame_embeddings:
        similar_frames = embeddings.search_similarity(vector_store, frame)
        frame['similar_frames'] = similar_frames
    
    ## find all similar frames that are related to the shots and store in the frames_in_shots
    for frames_in_shot in frames_in_shots:
        similar_frames_in_shot = frames.collect_similar_frames(frame_embeddings, frames_in_shot['frame_ids'])
        frames_in_shot['similar_frames_in_shot'] = similar_frames_in_shot
    
        related_shots = frames.collect_related_shots(frame_embeddings, similar_frames_in_shot)
        frames_in_shot['related_shots'] = related_shots

    # group shots into scenes
    shots_in_scenes = frames.group_shots_in_scenes(frames_in_shots)
    
    # store the scene_id to all structs
    for scene in shots_in_scenes:
        scene_id = scene['scene_id']
        shot_min, shot_max = scene['shot_ids']
        # update json files
        for shot_id in range(shot_min, shot_max + 1):
            frames_in_shots[shot_id]['scene_id'] = scene_id
            for frame_id in frames_in_shots[shot_id]['frame_ids']:
                frame_embeddings[frame_id]['scene_id'] = scene_id
    
    # update the json files
    # save to json file
    for file, data in [
        ('shots_in_scenes.json', shots_in_scenes),
        ('frames_in_shots.json', frames_in_shots),
        ('frame_embeddings.json', frame_embeddings)
    ]:
        output_file = os.path.join(video_dir, file)
        util.save_to_file(output_file, data)

    return shots_in_scenes, frames_in_shots, frame_embeddings, frame_embeddings_cost


def align_chapters_n_scenes(video_dir, conversations, shots_in_scenes, frames_in_shots, frame_embeddings):
    scenes_in_chapters = frames.group_scenes_in_chapters(
        conversations,
        shots_in_scenes,
        frames_in_shots
    )
    
    for scenes_in_chapter in scenes_in_chapters:
        chapter_id = scenes_in_chapter['chapter_id']
        scene_min, scene_max = scenes_in_chapter['scene_ids']
    
        # update json files
        for scene_id in range(scene_min, scene_max + 1):
            shots_in_scenes[scene_id]['chapter_id'] = chapter_id
            shot_min, shot_max = shots_in_scenes[scene_id]['shot_ids']
            for shot_id in range(shot_min, shot_max + 1):
                frames_in_shots[shot_id]['chapter_id'] = chapter_id
                for frame_id in frames_in_shots[shot_id]['frame_ids']:
                    frame_embeddings[frame_id]['chapter_id'] = chapter_id
    
    # update the json files
    for file, data in [
        ('scenes_in_chapters.json', scenes_in_chapters),
        ('shots_in_scenes.json', shots_in_scenes),
        ('frames_in_shots.json', frames_in_shots),
        ('frame_embeddings.json', frame_embeddings),
    ]:
        output_file = os.path.join(video_dir, file)
        util.save_to_file(output_file, data)

    return scenes_in_chapters, shots_in_scenes, frames_in_shots, frame_embeddings


def extract_min_max_timestamp(files):
    """
    Extracts the minimum and maximum second timestamp from a list of filenames.
    
    Args:
        files (list or str): A list of filenames or a single string containing filenames.
        
    Returns:
        tuple: A tuple containing the minimum and maximum second timestamps as integers.
    """
    if isinstance(files, str):
        files = [files]
    
    timestamps = []
    pattern = r'frames\.(\d+)\.jpg'
    
    for filename in files:
        match = re.search(pattern, filename)
        if match:
            timestamp = int(match.group(1))
            timestamps.append(timestamp)
    
    if timestamps:
        return min(timestamps), max(timestamps)
    else:
        return None, None

def generate_contextual_output(file_name, video_dir, scene_doc_dir, scenes_in_chapters, frames_in_chapters):
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
    }

    prefix = Path(file_name).stem
    
    for frames_in_chapter in frames_in_chapters:
        chapter_id = frames_in_chapter['chapter_id']
        text = frames_in_chapter['text']
        ch_frames = frames_in_chapter['frames']

        start, end = extract_min_max_timestamp(ch_frames)
    
        composite_images = frames.create_composite_images(ch_frames)
        num_images = len(composite_images)
    
        for j in range(num_images):
            composite_image = composite_images[j]
            w, h = composite_image.size
            scaled = composite_image.resize((w // 4, h // 4))


        contextual_response = brh.get_contextual_information(composite_images, text)
            
        # close the images
        for composite_image in composite_images:
            composite_image.close()
    
        usage = contextual_response['usage']
        contextual = contextual_response['content'][0]['json']
    
        # save the contextual to the chapter
        scenes_in_chapters[chapter_id]['contextual'] = {
            'usage': usage,
            **contextual
        }
    
        total_usage['input_tokens'] += usage['input_tokens']
        total_usage['output_tokens'] += usage['output_tokens']
        
        scene_doc = f"==== Chapter #{chapter_id:02d}: Contextual information ======\n\n"
        
        for key in ['description', 'sentiment']:
            
            scene_doc += f"## {key.capitalize()}\n"
            scene_doc += f"{contextual[key]['text']}\n\n"
    
        for key in ['brands_and_logos', 'relevant_tags']:
            items = ', '.join([item['text'] for item in contextual[key]])
            if len(items) == 0:
                items = 'None'
                
            scene_doc += f"## {key.capitalize()}\n"
            scene_doc += f"{items}\n\n"

        doc_file_name = f'{prefix}-{chapter_id:02d}.txt'
        doc_file = os.path.join(scene_doc_dir, doc_file_name)
        util.save_to_file(doc_file, scene_doc)

        # build the file metadata
        doc_metadata = {
            "metadataAttributes": {
                "video": file_name,
                "filename": doc_file_name,
                "start": start,
                "end": end
            }
        }

        metadata_file = os.path.join(scene_doc_dir, f"{doc_file_name}.metadata.json")
        util.save_to_file(metadata_file, doc_metadata)
    
    output_file = os.path.join(video_dir, 'scenes_in_chapters.json')
    util.save_to_file(output_file, scenes_in_chapters)
    
    contextual_cost = brh.display_contextual_cost(total_usage, display=False)

    return contextual_cost