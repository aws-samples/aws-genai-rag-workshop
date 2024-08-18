import os
import base64
import copy
from io import BytesIO
from functools import cmp_to_key
from PIL import Image, ImageDraw
from IPython.display import display
from lib import util
from lib import embeddings

def image_to_base64(image):
    buff = BytesIO()
    image.save(buff, format='JPEG')
    return base64.b64encode(buff.getvalue()).decode('utf8')

def skip_frames(frames, max_frames = 80):
    if len(frames) < max_frames:
        return frames

    # miminum step = 2
    skip_step = max(round(len(frames) / max_frames), 2)

    output_frames = []
    for i in range(0, len(frames), skip_step):
        output_frames.append(frames[i])
    
    return output_frames

def create_grid_image(image_files, max_ncol = 10, border_width = 2):
    should_resize = len(image_files) > 50

    with Image.open(image_files[0]) as image:
        width, height = image.size

    ncol = max_ncol
    if len(image_files) < max_ncol:
        ncol = len(image_files)

    nrow = len(image_files) // ncol
    if len(image_files) % ncol > 0:
        nrow += 1
    
    # Create a new image to hold the grid
    grid_width = width * ncol
    grid_height = height * nrow
    grid_image = Image.new("RGB", (grid_width, grid_height))

    draw = ImageDraw.Draw(grid_image)
    # Paste the individual images into the grid
    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        if should_resize:
            image = image.resize((width, height))
        x = (i % ncol) * width
        y = (i // ncol) * height
        grid_image.paste(image, (x, y))
        # draw border
        draw.rectangle((x, y, x + width, y + height), outline=(0, 0, 0), width=border_width)
    
    return grid_image

def create_composite_images(frames):
    reduced = skip_frames(frames, 280)
    # print(f"{len(frames)} -> {len(reduced)}")

    composite_images = []

    for i in range(0, len(reduced), 28):
        frames_per_image = reduced[i:i+28]
        composite_image = create_grid_image(frames_per_image, 4)
        composite_images.append(composite_image)

    return composite_images

def group_frames_to_shots(frame_embeddings, min_similarity = 0.80):
    shots = []
    current_shot = [frame_embeddings[0]]

    # group frames based on the similarity
    for i in range(1, len(frame_embeddings)):
        prev = current_shot[-1]
        cur = frame_embeddings[i]
        prev_embedding = prev['embedding']
        cur_embedding = cur['embedding']

        similarity = embeddings.cosine_similarity(prev_embedding, cur_embedding)
        cur['similarity'] = similarity

        if similarity > min_similarity:
            current_shot.append(cur)
        else:
            shots.append(current_shot)
            current_shot = [cur]

    if current_shot:
        shots.append(current_shot)

    frames_in_shots = []
    for i in range(len(shots)):
        shot = shots[i]
        frames_ids = [frame['frame_no'] for frame in shot]
        frames_in_shots.append({
            'shot_id': i,
            'frame_ids': frames_ids
        })

    return frames_in_shots


def plot_shots(directory, frame_embeddings, num_shots):
    util.mkdir(f'{directory}/shots')

    shots = [[] for _ in range(num_shots)]
    for frame in frame_embeddings:
        shot_id = frame['shot_id']
        file = frame['file']
        shots[shot_id].append(file)

    for i in range(len(shots)):
        shot = shots[i]
        num_frames = len(shot)
        skipped_frames = skip_frames(shot)
        grid_image = create_grid_image(skipped_frames)
        w, h = grid_image.size
        if h > 440:
            grid_image = grid_image.resize((w // 2, h // 2))
        w, h = grid_image.size
        print(f"Shot #{i:04d}: {num_frames} frames ({len(skipped_frames)} drawn) [{w}x{h}]")
        grid_image.save(f"{directory}/shots/shot-{i:04d}.jpg")
        display(grid_image)
        grid_image.close()
    print('====')

def collect_similar_frames(frame_embeddings, frame_ids):
    similar_frames = []
    for frame_id in frame_ids:
        similar_frames_ids = [frame['idx'] for frame in frame_embeddings[frame_id]['similar_frames']]
        similar_frames.extend(similar_frames_ids)
    # unique frames in shot
    return sorted(list(set(similar_frames)))

def collect_related_shots(frame_embeddings, frame_ids):
    related_shots = []
    for frame_id in frame_ids:
        related_shots.append(frame_embeddings[frame_id]['shot_id'])
    # unique frames in shot
    return sorted(list(set(related_shots)))


def group_shots_in_scenes(frames_in_shots):
    scenes = [
        [
            min(frames_in_shot['related_shots']),
            max(frames_in_shot['related_shots']),
        ] for frames_in_shot in frames_in_shots
    ]

    scenes = sorted(scenes, key=cmp_to_key(embeddings.cmp_min_max))

    stack = [scenes[0]]
    for i in range(1, len(scenes)):
        prev = stack[-1]
        cur = scenes[i]
        prev_min, prev_max = prev
        cur_min, cur_max = cur

        if cur_min >= prev_min and cur_min <= prev_max:
            new_scene = [
                min(cur_min, prev_min),
                max(cur_max, prev_max),
            ]
            stack.pop()
            stack.append(new_scene)
            continue
            
        stack.append(cur)

    return [{
        'scene_id': i,
        'shot_ids': stack[i],
    } for i in range(len(stack))]

def plot_scenes(directory, frame_embeddings, num_scenes):
    util.mkdir(f'{directory}/scenes')

    scenes = [[] for _ in range(num_scenes)]
    for frame in frame_embeddings:
        scene_id = frame['scene_id']
        file = frame['file']
        scenes[scene_id].append(file)

    for i in range(len(scenes)):
        scene = scenes[i]
        num_frames = len(scene)
        skipped_frames = skip_frames(scene)
        grid_image = create_grid_image(skipped_frames)
        w, h = grid_image.size
        if h > 440:
            grid_image = grid_image.resize((w // 2, h // 2))
        w, h = grid_image.size
        print(f"Scene #{i:04d}: {num_frames} frames ({len(skipped_frames)} drawn) [{w}x{h}]")
        grid_image.save(f"{directory}/scenes/scene-{i:04d}.jpg")
        display(grid_image)
        grid_image.close()
    print('====')


def make_chapter_item(chapter_id, scene_items, text = ''):
    scene_ids = [scene['scene_id'] for scene in scene_items]
    return {
        'chapter_id': chapter_id,
        'scene_ids': [min(scene_ids), max(scene_ids)],
        'text': text,
    }

def group_scenes_in_chapters(conversations, shots_in_scenes, frames_in_shots):
    scenes = copy.deepcopy(shots_in_scenes)

    chapters = []
    for conversation in conversations['chapters']:
        start_ms = conversation['start_ms']
        end_ms = conversation['end_ms']
        text = conversation['reason']

        stack = []
        while len(scenes) > 0:
            scene = scenes[0]
            shot_min, shot_max = scene['shot_ids']
            frame_start = min(frames_in_shots[shot_min]['frame_ids']) * 1000
            frame_end = max(frames_in_shots[shot_max]['frame_ids']) * 1000

            if frame_start > end_ms:
                break

            # scenes before any conversation starts
            if frame_end < start_ms:
                chapter = make_chapter_item(len(chapters), [scene])
                chapters.append(chapter)
                scenes.pop(0)
                continue

            stack.append(scene)
            scenes.pop(0)

        if stack:
            chapter = make_chapter_item(len(chapters), stack, text)
            chapters.append(chapter)

    ## There could be more scenes without converations, append them
    for scene in scenes:
        chapter = make_chapter_item(len(chapters), [scene])
        chapters.append(chapter)

    return chapters

def plot_chapters(directory, frame_embeddings, num_chapters):
    try:
        os.mkdir(f'{directory}/chapters')
    except Exception as e:
        print(e)

    chapters = [[] for _ in range(num_chapters)]
    for frame in frame_embeddings:
        chapter_id = frame['chapter_id']
        file = frame['file']
        chapters[chapter_id].append(file)

    for i in range(len(chapters)):
        chapter = chapters[i]
        num_frames = len(chapter)
        skipped_frames = skip_frames(chapter)
        grid_image = create_grid_image(skipped_frames)
        w, h = grid_image.size
        if h > 440:
            grid_image = grid_image.resize((w // 2, h // 2))
        w, h = grid_image.size
        print(f"Chapter #{i:04d}: {num_frames} frames ({len(skipped_frames)} drawn) [{w}x{h}]")
        grid_image.save(f"{directory}/chapters/chapter-{i:04d}.jpg")
        display(grid_image)
        grid_image.close()
    print('====')
