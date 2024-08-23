import re

CHAPTER_SEPARATOR = "\n\n\n\n" # The chapter separator used to extract the chapter content.

def extract_chapter(data):
    """
      Extracts chapter from the given data.
    """
    prev_chapter_end= 0
    chapter_contents = []
    while True:
        chapter_end = re.search(CHAPTER_SEPARATOR, data[prev_chapter_end:]) #use regex to find the end of each chapter.
        if chapter_end:
            chapter_end_start_pos = chapter_end.start()
            chapter_content = data[prev_chapter_end:prev_chapter_end+chapter_end_start_pos]
            prev_chapter_end = prev_chapter_end + chapter_end.end()
            chapter_contents.append(chapter_content)
        else:
            break
    return chapter_contents