from ..parsers import parse_md_with_frontmatter
from langchain.text_splitter import LatexTextSplitter, MarkdownTextSplitter


def transform_handbook_content(book_index, entry, chunk_size=8000):
    """
    entry here is a dict from cp_handbook.json

    cp_handbook.json is a list of dicts with keys
    'chapter_name', 'section_name', 'section_content' (in LaTeX), 'chapter_path' (path of LaTeX file)
    total length of section contents: 766939 char or approx 190k tokens
    longest section content: 24036 char or approx 6k tokens
    Note: quite a num of diagrams
    """
    main_text = entry['section_content']
    text_splitter = LatexTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunker = text_splitter.split_text
    # langchain latex chunker to split section_content to max 4k tokens
    raw_texts = chunker(main_text)
    texts = []
    metadatas = []
    num_texts = len(raw_texts)
    for entry_idx, raw_text in enumerate(raw_texts):
        # for each chunk, state 'chapter_name', 'section_name' and its index if it was split
        text = (
            f'Chapter: {entry["chapter_name"]}\nSection: {entry["section_name"]}\nPart: {entry_idx+1}/{num_texts}'
            f'\n{raw_text}'
        )
        # metadata store i, 'chapter_name', 'section_name', 'chapter_path' and its index if it was split
        metadata = {
            'chapter_name': entry['chapter_name'],
            'section_name': entry['section_name'],
            'chapter_path': entry['chapter_path'],
            'entry_idx': entry_idx,
            'num_texts': num_texts,
            'book_index': book_index,
        }

        texts.append(text)
        metadatas.append(metadata)
    return texts, metadatas


def transform_bookv2_content(book_index, entry, chunk_size=8000):
    """
    entry here is a dict from cpbook_v2.json

    the loaded json is a list of dicts with keys
    'article' (in markdown with latex eqns), 'full_article', 'problem_ids' (list), 'title'
    'article' is 'full_article' with the practice problems removed (links to online problems)
    'article' has YAML frontmatter with tags. needs to be removed
    code is delimited by triple backticks, mostly cpp
    total len of articles: 1387242 char or approx 350k tokens
    longest article: 65417 char or approx 16k tokens, need to chunk for gpt 3.5
    """
    main_text = entry['article']
    # remove YAML frontmatter
    _, main_text = parse_md_with_frontmatter(main_text)

    # langchain markdown chunker to split section_content to max 4k tokens
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunker = text_splitter.split_text
    raw_texts = chunker(main_text)
    texts = []
    metadatas = []
    num_texts = len(raw_texts)
    for entry_idx, raw_text in enumerate(raw_texts):
        # for each chunk, state 'title', and its index if it was split
        text = f'Title: {entry["title"]}\nPart: {entry_idx+1}/{num_texts}\n{raw_text}'
        # metadata store i, 'title' and its index if it was split
        metadata = {
            'title': entry['title'],
            'entry_idx': entry_idx,
            'num_texts': num_texts,
            'book_index': book_index,
        }
        texts.append(text)
        metadatas.append(metadata)
    return texts, metadatas
