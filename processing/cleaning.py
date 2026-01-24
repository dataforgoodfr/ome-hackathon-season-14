import re

CLEANING_TEXT_REGEX_PATTERN = re.compile(r"[^\w ]+")


def clean_input_text(input_text: str) -> str:
    """
    Main cleaning method :
    - Remove non alphanumeric characters
    - Remove leading and trailing whitespaces
    - Remove duplicated text based on first words
    """
    cleaned_text = CLEANING_TEXT_REGEX_PATTERN.sub("", input_text.strip())
    words = cleaned_text.split()
    # Select the first words to see duplicated
    first_words = " ".join(words[0:10])
    duplicated_texts = [segment for segment in cleaned_text.split(first_words) if len(segment) > 0]

    return first_words + duplicated_texts[0]
