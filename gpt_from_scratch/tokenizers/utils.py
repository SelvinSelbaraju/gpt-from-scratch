def create_char_vocab(text: str, sort=True) -> list:
    unique_chars = list(set(text))
    if sort:
        return sorted(unique_chars)
    else:
        return unique_chars
