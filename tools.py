import emoji

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI_ENGLISH:
            return True
    return False