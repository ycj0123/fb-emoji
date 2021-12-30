import emoji
import pandas as pd

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI_ENGLISH:
            return True
    return False

if __name__ == '__main__':
    count = 0
    corpus = pd.read_csv('clean.csv')
    posts = corpus['text'].to_list()
    print(len(posts))
    for i, post in enumerate(posts):
        # print(i)
        if text_has_emoji(post):
            count += 1
    print(count/len(posts))