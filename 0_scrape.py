from facebook_scraper import get_posts
import pandas as pd
from source import page_list
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-o', '--output', default='fb_posts.csv')
parser.add_argument('-c', '--cookie', default='cookies/cookie.json')
parser.add_argument('-l', '--list', type=int, default=0)
parser.add_argument('-p', '--page', type=int, default=40)
args = parser.parse_args()

pages = []
all_ids = []
text = []
reactions = { 'like': [], 'love': [], 'care': [], 'haha': [], 'wow': [], 'sad': [], 'angry': [] }
for page in page_list[args.list]:
    posts = get_posts(page, cookies=args.cookie, pages=args.page, options={"allow_extra_requests": False})
    ids = []
    for post in posts:
        ids.append(str(post['post_id']))
    print(ids)
    gposts = get_posts(post_urls=ids, cookies=args.cookie, options={"reactions": True})
    n_posts = len(ids)
    post_iter = iter(gposts)
    for i in range(n_posts):
        try:
            post = next(post_iter)
            text.append(post['post_text'])
            pages.append(page)
            all_ids.append(ids[i])
        except Exception as e:
            print(e)
            continue
        for reaction in reactions:
            try:
                reactions[reaction].append(post["reactions"][reaction])
            except Exception as e:
                # print(e)
                reactions[reaction].append(0)
        print(page, ids[i])
        print(text[-1])
        for r in reactions:
            print(f'{r}: {reactions[r][-1]}')
        data = {'page':pages, 'id':all_ids, 'text': text, **reactions}
        df = pd.DataFrame(data)
        df.to_csv(args.output)