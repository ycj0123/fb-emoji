from facebook_scraper import get_posts
import pandas as pd

page_list = [
    # 'tsaiingwen',
    # 'ETtoday',
    # 'tiktoktaiwan',
    # 'TerryGou1018',
    # 'myudn',
    # '100044226139684',
    # 'DoctorKoWJ',
    # 'mohw.gov.tw',
    # '100044207960749',
    # '100044580272976',
    # 'twherohan',
    # 'namewee',
    # 'posenkid',
    # 'debatelo',
    # 'ETtodaySTAR',
    # 'NTDChinese',
    # 'CTimefans',
    # 'setnews',
    # 'noc.tpe',
    # 'thisgroupofpeople',
    # 'YESRANGER',
    'RayDuEnglish',
    'WithGaLoveTaiwan',
    'jay',
    'JoemanStation',
    'Muyao4',
    'etman0909',
    'amogood2.0',
    'ruruspiano',
    'HuangHuangBrother',
    'Chienseating',
    'fumeancats',
    'Wackyboys.Fans',
]
text = []
reactions = { 'like': [], 'love': [], 'care': [], 'haha': [], 'wow': [], 'sad': [], 'angry': [] }
for page in page_list:
    posts = get_posts(page, cookies='/home/iuthing/fb/cookie.json', pages=10, options={"allow_extra_requests": False})
    ids = []
    for post in posts:
        ids.append(str(post['post_id']))
    print(ids)
    gposts = get_posts(post_urls=ids, cookies='/home/iuthing/fb/cookie.json', options={"reactions": True})
    for post in gposts:
        try:
            text.append(post['post_text'])
        except Exception as e:
            print(e)
            continue
        for reaction in reactions:
            try:
                reactions[reaction].append(post["reactions"][reaction])
            except Exception as e:
                # print(e)
                reactions[reaction].append(0)
        # print('------------------------------------------------------------')
        # print(f'''Text: {post['post_text']}
        # Comments: {post["comments"]}
        # Likes: {post["likes"]}
        # Reactions: {post["reaction_count"]} ({post["reactions"]})''')
        # print('------------------------------------------------------------')
        # for comment in post["comments_full"]:
        #     print(f'comment text: {comment["comment_text"]}')
        #     print(f'comment reactions: {comment["comment_reactions"]}')
        #     print('------------------------------------------------------------')
        print(page)
        print(text[-1])
        for r in reactions:
            print(f'{r}: {reactions[r][-1]}')
        data = {'text': text, **reactions}
        df = pd.DataFrame(data)
        df.to_csv('fb_posts.csv')