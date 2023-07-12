import argparse
import json
import time
import random
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import uuid

def get_ytb_video_list_by_search_selelium(query, max_search_result=500, nretry=1, sleep=0.67):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(
        executable_path=ChromeDriverManager().install(),
        chrome_options=chrome_options)
    driver.maximize_window()

    #driver.get(f"https://www.youtube.com/results?search_query='{query.replace(' ', '+')}'&sp=CAMSAhAB")
    #driver.get("https://www.youtube.com/results?search_query=intitle%3A%22ZOOM+panel+discussion%22&sp=CAISAhgC")
    driver.get("https://www.youtube.com/results?search_query={}".format(query))

    nvideo = 0
    while True:
        driver.execute_script(
            "window.scrollTo(0,Math.max(document.documentElement.scrollHeight,document.body.scrollHeight,document.documentElement.clientHeight));"
        )
        content = driver.page_source.encode('utf-8')
        soup = BeautifulSoup(content, 'lxml')
        rst = soup.find_all('ytd-video-renderer')
        if max_search_result and len(rst) >= max_search_result:
            break
        if len(rst) <= nvideo:
            if nretry > 0:
                nretry -= 1
            else:
                break
        else:
            nvideo = len(rst)
        print(f'crawled {nvideo} videos')
        time.sleep(sleep+random.random())
    raw_html = driver.page_source.encode('utf-8')
    soup = BeautifulSoup(content, 'lxml')
    rst = soup.find_all('ytd-video-renderer')
    results = []
    for r in rst:
        try:
            results.append(parse_html(r, query))
        except:
            continue
    
    return results, raw_html

def save_results(driver):
    return results, raw_html

# def parse_html(node, query):
#     thumbnail_dct = get_thumbnail(node)
#     title, video_id = get_title(node)
#     channel_dct = get_channel_info(node)
#     ret = dict()
#     ret.update(channel_dct)
#     ret.update(dict(video_id=video_id, title=title, thumbnail=thumbnail_dct, query=query))
#     return ret

def parse_html(node, query):
    title, video_id = get_title(node)
    ret = {'title': title, 'video_id': video_id}
    return ret

def get_thumbnail(node):
    thumbnail = node.find('ytd-thumbnail').find('img')
    src = thumbnail.get('src', None)
    width = thumbnail.get('width', None)
    thumbnail_dct = dict(url=src, width=width)
    return thumbnail_dct

def get_title(node):
    ele = node.find('a', id='video-title')
    title = ele.get('title', None)
    video_id = ele.get('href', None)
    if video_id is not None:
        video_id = video_id[-11:]
    return title, video_id

def get_channel_info(node):
    channel_scope = node.find('ytd-video-meta-block').find(
        'ytd-channel-name').find('yt-formatted-string')
    channel_url = channel_scope.a.get('href', None)
    channel_name, channel_id = None, None
    if channel_url and channel_url.startswith('/user'):
        channel_name = channel_url[6:]
    if channel_url and channel_url.startswith('/channel'):
        channel_id = channel_url[9:]
    return dict(
        channel_url=channel_url,
        channel_id=channel_id,
        channel_name=channel_name)

if __name__ == '__main__':
    description = 'Runs a YouTube search request for a given text query.'
    parser = argparse.ArgumentParser(description=description)
    # Text based query as input.
    parser.add_argument('--query', type=str, default='intitle%3A"-Meeting+Recording"&sp=CAI%253D',
                        help='Text based search query.')
    # parser.add_argument('--output-filename', type=str, default='search_results/cats.json')
    parser.add_argument('--max_video_number', type=int, default=25000)
    args = parser.parse_args()
    keyword = args.query.replace(" ", "+")
    result, _ = get_ytb_video_list_by_search_selelium(keyword, max_search_result=args.max_video_number)
    if len(result) == 0:
        print('No videos found: {}'.format(args.query))
        sys.exit()
    output_filename = 'results/{}.json'.format(str(uuid.uuid4()))
    # with open(output_filename, 'w') as fobj:
    #     fobj.write(json.dumps(result))
    print('Query completed: {} - found {} videos'.format(args.query, len(result)))
    import pdb; pdb.set_trace()