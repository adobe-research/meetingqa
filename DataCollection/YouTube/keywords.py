import os
import sys
from fabian_crawler import *
from tqdm import tqdm

keywords = ["GitLab Unfiltered Meetings Playlists", "GitLab Unfiltered Meetings", 'Zoom Meeting Recording', 'Teams Meeting Recording', 'Zoom Class Recording', 'Virtual Panel Discussion', 'Virtual Round Table Discussion']

default_max_video_number = 5000

running_video_ids = []
running_keywords = []

for query in tqdm(keywords):
    if 'Playlists' in query: max_video_number = 200
    else: max_video_number = default_max_video_number
    print(query, max_video_number)
    keyword = query.replace(" ", "+")
    result, playlist, _ = get_ytb_video_list_by_search_selelium(keyword, max_search_result=max_video_number)
    if len(result) == 0:
        print('No videos found: {}'.format(query))
        sys.exit()
    output_filename = 'YouTubeIDs/{}.json'.format(query.replace(" ", '_'))
    video_ids = [l['video_id'] for l in result]

    for playlist_id in playlist:
        playlist_html = urlopen(get_playlist_url(playlist_id))
        playlist_videos = list(set(re.findall(r"watch\?v=(\S{11})", playlist_html.read().decode())))
        video_ids.extend(playlist_videos)

    video_ids = list(set(video_ids))
    video_ids = list(set(video_ids) - (set(video_ids) & set(running_video_ids)))
    running_video_ids.extend(video_ids)
    running_keywords.extend([query]*len(video_ids))
    assert len(running_video_ids) == len(running_keywords)

    with open(output_filename, 'w+') as fobj:
        fobj.write(json.dumps(video_ids))
    print('Query completed: {} - found {} videos and {} playlists'.format(query, len(video_ids), len(playlist)))

print('\n')
print('In total, found {} videos'.format(len(running_video_ids)))
output_filename = 'YouTubeIDs/{}.json'.format('combined-ids')
combined = [{'video-id':running_video_ids[i], 'query':running_keywords[i]} for i in range(len(running_video_ids))]
with open(output_filename, 'w+') as fobj:
        fobj.write(json.dumps(combined))

    
