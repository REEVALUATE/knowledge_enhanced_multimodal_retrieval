def combine_descriptions(metadata,content):

    # Simple concatenation with space
    if metadata and content:
        first_part = metadata.split(",")[0]
        if first_part.startswith("This is a"):
            o = first_part.split("This is a")[-1].strip()
            if o.lower() in content:
                hybrid = content + metadata.split(first_part)[-1].strip()
            else:
                hybrid = content + ". " + metadata
        elif first_part.startswith("A "):
            o = first_part.split("A ")[-1].strip()
            if o.lower() in content:
                hybrid = content + metadata.split(first_part)[-1].strip()
            else:
                hybrid = content + ". " + metadata
        else:
            hybrid = content + ". " + metadata
    elif metadata:
        # Only metadata available
        hybrid = metadata
    elif content:
        # Only content available
        hybrid = content
    else:
        # Both empty
        hybrid = ""
    # make the first letter uppercase
    if len(hybrid) > 0:
        hybrid = hybrid[0].upper() + hybrid[1:]

    if "painting" in content and ". This is a painting" in hybrid:
        hybrid = hybrid.replace(". This is a painting", ",")
    if "painting" in content and ". A painting" in hybrid:
        hybrid = hybrid.replace(". A painting", ",")
    if "church" in content and ". This is a church" in hybrid:
        hybrid = hybrid.replace(". This is a church", ",")
    if "church" in content and ". A church" in hybrid:
        hybrid = hybrid.replace(". A church", ",")
    if "temples" in content and ". This is a Temples" in hybrid:
        hybrid = hybrid.replace(". This is a Temples", ",")
    if "temples" in content and ". A Temples" in hybrid:
        hybrid = hybrid.replace(". A Temples", ",")

    return hybrid

import random
def random_select_content(content_descriptions):
    content1 = random.choice(content_descriptions)
    while "the church of the person" in content1 or len(content1) < 10:
        # remove this from content_descriptions
        content_descriptions.remove(content1)
        if content_descriptions == []:
            return "",""
        content1 = random.choice(content_descriptions)

    content_descriptions.remove(content1)

    content2 = random.choice(content_descriptions) if content_descriptions != [] else ""
    while "the church of the person" in content2 or len(content2) < 10:
        # remove this from content_descriptions
        content_descriptions.remove(content2)
        if content_descriptions == []:
            return "",""
        content2 = random.choice(content_descriptions)
    return content1, content2

import os
metadata_uuid = [f.split(".")[0] for f in os.listdir('../ArtKB/texts/metadata_texts/')]
content_uuid = [f.split(".")[0] for f in os.listdir('../ArtKB/texts/content_texts/')]
image_uuid = [f.split(".")[0] for f in os.listdir('../ArtKB/images/')]

uuid_set = set(metadata_uuid) & set(content_uuid) & set(image_uuid)
uuids = list(uuid_set)

from tqdm import tqdm
import json
error_content = []
for uuid in tqdm(uuids):
    with open('../ArtKB/texts/metadata_texts/' + uuid + '.json', 'r') as f:
        metadata_data = json.load(f)
        target = metadata_data['metadata_descriptions']
        target = random.choice(target)
        uuid_now = metadata_data['uuid']

    with open('../ArtKB/texts/content_texts/' + uuid + '.json', 'r') as f:
        content_data = json.load(f)
        content_descriptions = content_data['content_descriptions']
        content_text =random.choice(content_descriptions)
        
    if content_text == "" or target == "":
        error_content.append(uuid)

    target = combine_descriptions(target, content_text)

    hybrid_data = {
        "uuid": uuid,
        "target_text": target
    }
    with open('../ArtKB/texts/texts_final/' + uuid + '.json', 'w') as f:
        json.dump(hybrid_data, f, indent=2, ensure_ascii=False)
