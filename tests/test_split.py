from src.yolo_prep.prep_positives import split_by_video, video_id

SITES = ["ESP", "IACO", "LOOI", "STRIK"]

# Makes annotations for 5 videos of 10 frames at each site
def make_annotations():
    return {
        f"{site}_{video:03d}_image_{frame:05d}.jpg": []
        for site in SITES
        for video in range(5)
        for frame in range(10)
    }

def videos_in(split):
    return {video_id(filename) for filename in split}

def sites_in(split):
    return {video.split("_")[0] for video in videos_in(split)}

# Check that double underscore filenames give the same video id as single
def test_video_id_handles_double_underscore():
    assert video_id("ESP_EK000027__image_00003.jpg") == "ESP_EK000027"
    assert video_id("ESP_EK000045_image_00031.jpg") == "ESP_EK000045"

# Check that adjacent frames of a video cannot leak between train and val
def test_no_video_in_both_splits():
    train, val = split_by_video(make_annotations())

    assert videos_in(train).isdisjoint(videos_in(val))

# Check that val measures every site, and train still learns from every site
def test_every_site_in_both_splits():
    train, val = split_by_video(make_annotations())

    assert sites_in(train) == set(SITES)
    assert sites_in(val) == set(SITES)

# Check that rebuilding the dataset gives the same split each time
def test_split_is_deterministic():
    first = split_by_video(make_annotations())
    second = split_by_video(make_annotations())

    assert first == second
