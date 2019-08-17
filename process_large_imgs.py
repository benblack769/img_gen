import os
from PIL import Image, ExifTags

def process(name):
    image_path = "sawtooth_backpack/"+name
    thumbnail_path = "example_data/"+name
    with Image.open(image_path) as img:

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(img._getexif().items())

        if exif[orientation] == 3:
            img=img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img=img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img=img.rotate(90, expand=True)
        # create a thumbnail from desired image
        # the thumbnail will have dimensions of the same ratio as before, capped by
        # the limiting dimension of max_dim
        img.thumbnail((512,512),Image.ANTIALIAS)
        # save the image under a new filename in thumbnails directory
        img.save(thumbnail_path)
def all():
    res = os.listdir("sawtooth_backpack")
    for name in res:
        process(name)
all()
