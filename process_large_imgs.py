import os
from PIL import Image, ExifTags

def process(name):
    image_path = "data/train2014/"+name
    thumbnail_path = "data/gen_data/"+name
    with Image.open(image_path) as img:
        '''
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        if img._getexif():
            exif= dict(img._getexif().items())

            if exif[orientation] == 3:
                img=img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img=img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img=img.rotate(90, expand=True)'''
        # create a thumbnail from desired image
        # the thumbnail will have dimensions of the same ratio as before, capped by
        # the limiting dimension of max_dim
        #print(img.size)
        if img.size[0] >= 640 and img.size[1] >= 400 and img.mode == "RGB":
            area = (0,0,640,400)
            img = img.crop(area)
            print(img.size)
            img.thumbnail((320,200),Image.ANTIALIAS)
            # save the image under a new filename in thumbnails directory
            img.save(thumbnail_path)
def all():
    res = os.listdir("data/train2014/")
    for name in res:
        try:
            process(name)
        except OSError as ose:
            print(name)
            raise ose
all()
