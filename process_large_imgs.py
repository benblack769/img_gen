import os
from PIL import Image, ExifTags
import multiprocessing
import subprocess



def get_all_path_pairs(img_root,process_root):
    cmd = ["find", ".", "-type", "f"]
    out = subprocess.run(cmd,cwd=img_root,stdout=subprocess.PIPE).stdout
    outlines = out.decode("utf-8").split("\n")

    pair_paths = [((img_root+path),(process_root+path)) for path in outlines]
    return pair_paths


def crop_to_square(img):
    xsize,ysize = img.size
    if xsize < ysize:
        diffsize = ysize-xsize
        div2 = diffsize // 2
        area = (0,div2,xsize,xsize+div2)
        img = img.crop(area)
    elif xsize > ysize:
        diffsize = xsize-ysize
        div2 = diffsize // 2
        area = (div2,0,ysize+div2,ysize)
        img = img.crop(area)
    return img

def process(image_path,thumbnail_path):
    thumbnail_path = os.path.abspath(thumbnail_path)
    thumbnail_path = thumbnail_path.split(".")[0]+".jpg"
    os.makedirs(os.path.dirname(thumbnail_path),exist_ok=True)
    with Image.open(image_path) as img:
        '''for orientation in ExifTags.TAGS.keys():
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
        img = img.convert("RGB")
        IMG_SIZE = 96
        new_size = (IMG_SIZE,IMG_SIZE)
        img = crop_to_square(img)
        if img.size[0] > IMG_SIZE:
            img.thumbnail(new_size, Image.ANTIALIAS)
        else:
            img = img.resize(new_size, Image.BICUBIC)
        img.save(thumbnail_path,format="JPEG")

def proc_pair(pair):
    return process(pair[0],pair[1])

def all():
    #img_root = "/home/ben/Downloads/img_net/"
    #process_root = "/home/ben/fun_projs/img_gen/data/image_net_64/"
    img_root = "/home/ben/fun_projs/img_gen/data/train2014/"
    process_root = "/home/ben/fun_projs/img_gen/data/coco96/"
    all_path_pairs = get_all_path_pairs(img_root,process_root)
    #print(len(all_path_pairs))
    print(all_path_pairs[:5])
    #exit(0)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(proc_pair, all_path_pairs)
    return
    for pair in all_path_pairs:
        try:
            proc_pair(pair)
        except OSError as ose:
            print(pair)
            raise ose
all()
