
from PIL import Image


def images_to_gif(image_fnames, fname):
    image_fnames.sort(key=lambda x: int(x.name.split('_')[-2])) #sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
               save_all=True, duration=DURATION, loop=0)



