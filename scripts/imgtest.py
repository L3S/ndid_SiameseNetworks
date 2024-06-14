from glob import glob
from struct import unpack
from tqdm import tqdm
import os
import tensorflow as tf
from sidd.utils.common import get_dataset


marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

DEFAULT_IMAGE_SIZE = (500, 500)

class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                raise TypeError("issue reading jpeg file")   


corrupted_jpegs = []

images = glob('/home/astappiev/nsir/*.jpg')
for image_path in tqdm(images):
  image = JPEG(image_path) 
  image_raw = tf.io.read_file(image_path)
  try:
    image.decode()
    decoded = tf.image.decode_jpeg(image_raw, channels=3, try_recover_truncated=True, acceptable_fraction=0.5)
    resized = tf.image.resize(decoded, DEFAULT_IMAGE_SIZE, method='nearest') 
  except:
    corrupted_jpegs.append(image_path)
    print(f"Corrupted image: {image_path}")

print(corrupted_jpegs)
#for image_path in corrupted_jpegs:
  #os.remove(os.path.join(root_img,image_path))

# /home/astappiev/nsir/datasets/mirflickr/images/5/59898.jpg
# /home/astappiev/nsir/datasets/mirflickr/images/10/104442.jpg
# /home/astappiev/nsir/datasets/mirflickr/images/10/107349.jpg
# /home/astappiev/nsir/datasets/mirflickr/images/10/108460.jpg
# /home/astappiev/nsir/datasets/mirflickr/images/68/686806.jpg
