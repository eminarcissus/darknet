import sys
import glob
import os
from PIL import Image
import numpy as np
from cStringIO import StringIO
import pyDarknet
import base64
import cPickle
import time

src_path = sys.argv[1]
out_path = sys.argv[2]
if len(sys.argv) > 3:
    gpu = int(sys.argv[3])
else:
    gpu = 0

parts = glob.glob(os.path.join(src_path,'part-*'))

cnt = 0

#init detector
pyDarknet.ObjectDetector.set_device(gpu)
detector = pyDarknet.ObjectDetector('cfg/yolo.cfg','/media/ssd/models/yolo/yolo.weights')


for part in parts:
    rst_list = []
    start = time.time()
    for line in open(part):
        im_id = line.strip().split('\t')[0]
        im_data  = line.strip().split('\t')[-1]
        if im_data == 'N/A':
            continue
        im = Image.open(StringIO(base64.b64decode(im_data)))
        try:
            im.load()
        except IOError:
            pass

        if im.mode == 'L':
            im = im.convert('RGB')

        rst, rt = detector.detect_object(im)

        rst_list.append((im_id, rst))
        cnt += 1

        if cnt % 500 == 0:
            cur_time = time.time()
          
            print cnt, int(cur_time - start)
    
    part_name = part.split('/')[-1]+'.pc'
    out_name = os.path.join(out_path, part_name)
    cPickle.dump(open(out_name,'wb'), rst_list, cPickle.HIGHEST_PROTOCOL)
    print '{} finished'.format(part)




