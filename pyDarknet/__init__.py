from detector import Darknet_ObjectDetector as ObjectDetector
from detector import BBox

if __name__ == '__main__':
    from PIL import Image
    det = ObjectDetector('../cfg/yolo.cfg','/media/ssd/models/yolo/yolo.weights')
    for i in xrange(4):
        rst, run_time = det.detect_object(Image.open('../data/dog.jpg'))

    print 'got {} objects in {} seconds'.format(len(rst), run_time)

    for bbox in rst:
        print '{} {} {} {} {} {}'.format(bbox.top, bbox.left, bbox.bottom, bbox.right, bbox.cls, bbox.confidence)
