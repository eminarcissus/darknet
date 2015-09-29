#include <iostream>
#include <boost/python.hpp>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"

using namespace std;
namespace bp = boost::python;

string voc_class_names[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"};

struct BBox{
    int left;
    int right;
    int top;
    int bottom;
    float confidence;
    int cls;
};

class DarknetObjectDetector{
public:
    DarknetObjectDetector(bp::str cfg_name, bp::str weight_name){
      string cfg_c_name = string(((const char*)bp::extract<const char*>(cfg_name)));
      string weight_c_name = string(((const char*)bp::extract<const char*>(weight_name)));
      cout<<"loading network spec from"<<cfg_c_name<<'\n';
      net = parse_network_cfg((char*)cfg_c_name.c_str());

      cout<<"loading network weights from"<<weight_c_name<<'\n';
      load_weights(&net, (char*)weight_c_name.c_str());

      cout<<"network initialized!\n";
      layer = get_network_detection_layer(net);
      set_batch_network(&net, 1);srand(2222222);

      thresh = 0.2;
    };

    ~DarknetObjectDetector(){};

    bp::list detect_object(bp::str img_data, int img_width, int img_height, int img_channel){

      // preprocess input image
      const unsigned char* data = (const unsigned char*)((const char*)bp::extract<const char*>(img_data));

      assert(img_channel == 3);
      image im = make_image(img_width, img_height, img_channel);

      int cnt = img_height * img_channel * img_width;
      for (int i = 0; i < cnt; ++i){
        im.data[i] =(float)data[i]/255.;
      }

      image sized = resize_image(im, net.w, net.h);
      float *X = sized.data;
      float *predictions = network_predict(net, X);

      free_image(im);
      free_image(sized);


      return parse_yolo_detection(predictions, 7, layer.objectness,
                                  thresh, im.w, im.h);
    };

private:

    bp::list parse_yolo_detection(float *box, int side,
                              int objectness, float thresh,
                              int im_width, int im_height)
    {
      int classes = 20;
      int elems = 4+classes+objectness;
      int j;
      int r, c;

      bp::list ret_list = bp::list();

      for(r = 0; r < side; ++r){
        for(c = 0; c < side; ++c){
          j = (r*side + c) * elems;
          float scale = 1;
          if(objectness) scale = 1 - box[j++];
          int cls = max_index(box+j, classes);
          if(scale * box[j+cls] > thresh){
            //valid detection over threshold
            float conf = scale * box[j+cls];
            printf("%f %s\n", conf, voc_class_names[cls].c_str());

            j += classes;
            float x = box[j+0];
            float y = box[j+1];
            x = (x+c)/side;
            y = (y+r)/side;
            float w = box[j+2]; //*maxwidth;
            float h = box[j+3]; //*maxheight;
            h = h*h;
            w = w*w;

            int left  = (x-w/2)*im_width;
            int right = (x+w/2)*im_width;
            int top   = (y-h/2)*im_height;
            int bottom   = (y+h/2)*im_height;

            BBox bbox = {left, right, top, bottom, conf, cls};
            ret_list.append<BBox>(bbox);
          }
        }
      }

      return ret_list;
    }

    network net;
    detection_layer layer;
    float thresh;
};

BOOST_PYTHON_MODULE(libpydarknet)
{
  bp::class_<DarknetObjectDetector>("DarknetObjectDetector", bp::init<bp::str, bp::str>())
      .def("detect_object", &DarknetObjectDetector::detect_object);

  bp::class_<BBox>("BBox")
      .def_readonly("left", &BBox::left)
      .def_readonly("right", &BBox::right)
      .def_readonly("top", &BBox::top)
      .def_readonly("bottom", &BBox::bottom)
      .def_readonly("confidence", &BBox::confidence)
      .def_readonly("cls", &BBox::cls);
}