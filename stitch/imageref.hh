//File: imageref.hh
//Author: Yuxin Wu

#pragma once
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "lib/mat.h"
#include "lib/imgproc.hh"
#include "match_info.hh"
#include "common/common.hh"
using namespace std;
namespace pano {
// A transparent reference to a image in file
struct ImageRef {
  std::string fname;
  Mat32f* img = nullptr;
  int _width, _height;

  ImageRef(const std::string& fname): fname(fname) {}
  //ImageRef(const ImageRef& ) = delete;  // TODO make it work
  ~ImageRef() { if (img) delete img; }

  void load() {
    if (img) return;
    img = new Mat32f{read_img(fname.c_str())};
    _width = img->width();
    _height = img->height();
  }
  
  void load_mat32f(Mat32f LRimg) 
  {
    cout << "new0" << endl;
    // img = new Mat32f{LRimg};
    //auto test = new Mat32f(LRimg.rows(), LRimg.cols(), 3);
    //Mat32f img1 = Mat32f(LRimg.rows(), LRimg.cols(),3);
    //img = new Mat32f{img1};
    img = &LRimg;
    cout << "new1" << endl;

    // REP(i, LRimg.rows()){
    //   std::cout << "img->ptr" << std::endl;
    //   float* dst = img->ptr(i);
    //   std::cout << "LRimg.ptr" << std::endl;
		//   const float* src = LRimg.ptr(i);
    //   std::cout << "memcpy" << std::endl;
    //   memcpy(dst, src, sizeof(float) * LRimg.cols()*3);
    // }
    std::cout << "new0" << std::endl;

    _width = img->width();
    _height = img->height();
  }

  void load_opencv(cv::Mat img_cv){
    //print_debug("load opencv image");    
    cv::cvtColor(img_cv, img_cv, cv::COLOR_BGR2RGBA);    
	  unsigned w = img_cv.cols, h = img_cv.rows;
    cout << w << " " << h << endl;
	  Mat32f *mat = new Mat32f(h, w, 3); 
	  unsigned npixel = w * h;    
	  float* p = mat->ptr();
	  unsigned char* data = img_cv.data;
    
	  REP(i, npixel) {
		  *(p++) = (float)*(data++) / 255.0;
		  *(p++) = (float)*(data++) / 255.0;
		  *(p++) = (float)*(data++) / 255.0;
		  data++;	// rgba
	  }

	  img = mat;
    _width = w;
    _height = h;
  }

  void release() { if (img) delete img; img = nullptr; }

  int width() const { return _width; }
  int height() const { return _height; }
  Shape2D shape() const { return {_width, _height}; }

};

}
