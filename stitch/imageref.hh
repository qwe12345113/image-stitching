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
  
  void load_opencv(cv::Mat img_cv){
    //print_debug("load opencv image");
    cv::cvtColor(img_cv, img_cv, cv::COLOR_BGR2RGBA);
	  unsigned w = img_cv.cols, h = img_cv.rows;
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
