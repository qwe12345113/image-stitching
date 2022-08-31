//File: imageref.hh
//Author: Yuxin Wu

#pragma once
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "lib/mat.h"
#include "lib/imgproc.hh"
#include "lib/timer.hh"
#include "match_info.hh"
#include "common/common.hh"
#include <iostream>
namespace pano {
// A transparent reference to a image in file
struct ImageRef {
  std::string fname;
  Mat32f* img = nullptr;
  Matuc* imguc = nullptr;
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
    if (img) delete img;
    cv::cvtColor(img_cv, img_cv, cv::COLOR_BGR2RGBA);
	  unsigned w = img_cv.cols, h = img_cv.rows;
	  Mat32f *mat = new Mat32f(h, w, 3);

	  unsigned npixel = w * h;
	  float* p = mat->ptr();
	  unsigned char* data = img_cv.data;

	  REP(i, npixel) {
		  *(p++) = *(data++) / 255.0;
		  *(p++) = *(data++) / 255.0;
		  *(p++) = *(data++) / 255.0;
		  data++;	// rgba
	  }

	  img = mat;
    _width = w;
    _height = h;
  }
  
  void load_opencv_uc(cv::Mat img_cv, int shift) {
    if (imguc) delete imguc;
    cv::cvtColor(img_cv, img_cv, cv::COLOR_BGR2RGB);
    int w = img_cv.cols-2*shift, h = img_cv.rows;
    Matuc *mat = new Matuc(h, w, 3);

    REP(i, h){
      unsigned char* dst = mat->ptr(i, 0);
	    unsigned char* src = img_cv.ptr(i, shift);
	    memcpy(dst, src, 3 * w * sizeof(unsigned char));
    }

    imguc = mat;
    _width = w;
    _height = h;
  }
  void load_mat32f(Mat32f LRimg) {
      release();
      img = new Mat32f{LRimg.clone()};;
      _width = img->width();
      _height = img->height();
  }

  void load_matuc(Matuc LRimg) {
	  if (imguc) delete imguc;
      Matuc *mat = new Matuc{LRimg.clone()};
	  imguc = mat;
	  _width = imguc->width();
	  _height = imguc->height();
  }
  
  void cropped(int startX, int startY, int width, int height){

    if(startX + width > _width || startY + height > _height) error_exit("Failed to crop image\n");
    Mat32f *mat = new Mat32f(height, width, 3);
  #pragma omp parallel for schedule(dynamic) 
    REP(i, height)
			REP(j, width) {
        mat->at(i, j, 0) = img->at(i+startY, j+startX, 0);
				mat->at(i, j, 1) = img->at(i+startY, j+startX, 1);
				mat->at(i, j, 2) = img->at(i+startY, j+startX, 2);
      }
    delete img;
    img = mat;
    _width = width;
    _height = height;
  }

  void release() { if (img) delete img; img = nullptr; }

  int width() const { return _width; }
  int height() const { return _height; }
  Shape2D shape() const { return {_width, _height}; }

};

}
