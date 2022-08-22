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
#include <iostream>
#define IMGFILE(x) #x ".jpg"
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
    if (img) delete img;
    //cv::resize(img_cv, img_cv, cv::Size(1280, 720));
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
    //std::cout << w << ", " << h << std::endl;

	  img = mat;
    _width = w;
    _height = h;
  }

  void load_mat32f(Mat32f LRimg) {
      //if (img)
      //  return;
      Mat32f *mat = new Mat32f{LRimg.clone()};
  //     Mat32f *mat = new Mat32f(LRimg.height(), LRimg.width(), 3);
  // #pragma omp parallel for schedule(dynamic)
  //     REP(i, LRimg.height())
  //       REP(j, LRimg.width()){
  //         mat->at(i, j, 0) = LRimg.at(i, j, 0);
  //         mat->at(i, j, 1) = LRimg.at(i, j, 1);
  //         mat->at(i, j, 2) = LRimg.at(i, j, 2);
  //       }
      delete img;
      
      img = mat;
      _width = img->width();
      _height = img->height();
  }

  void cropped(int startX, int startY, int width, int height){
    //std::cout << "1. " << startX + width << ", " << startY + height << std::endl;
    //std::cout << "2. "<< _width << ", " << _height << std::endl;
    if(startX + width > _width || startY + height > _height) error_exit("Failed to crop image\n");
    Mat32f *mat = new Mat32f(height, width, 3);
    // REP(i, left.height())
    // {
    //   float* dst = left.ptr(i, 0);
    //   const float* src = res.ptr(i);
    //   memcpy(dst, src, 3 * left.width() * sizeof(float));
    // }
    // std::cout << "123" << std::endl;
    // REP(i, height){
    //   float* dst = mat->ptr(i, 0);
    //   const float* src = img->ptr(i+startY);
    //   memcpy(dst, src, 3 * (width - startX) * sizeof(float));
    // }
    // std::cout << "456" << std::endl;
    // write_rgb(IMGFILE(out), mat);
    // std::cout << "789" << std::endl;
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
