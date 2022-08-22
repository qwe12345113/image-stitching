//File: blender.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "blender.hh"

#include <iostream>
#include "lib/config.hh"
#include "lib/imgproc.hh"
#include "lib/timer.hh"
#include "lib/geometry.hh"
using namespace std;
using namespace config;
#define IMGFILE(x) #x ".jpg"
namespace pano {

void LinearBlender::add_image(
			const Coor& upper_left,
			const Coor& bottom_right,
			ImageRef &img,
			std::function<Vec2D(Coor)> coor_func) {
	images.emplace_back(ImageToAdd{Range{upper_left, bottom_right}, img, coor_func});
	target_size.update_max(bottom_right);
}

Mat32f LinearBlender::run() {
	Mat32f target(target_size.y, target_size.x, 3);
	//print_debug("target: (%d, %d)\n", target.height(), target.width());
	//write_rgb(IMGFILE(target), target);
#define GET_COLOR_AND_W \
					Vec2D img_coor = img.map_coor(i, j); \
					if (img_coor.isNaN()) continue; \
					float r = img_coor.y, c = img_coor.x; \
					auto color = interpolate(*img.imgref.img, r, c); \
					if (color.x < 0) continue; \
					float	w = 0.5 - fabs(c / img.imgref.width() - 0.5); \
					if (not config::ORDERED_INPUT) /* blend both direction */\
						w *= (0.5 - fabs(r / img.imgref.height() - 0.5)); \
					color *= w

	if (LAZY_READ) {
		// Use weighted pixel, to iterate over images (and free them) instead of target.
		// Will be a little bit slower
		Mat<float> weight(target_size.y, target_size.x, 1);
		memset(weight.ptr(), 0, target_size.y * target_size.x * sizeof(float));
		fill(target, Color::BLACK);
#pragma omp parallel for schedule(dynamic)
		REP(k, (int)images.size()) {
			auto& img = images[k];
			img.imgref.load();
			auto& range = img.range;
			for (int i = range.min.y; i < range.max.y; ++i) {
				float *row = target.ptr(i);
				float *wrow = weight.ptr(i);
				for (int j = range.min.x; j < range.max.x; ++j) {
					GET_COLOR_AND_W;
					//#pragma omp critical
					{
						row[j*3] += color.x;
						row[j*3+1] += color.y;
						row[j*3+2] += color.z;
						wrow[j] += w;
					}
				}
			}
			img.imgref.release();
		}
#pragma omp parallel for schedule(dynamic)
		REP(i, target.height()) {
			auto row = target.ptr(i);
			auto wrow = weight.ptr(i);
			REP(j, target.width()) {
				if (wrow[j]) {
					*(row++) /= wrow[j]; *(row++) /= wrow[j]; *(row++) /= wrow[j];
				} else {
					*(row++) = -1; *(row++) = -1; *(row++) = -1;
				}
			}
		}
	} 
	else {
		//GuardedTimer tm("blend"); //test time
		fill(target, Color::NO);
/* 
test 
#pragma omp parallel for schedule(dynamic)
		for (auto& img : images){
			// print_debug("images.size(): (%ld, %ld, %ld, %ld)\n", 
 			// 		img.range.min.x, img.range.max.x, img.range.min.y, img.range.max.y);
		#pragma omp parallel for schedule(dynamic)
			for (int i = img.range.min.y; i < img.range.max.y; i ++) {				
				for (int j = img.range.min.x; j < img.range.max.x; j ++) {
					if (target.at(i,j, 0)==-1)
					// if (target.at(i,j, 0)==-1 || target.at(i,j,1)==-1 || target.at(i,j,2)==-1)
					{
						target.at(i, j, 0) = img.imgref.img->at(i, j-img.range.min.x, 0);
						target.at(i, j, 1) = img.imgref.img->at(i, j-img.range.min.x, 1);
						target.at(i, j, 2) = img.imgref.img->at(i, j-img.range.min.x, 2);
					}
					else
					{
						target.at(i, j, 0) = (target.at(i, j, 0) + img.imgref.img->at(i, j-img.range.min.x, 0)) / 2;
						target.at(i, j, 1) = (target.at(i, j, 1) + img.imgref.img->at(i, j-img.range.min.x, 1)) / 2;
						target.at(i, j, 2) = (target.at(i, j, 2) + img.imgref.img->at(i, j-img.range.min.x, 2)) / 2;
					}
				}
			}
		}
		write_rgb(IMGFILE(target123), target);
*/
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < target.height(); i ++) {
			float *row = target.ptr(i);
			for (int j = 0; j < target.width(); j ++) {
				Color isum = Color::BLACK;
				float wsum = 0;
				for (auto& img : images) // 幾張img loop幾次
				{					
					// print_debug("img: (%d)\n", img.imgref.img);
					// print_debug("images.size(): (%ld, %ld, %ld, %ld)\n", 
					// img.range.min.x, img.range.max.x, img.range.min.y, img.range.max.y);
					if (img.range.contain(i, j))//確認該點是否在img範圍內
					{
						GET_COLOR_AND_W;
						isum += color;			
						wsum += w;			
					}
				}
				if (wsum > 0)	// keep original Color::NO
					// print_debug("row: (%f)\n", *row);
					// print_debug("row + j * 3: (%f)\n", row + j * 3);
					(isum / wsum).write_to(row + j * 3);
			}
		}
	}
	//write_rgb(IMGFILE(target1), target);
	//print_debug("target: (%d, %d)\n", target.height(), target.width());
	return target;
}

}
