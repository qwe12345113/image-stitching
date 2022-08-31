//File: stitcherbase.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "stitcherbase.hh"
#include "lib/timer.hh"
#include "lib/imgproc.hh"
#include <string>

namespace pano {

void StitcherBase::calc_feature() {
  GuardedTimer tm("calc_feature()");
  feats.resize(imgs.size());
  keypoints.resize(imgs.size());
  // detect feature
#pragma omp parallel for schedule(dynamic)
  REP(k, (int)imgs.size()) {
    imgs[k].load();
    feats[k] = feature_det->detect_feature(*imgs[k].img);
    if (config::LAZY_READ)
      imgs[k].release();
    if (feats[k].size() == 0)
      error_exit(ssprintf("Cannot find feature in image %d!\n", k));
    print_debug("Image %d has %lu features\n", k, feats[k].size());
    keypoints[k].resize(feats[k].size());
    REP(i, feats[k].size()){
      keypoints[k][i] = feats[k][i].coor;
      }
  }
  std::cout << std::endl;
}

void StitcherBase::load_stream(int number, char*argv[]) {
	REPL(i, 0, number) {
    caps[i].open(argv[i+2]);
	}
}

void StitcherBase::load_camera(int number) {
	
	//std::string sub = ".webm";
  std::cout << "load camera" << std::endl;
  /*caps[0].open(0);
  caps[1].open(1);
  caps[2].open(2);
  caps[3].open(3);
  */
  caps[0].open(0, cv::CAP_V4L2);
  caps[1].open(3, cv::CAP_V4L2);
  caps[2].open(1, cv::CAP_V4L2);
  caps[3].open(2, cv::CAP_V4L2);
	for(int i = 0;i < number;i++){
    caps[i].set(cv::CAP_PROP_FPS, 30);
    caps[i].set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    caps[i].set(cv::CAP_PROP_FRAME_HEIGHT, 800);
    caps[i].set(cv::CAP_PROP_CONVERT_RGB, 1);
	}
  std::cout << "load camera end" << std::endl;
}

void StitcherBase::save_image(int number) {
  
  cv::Mat tmp[number];
  for(int i = 0;i < number;i++){
    caps[i].read(tmp[i]);
    cv::imwrite(imgs[i].fname, tmp[i]);
	}
}

void StitcherBase::free_feature() {
  feats.clear(); feats.shrink_to_fit();  // free memory for feature
  keypoints.clear(); keypoints.shrink_to_fit();  // free memory for feature
}

}
