//File: cylstitcher.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "cylstitcher.hh"

#include "lib/timer.hh"
#include "lib/config.hh"
#include "lib/imgproc.hh"
#include "feature/matcher.hh"
#include "transform_estimate.hh"
#include "blender.hh"
#include "match_info.hh"
#include "warp.hh"

#include <opencv2/opencv.hpp>

using namespace config;
using namespace std;

namespace pano {

const static char* HOMOGRAPHY_DUMP = "parameter";
const static char* HOMOGRAPHY_DUMP2 = "parameter2";

bool once = true;
bool once1 = true;
cv::Mat map[2];

void CylinderStitcher::Calibrate(){
	float K[3][3] = {439.8081431898627, -1.11273685072125, 657.1262692712136, 0.0, 438.94230771583267, 398.84726805778035, 0., 0., 1.};
	float D[4][1] = {-0.05426531284660334, 0.042009650323603445, -0.023596004020591154, 0.0046584199928515774};
	cv::Mat camMat = cv::Mat(3,3,CV_32FC1, K);
	cv::Mat distortion = cv::Mat(4, 1, CV_32FC1, D);
	cv::fisheye::initUndistortRectifyMap(
		camMat, distortion, cv::Matx33d::eye(), camMat, cv::Size(1280, 800), CV_16SC2, map[0], map[1]
	);
}

Mat32f CylinderStitcher::build() {
    if(LOADHOMO){
        REP(k, (int)imgs.size()){
    		imgs[k].load();
        }

		bundle.identity_idx = imgs.size() >> 1;
        bundle.load_homography(HOMOGRAPHY_DUMP);
    }
    else{
        calc_feature();
        bundle.identity_idx = imgs.size() >> 1;
	    build_warp();
		bundle.save_homography(HOMOGRAPHY_DUMP);
	    free_feature();
    }
	std::cout << "blend" << std::endl;
	bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
	bundle.update_proj_range();
	auto ret = bundle.blend();
	return ret;
}

Mat32f CylinderStitcher::build_new() {Mat32f tmp; return tmp;}

Mat32f CylinderStitcher::build_stream(int shift) {
	cv::Mat tmp[(int)imgs.size()];
	cv::Mat undistortImg[(int)imgs.size()];
	//GuardedTimer tm("build stream");	
#pragma omp parallel for schedule(dynamic)
	REP(i, (int)imgs.size()){
		if(!caps[i].read(tmp[i]))
			std::cout << "No frame to show, program stop." << std::endl;
		if(FISHEYE){
			cv::remap(tmp[i], undistortImg[i], map[0], map[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
			imgs[i].load_opencv(undistortImg[i]);
		}
		else	
			imgs[i].load_opencv(tmp[i]);
		imgs[i].cropped(shift, 0, imgs[i].width() - shift*2,imgs[i].height());
	}
	if(once){
		bundle.identity_idx = imgs.size() >> 1;
		bundle.load_homography(HOMOGRAPHY_DUMP);
		bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
		bundle.update_proj_range();
		once = false;
	}
	return bundle.blend();
	
	//original code. if don't care y-axis and rotation, it is not needed.
	//auto ret = bundle.blend();
	//return perspective_correction(ret);
}

Matuc CylinderStitcher::build_stream_uc(int shift) {
	cv::Mat tmp[(int)imgs.size()];
	cv::Mat undistortImg[(int)imgs.size()];
#pragma omp parallel for schedule(dynamic)
	REP(i, (int)imgs.size()){
		caps[i] >> tmp[i];
		if(FISHEYE){
			cv::remap(tmp[i], undistortImg[i], map[0], map[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
			imgs[i].load_opencv_uc(undistortImg[i], shift);
		}
		else
			imgs[i].load_opencv_uc(tmp[i], shift);
    }
    if(once){
        bundle.identity_idx = imgs.size() >> 1;
        bundle.load_homography(HOMOGRAPHY_DUMP);
        bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
        bundle.update_proj_range();
        once = false;
    }
    return bundle.blend_uc();
}

Mat32f CylinderStitcher::build_two_image(Mat32f right, Mat32f left) {
	
	//GuardedTimer tm("build twoImg");
    imgs[0].load_mat32f(right);
	imgs[1].load_mat32f(left);

    if(once1){
		bundle.identity_idx = imgs.size() >> 1;
		bundle.load_homography(HOMOGRAPHY_DUMP2);
		bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
		bundle.update_proj_range();
		once1 = false;
	}
	return bundle.blend();
	//auto ret = bundle.blend();
	//return perspective_correction(ret);
}

Matuc CylinderStitcher::build_two_image_uc(Matuc right, Matuc left) {
    imgs[0].load_matuc(right);
    imgs[1].load_matuc(left);
    if(once1){
        bundle.identity_idx = imgs.size() >> 1;
        bundle.load_homography(HOMOGRAPHY_DUMP2);
        bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
        bundle.update_proj_range();
        once1 = false;
    }
    return bundle.blend_uc();
}

bool CylinderStitcher::build_save(const char* filename, Mat32f& mat) {
	
    calc_feature();
    bundle.identity_idx = imgs.size() >> 1;
	if(!build_warp()){
		if(strcmp(filename, "parameter") == 0)
			crop_save();
		return false;
	}
	bundle.save_homography(filename);
	free_feature();
	bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
	bundle.update_proj_range();
	auto ret = bundle.blend();
	mat = perspective_correction(ret);
	if(strcmp(filename, "parameter") == 0)
		crop_save();
	return true;
}

Mat32f CylinderStitcher::build_load(const char* filename) {
	
    REP(k, (int)imgs.size()){
    	imgs[k].load();
    }

	bundle.identity_idx = imgs.size() >> 1;
    bundle.load_homography(filename);
	bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
	bundle.update_proj_range();
	auto ret = bundle.blend();
	return perspective_correction(ret);
}

bool CylinderStitcher::build_warp() {
	//GuardedTimer tm("build_warp()");
	int n = imgs.size(), mid = bundle.identity_idx;
	REP(i, n) bundle.component[i].homo = Homography::I();

	Timer timer;
	vector<MatchData> matches;		// matches[k]: k,k+1
	PairWiseMatcher pwmatcher(feats);
	matches.resize(n-1);
#pragma omp parallel for schedule(dynamic)
	REP(k, n - 1)
		matches[k] = pwmatcher.match(k, (k + 1) % n);
	//print_debug("match time: %lf secs\n", timer.duration());

	vector<Homography> bestmat;

	float minslope = numeric_limits<float>::max();
	float bestfactor = 1;
	if (n - mid > 1) {
		float newfactor = 1;
		// XXX: ugly
		float slope = update_h_factor(newfactor, minslope, bestfactor, bestmat, matches);
		if (bestmat.empty()){
			return false;
		}
			//error_exit("Failed to find hfactor");
		float centerx1 = 0, centerx2 = bestmat[0].trans2d(0, 0).x;
		float order = (centerx2 > centerx1 ? 1 : -1);
		REP(k, 3) {
			if (fabs(slope) < SLOPE_PLAIN) break;
			newfactor += (slope < 0 ? order : -order) / (5 * pow(2, k));
			slope = update_h_factor(newfactor, minslope, bestfactor, bestmat, matches);
		}
	}
	print_debug("Best hfactor: %lf\n", bestfactor);
	CylinderWarper warper(bestfactor);
	REP(k, n) imgs[k].load();
#pragma omp parallel for schedule(dynamic)
	REP(k, n) warper.warp(*imgs[k].img, keypoints[k]);

	// accumulate
	REPL(k, mid + 1, n) bundle.component[k].homo = move(bestmat[k - mid - 1]);
//#pragma omp parallel for schedule(dynamic)
	REPD(i, mid - 1, 0) {
		matches[i].reverse();
		MatchInfo info;
		bool succ = TransformEstimation(
				matches[i], keypoints[i + 1], keypoints[i],
				imgs[i+1].shape(), imgs[i].shape()).get_transform(&info);
		// Can match before, but not here. This would be a bug.
		if (! succ){
			REP(k, n) imgs[k].release();
			return false;
		}
			//error_exit(ssprintf("Failed to match between image %d and %d.", i, i+1));
		// homo: operate on half-shifted coor
		bundle.component[i].homo = info.homo;
	}
	REPD(i, mid - 2, 0)
		bundle.component[i].homo = bundle.component[i + 1].homo * bundle.component[i].homo;
	bundle.calc_inverse_homo();
	REP(k, n) imgs[k].release();
	return true;
}

float CylinderStitcher::update_h_factor(float nowfactor,
		float & minslope,
		float & bestfactor,
		vector<Homography>& mat,
		const vector<MatchData>& matches) {
	const int n = imgs.size(), mid = bundle.identity_idx;
	const int start = mid, end = n, len = end - start;

	vector<Shape2D> nowimgs;
	vector<vector<Vec2D>> nowkpts;
	REPL(k, start, end) {
		nowimgs.emplace_back(imgs[k].shape());
		nowkpts.push_back(keypoints[k]);
	}			// nowfeats[0] == feats[mid]

	CylinderWarper warper(nowfactor);
#pragma omp parallel for schedule(dynamic)
	REP(k, len)
		warper.warp(nowimgs[k], nowkpts[k]);

	vector<Homography> nowmat;		// size = len - 1
	nowmat.resize(len - 1);
	bool failed = false;
#pragma omp parallel for schedule(dynamic)
	REPL(k, 1, len) {
		MatchInfo info;
		bool succ = TransformEstimation(
				matches[k - 1 + mid], nowkpts[k - 1], nowkpts[k],
				nowimgs[k-1], nowimgs[k]).get_transform(&info);
		if (! succ)
			failed = true;
		//error_exit("The two image doesn't match. Failed");
		nowmat[k-1] = info.homo;
	}
	if (failed) return 0;

	REPL(k, 1, len - 1)
		nowmat[k] = nowmat[k - 1] * nowmat[k];	// transform to nowimgs[0] == imgs[mid]

	// check the slope of the result image
	Vec2D center2 = nowmat.back().trans2d(0, 0);
	const float slope = center2.y/ center2.x;
	print_debug("slope: %lf\n", slope);
	if (update_min(minslope, fabs(slope))) {
		bestfactor = nowfactor;
		mat = move(nowmat);
	}
	return slope;
}

Mat32f CylinderStitcher::perspective_correction(const Mat32f& img) {
	int w = img.width(), h = img.height();
	int refw = imgs[bundle.identity_idx].width(),
			refh = imgs[bundle.identity_idx].height();
	auto homo2proj = bundle.get_homo2proj();
	Vec2D proj_min = bundle.proj_range.min;

	vector<Vec2D> corners;
	auto cur = &(bundle.component.front());
	auto to_ref_coor = [&](Vec2D v) {
		v.x *= cur->imgptr->width(), v.y *= cur->imgptr->height();
		Vec homo = cur->homo.trans(v);
		homo.x /= refw, homo.y /= refh;
		Vec2D t_corner = homo2proj(homo);
		t_corner.x *= refw, t_corner.y *= refh;
		t_corner = t_corner - proj_min;
		corners.push_back(t_corner);
	};
	to_ref_coor(Vec2D(-0.5, -0.5));
	to_ref_coor(Vec2D(-0.5, 0.5));
	cur = &(bundle.component.back());
	to_ref_coor(Vec2D(0.5, -0.5));
	to_ref_coor(Vec2D(0.5, 0.5));

	// stretch the four corner to rectangle
	vector<Vec2D> corners_std = {
		Vec2D(0, 0), Vec2D(0, h),
		Vec2D(w, 0), Vec2D(w, h)};
	Matrix m = getPerspectiveTransform(corners, corners_std);
	Homography inv(m);

	LinearBlender blender;
	ImageRef tmp("this_should_not_be_used");
	tmp.img = new Mat32f(img);
	tmp._width = img.width(), tmp._height = img.height();
	blender.add_image(
			Coor(0,0), Coor(w,h), tmp,
			[=](Coor c) -> Vec2D {
		return inv.trans2d(Vec2D(c.x, c.y));
	});
	return blender.run();
}

void CylinderStitcher::crop_save() {
int shift = 5;
#pragma omp parallel for schedule(dynamic)
  	REP(k, (int)imgs.size()) {
		imgs[k].load();		
		imgs[k].cropped(shift, 0, imgs[k]._width-(shift*2), imgs[k]._height);
		//std::cout << imgs[k].fname << std::endl;
		write_rgb(imgs[k].fname, *imgs[k].img);
		//write_rgb(std::to_string(k+1) + ".png", *imgs[k].img);
		imgs[k].release();
  	}

}

}
