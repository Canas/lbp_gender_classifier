#pragma once
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

cv::Mat LBP(cv::Mat input); // get LBP image from one input
std::vector<int> initHistogram(); // initialize histogram that will store LBP features
std::vector<int> histLBP(cv::Mat input, std::vector<int> bins); // fill the histogram initialized in initHistogram
std::vector<std::vector<int> > getFeatures(std::vector<cv::Mat> imLBP); // use LBP histogram to get LBP features described in reference 