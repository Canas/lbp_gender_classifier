#include "LBP.h"

cv::Mat LBP(cv::Mat input){
	cv::Mat lbp((input.rows - 2), (input.cols - 2), CV_8U);
	int center;
	int BIT[8];
	for (int m = 0; m<(input.rows - 2); m++){
		for (int n = 0; n<(input.cols - 2); n++){
			center = (input.at<uchar>(m + 1, n + 1));
			if (input.at<uchar>(m, n)>center)
				BIT[0] = 1;
			else
				BIT[0] = 0;
			if (input.at<uchar>(m, n + 1) >= center)
				BIT[1] = 1;
			else
				BIT[1] = 0;
			if (input.at<uchar>(m, n + 2) >= center)
				BIT[2] = 1;
			else
				BIT[2] = 0;
			if (input.at<uchar>(m + 1, n + 2) >= center)
				BIT[3] = 1;
			else
				BIT[3] = 0;
			if (input.at<uchar>(m + 2, n + 2) >= center)
				BIT[4] = 1;
			else
				BIT[4] = 0;
			if (input.at<uchar>(m + 2, n + 1) >= center)
				BIT[5] = 1;
			else
				BIT[5] = 0;
			if (input.at<uchar>(m + 2, n) >= center)
				BIT[6] = 1;
			else
				BIT[6] = 0;
			if (input.at<uchar>(m + 1, n) >= center)
				BIT[7] = 1;
			else
				BIT[7] = 0;
			lbp.at<uchar>(m, n) = (int)(BIT[7] + 2 * BIT[6] + 4 * BIT[5] + 8 * BIT[4] + 16 * BIT[3] + 32 * BIT[2] + 64 * BIT[1] + 128 * BIT[0]);
		}
	}
	return lbp;
}

std::vector<int> initHistogram(){
	std::vector<int> bins(59, 0);
	const int BIT[8] = { 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80 };

	int val[8];
	int index = 0;
	int cont = 0;

	for (int i = 0; i < 256; i++){
		for (int k = 0; k < 8; k++){
			if (i & BIT[k])
				val[k] = 1;
			else
				val[k] = 0;
		}

		if (val[7] != val[0]){ cont++; }
		if (val[0] != val[1]){ cont++; }
		if (val[1] != val[2]){ cont++; }
		if (val[2] != val[3]){ cont++; }
		if (val[3] != val[4]){ cont++; }
		if (val[4] != val[5]){ cont++; }
		if (val[5] != val[6]){ cont++; }
		if (val[6] != val[7]){ cont++; }

		if (cont < 3){
			bins[index] = i;
			//cout << "bins[" << index << "] = " << i << endl;
			index++;
		}

		cont = 0;
	}
	return bins;
}

std::vector<int> histLBP(cv::Mat input, std::vector<int> bins){
	std::vector<int> hist(59, 0);
	uchar pixel;
	bool isUniform = false;

	for (int i = 0; i < input.rows; i++){
		for (int j = 0; j < input.cols; j++){
			pixel = input.at<uchar>(i, j);
			for (int k = 0; k < 58; k++){
				if (pixel == bins[k]){
					hist[k]++;
					isUniform = true;
				}
			}
			if (!isUniform){
				hist[58]++;
			}
			isUniform = false;
		}
	}

	return hist;
}

std::vector<std::vector<int> > getFeatures(std::vector<cv::Mat> imLBP){
	int m = imLBP[0].rows / 2;
	int n = imLBP[0].cols / 2;

	std::vector<int> bins = initHistogram();
	std::vector<std::vector<int> > features;

	for (unsigned int k = 0; k < imLBP.size(); k++){
		cv::Mat quad1(m, n, CV_8U);
		cv::Mat quad2(m, n, CV_8U);
		cv::Mat quad3(m, n, CV_8U);
		cv::Mat quad4(m, n, CV_8U);

		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++){
				quad1.at<uchar>(i, j) = imLBP[k].at<uchar>(i, j);
				quad2.at<uchar>(i, j) = imLBP[k].at<uchar>(i, j+n);
				quad3.at<uchar>(i, j) = imLBP[k].at<uchar>(i+m, j);
				quad4.at<uchar>(i, j) = imLBP[k].at<uchar>(i+m, j+n);
			}
		}

		std::vector<int> histLBP_1 = histLBP(quad1, bins);
		std::vector<int> histLBP_2 = histLBP(quad2, bins);
		std::vector<int> histLBP_3 = histLBP(quad3, bins);
		std::vector<int> histLBP_4 = histLBP(quad3, bins);

		std::vector<int> feature;
		feature.reserve(histLBP_1.size() + histLBP_2.size() + histLBP_3.size() + histLBP_4.size());
		feature.insert(feature.end(), histLBP_1.begin(), histLBP_1.end());
		feature.insert(feature.end(), histLBP_2.begin(), histLBP_2.end());
		feature.insert(feature.end(), histLBP_3.begin(), histLBP_3.end());
		feature.insert(feature.end(), histLBP_4.begin(), histLBP_4.end());

		features.push_back(feature);
	}

	return features;
}