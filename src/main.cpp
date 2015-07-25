#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <string.h>

using namespace std;
using namespace cv;

#define TRUE 1
#define FALSE 0

#ifdef __linux
template <typename T>
std::string to_string(T value){
	//create an output string stream
	std::ostringstream os ;
	//throw the value into the string stream
	os << value ;
	//convert the string stream into a string and return
	return os.str() ;
}
#endif

vector<Mat> loadSet(int setID, bool hair){
	string path;
	vector<Mat> output;

	string mPath, fPath;
	if(hair == TRUE){
		mPath = "preprocessed/MaleFinal2";
		fPath = "preprocessed/FemaleFinal2";
	}
	else{
		mPath = "preprocessed/MaleFinal";
		fPath = "preprocessed/FemaleFinal";
	}


	if (setID == 0){
		for (int i = 0; i < 140; i++){
			path = mPath + "/im_(" + to_string(i + 1) + ").jpg";
			//cout << path << endl;
			output.push_back(imread(path, 0));
		}
		for (int i = 0; i < 140; i++){
			path = fPath + "/if_(" + to_string(i + 1) + ").jpg";
			//cout << path << endl;
			output.push_back(imread(path, 0));
		}
	}

	else if (setID == 1){
		for (int i = 140; i < 200; i++){
			path = mPath + "/im_(" + to_string(i + 1) + ").jpg";
			output.push_back(imread(path, 0));
		}
		for (int i = 140; i < 200; i++){
			path = fPath + "/if_(" + to_string(i + 1) + ").jpg";
			output.push_back(imread(path, 0));
		}
	}

	return output;
}

vector<float> loadLabels(int setID){
	vector<float> output;

	if (setID == 0){
		for (int i = 0; i < 140; i++){
			output.push_back(1.0);
		}
		for (int i = 0; i < 140; i++){
			output.push_back(-1.0);
		}
	}

	else if (setID == 1){
		for (int i = 140; i < 200; i++){
			output.push_back(1.0);
		}
		for (int i = 140; i < 200; i++){
			output.push_back(-1.0);
		}
	}
	return output;
}

void showImages(vector<Mat> imArray){
	for (unsigned int i = 0; i < imArray.size(); i++)
		imshow("imagen" + to_string(i), imArray[i]);
}

Mat LBP(Mat input){
	Mat lbp((input.rows - 2), (input.cols - 2), CV_8U);
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

vector<int> initHistogram(){
	vector<int> bins(59, 0);
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

vector<int> histLBP(Mat input, vector<int> bins){
	vector<int> hist(59, 0);
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

vector<vector<int> > getFeatures(vector<Mat> imLBP){
	int m = imLBP[0].rows / 2;
	int n = imLBP[0].cols / 2;

	vector<int> bins = initHistogram();
	vector<vector<int> > features;

	for (unsigned int k = 0; k < imLBP.size(); k++){
		Mat quad1(m, n, CV_8U);
		Mat quad2(m, n, CV_8U);
		Mat quad3(m, n, CV_8U);
		Mat quad4(m, n, CV_8U);

		for (int i = 0; i < m; i++){
			for (int j = 0; j < n; j++){
				quad1.at<uchar>(i, j) = imLBP[k].at<uchar>(i, j);
				quad2.at<uchar>(i, j) = imLBP[k].at<uchar>(i, j+n);
				quad3.at<uchar>(i, j) = imLBP[k].at<uchar>(i+m, j);
				quad4.at<uchar>(i, j) = imLBP[k].at<uchar>(i+m, j+n);
			}
		}

		vector<int> histLBP_1 = histLBP(quad1, bins);
		vector<int> histLBP_2 = histLBP(quad2, bins);
		vector<int> histLBP_3 = histLBP(quad3, bins);
		vector<int> histLBP_4 = histLBP(quad3, bins);

		vector<int> feature;
		feature.reserve(histLBP_1.size() + histLBP_2.size() + histLBP_3.size() + histLBP_4.size());
		feature.insert(feature.end(), histLBP_1.begin(), histLBP_1.end());
		feature.insert(feature.end(), histLBP_2.begin(), histLBP_2.end());
		feature.insert(feature.end(), histLBP_3.begin(), histLBP_3.end());
		feature.insert(feature.end(), histLBP_4.begin(), histLBP_4.end());

		features.push_back(feature);
	}

	return features;
}


int main(int argc, char *argv[]){
	int kernel_id;
	bool hasHair;
	
	if(argc == 3){
		hasHair = atoi(argv[2]);
		kernel_id = atoi(argv[1]);
	}
	else if(argc == 2){
		kernel_id = atoi(argv[1]);
	}
	else{
		kernel_id = 1;
		hasHair = FALSE;
	}

	// Cargar imagenes y labels
	vector<Mat> trainImages = loadSet(0, hasHair);
	vector<float> trainLabels = loadLabels(0);

	vector<Mat> testImages = loadSet(1, hasHair);
	vector<float> testLabels = loadLabels(1);
	
	// Obtener labels en formato adecuado
	float trainLabels_array[280];
	for (unsigned int i = 0; i < trainLabels.size(); i++){
		trainLabels_array[i] = trainLabels[i];
	}

	float testLabels_array[120];
	for (unsigned int i = 0; i < testLabels.size(); i++){
		testLabels_array[i] = testLabels[i];
	}

	Mat trainLabels_Mat(280, 1, CV_32FC1, trainLabels_array);
	Mat testLabels_Mat(120, 1, CV_32FC1, testLabels_array);
	
	// Obtener Imagenes LBP
	vector<Mat> trainImagesLBP;
	vector<Mat> testImagesLBP;
	//imwrite("LBP_im1.jpg",LBP(trainImages[0]));
	for (unsigned int i = 0; i < trainImages.size(); i++)
		trainImagesLBP.push_back(LBP(trainImages[i]));
	for (unsigned int i = 0; i < testImages.size(); i++)
		testImagesLBP.push_back(LBP(testImages[i]));

	// Obtener vectores de caracteristicas
	vector<vector<int> > trainFeatures = getFeatures(trainImagesLBP);
	vector<vector<int> > testFeatures = getFeatures(testImagesLBP);

	float trainData[280][236];
	float testData[120][236];

	for (int i = 0; i < 280; i++){
		for (int j = 0; j < 236; j++){
			trainData[i][j] = (float)trainFeatures[i][j];
		}
	}

	for (int i = 0; i < 120; i++){
		for (int j = 0; j < 236; j++){
			testData[i][j] = (float)testFeatures[i][j];
		}
	}

	Mat trainData_Mat(280, 236, CV_32FC1, trainData);
	Mat testData_Mat(120, 236, CV_32FC1, testData);


	// Configurar SVM
	CvSVMParams params;
	switch(kernel_id){
		case 1: // LINEAL
			cout << "Kernel: LINEAR" << endl;
			params.svm_type = CvSVM::C_SVC;
			params.kernel_type = CvSVM::LINEAR;
			params.C = 0.1;
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-7);
			break;
		case 2: // POLINOMIAL
			cout << "Kernel: POLY" << endl;
			params.svm_type = CvSVM::C_SVC;
			params.kernel_type = CvSVM::POLY;
			params.gamma = 1e-5;
			params.degree = 0.49;//d1=5 (0.25), d2=5 (0.133), d3=3 (0.183)// 50:  d1=1 (0.23), d2=3 (0.1583), d3=6 (0.225)
			params.coef0 = 274.4;
			params.C = 62.5;
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-7);
			break;
		case 3: // RBF
			cout << "Kernel: RBF" << endl;
			params.svm_type = CvSVM::C_SVC;
			params.kernel_type = CvSVM::RBF;
			params.gamma = 1;
			params.C = 2.5;
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-7);
			break;
		case 4: // SIGMOIDAL
			cout << "Kernel: SIGMOIDAL" << endl;
			params.svm_type = CvSVM::C_SVC;
			params.kernel_type = CvSVM::SIGMOID;
			params.gamma = 1e-5;
			params.coef0 = 19.6;
			params.C = 0.1;
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-7);
	}


	// Entrenar 
	CvSVM SVM;
	SVM.train(trainData_Mat, trainLabels_Mat, Mat(), Mat(), params); //Defaul LINEAL
	
	//CvSVMParams params; // DESCOMENTAR PARA ENCONTRAR PARAMETROS OPTIMOS SEGUN KERNEL
	//SVM.train_auto(trainData_Mat, trainLabels_Mat, Mat() , Mat(), params);

	// Testear
	//SVM.predict(sampleMat, CvMat* results)
	float response;
	double error = 0.0;
	vector<float> aux;
	for (int i = 0; i<120; i++){
		response = SVM.predict(testData_Mat.row(i)); aux.push_back(response);
		if (testLabels_array[i] != response)
			error = error + 1;
		//cout << response << endl;
	}
	error = error / 120.0;
	cout << "Error rate: " << error << endl;
	cout << "Hit rate: " << 1 - error << endl;

	cvWaitKey(0);
	return 0;
}