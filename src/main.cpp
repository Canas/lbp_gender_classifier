#include <iostream>
#include <vector>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "LBP.h"

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