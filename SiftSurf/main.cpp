#include <string>
#include <stdio.h>
#include <iostream>
#include "opencv/cv.h"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/core.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


//#define DEBUG 1
#define NO_IMAGES 100
#define NO_PERSPECTIVE 16
#define NO_SCALED 18
#define NO_ROTATED 18
#define NO_NOISED 11
#define NO_BLURED 10
#define _CRT_SECURE_NO_DEPRECATE

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

string pathReference = ("D:\\data\\original\\");

string pathPerspective = ("D:\\data\\modified\\perspectived\\");
string pathRotated = ("D:\\data\\modified\\rotated\\");
string pathScaled = ("D:\\data\\modified\\scaled\\");
string pathNoised = ("D:\\data\\modified\\noised\\");
string pathBlured = ("D:\\data\\modified\\blured\\");

string resultsPerspectiveSURF = ("D:\\data\\results\\resultsPerspectiveSURF.csv");
string resultsScaledSURF = ("D:\\data\\results\\resultsScaledSURF.csv");
string resultsRotatedSURF = ("D:\\data\\results\\resultsRotatedSURF.csv");
string resultsNoisedSURF = ("D:\\data\\results\\resultsNoisedSURF.csv");
string resultsBluredSURF = ("D:\\data\\results\\resultsBluredSURF.csv");

string resultsPerspectiveSURFTime = ("D:\\data\\results\\resultsPerspectiveSURFTime.csv");
string resultsScaledSURFTime = ("D:\\data\\results\\resultsScaledSURFTime.csv");
string resultsRotatedSURFTime = ("D:\\data\\results\\resultsRotatedSURFTime.csv");
string resultsNoisedSURFTime = ("D:\\data\\results\\resultsNoisedSURFTime.csv");
string resultsBluredSURFTime = ("D:\\data\\results\\resultsBluredSURFTime.csv");

string resultsPerspectiveSIFT = ("D:\\data\\results\\resultsPerspectiveSIFT.csv");
string resultsScaledSIFT = ("D:\\data\\results\\resultsScaledSIFT.csv");
string resultsRotatedSIFT = ("D:\\data\\results\\resultsRotatedSIFT.csv");
string resultsNoisedSIFT = ("D:\\data\\results\\resultsNoisedSIFT.csv");
string resultsBluredSIFT = ("D:\\data\\results\\resultsBluredSIFT.csv");

string resultsPerspectiveSIFTTime = ("D:\\data\\results\\resultsPerspectiveSIFTTime.csv");
string resultsScaledSIFTTime = ("D:\\data\\results\\resultsScaledSIFTTime.csv");
string resultsRotatedSIFTTime = ("D:\\data\\results\\resultsRotatedSIFTTime.csv");
string resultsNoisedSIFTTime = ("D:\\data\\results\\resultsNoisedSIFTTime.csv");
string resultsBluredSIFTTime = ("D:\\data\\results\\resultsBluredSIFTTime.csv");

int surfMatching(Mat img1, Mat img2)
{
	int minHessian = 600;

	Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector->detect(img1, keypoints_object);
	detector->detect(img2, keypoints_scene);

	Mat descriptors_object, descriptors_scene;

	detector->compute(img1, keypoints_object, descriptors_object);
	detector->compute(img2, keypoints_scene, descriptors_scene);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	for (int i = 0; i <  descriptors_object.rows; i++)
	{
		if (matches[i].distance <= 4 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(img1, keypoints_object, img2, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("image", WINDOW_NORMAL);
	imshow("image", img_matches);
	waitKey(0);

	return (int)good_matches.size();
}

int siftMatching(Mat img1, Mat img2)
{
	//Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create(0, 3, 0.125, 10);
	cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector->detect(img1, keypoints_object);
	detector->detect(img2, keypoints_scene);

	//Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();

	Mat descriptors_object, descriptors_scene;

	detector->compute(img1, keypoints_object, descriptors_object);
	detector->compute(img2, keypoints_scene, descriptors_scene);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);


	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance <= 4 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(img1, keypoints_object, img2, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("image", WINDOW_NORMAL);
	imshow("image", img_matches);
	waitKey(0);

	return (int)good_matches.size();
}

void matchRotated(bool isSURF)
{
	ofstream matchesRes;
	ofstream timeRes;
	

	//Если сравниваем по SURF
	if (isSURF)
	{
		matchesRes.open(resultsRotatedSURF);
		timeRes.open(resultsRotatedSURFTime);
	}
	else
	{
		matchesRes.open(resultsRotatedSIFT);
		timeRes.open(resultsRotatedSIFTTime);
	}

	matchesRes << "Picture № " << ";";
	timeRes << "Picture № " << ";";

	for (int j = 10; j <= 180; j = j + 10)
	{
		matchesRes << j << ";";
		timeRes << j << ";";
	}



	matchesRes << endl;
	timeRes << endl;
	
	cout << "Entered Func ROTATA" << endl;
	for (int i = 1; i <= NO_IMAGES; i++)
	{
		string str = (pathReference)+to_string(i) + ".jpg";

		Mat img_1 = imread(str, IMREAD_COLOR);

		int parameter = 10; //degrees
		
		matchesRes << i << ";";
		timeRes << i << ";";
		for (int j = 1; j <= NO_ROTATED; j++)
		{
			string str1 = (pathRotated)+to_string(i) + "_" + to_string(j) + ".jpg";
			cout << str << endl;
			cout << str1 << endl << endl;

			Mat img_2 = imread(str1, IMREAD_COLOR);

			int timer_before=0;
			int timer_after = 0;
			int matches=0;

			if (isSURF)
			{
				timer_before = getTickCount();
				matches = surfMatching(img_1, img_2);
				timer_after = getTickCount();
			}
			else
			{
				timer_before = getTickCount();
				matches = siftMatching(img_1, img_2);
				timer_after = getTickCount();
			}
			 		
			float time = (timer_after - timer_before) / (getTickFrequency() * 1.0000) / 2;
			
			matchesRes << matches << ";";
			timeRes << time << ";";
			cout << " Matches: " << matches << endl << endl;

			parameter += 10;
		}
		matchesRes << endl;
		timeRes << endl;
	}

	matchesRes.close();
	timeRes.close();
}

void matchScaled(bool isSURF)
{
	ofstream matchesRes;
	ofstream timeRes;


	//Если сравниваем по SURF
	if (isSURF)
	{
		matchesRes.open(resultsScaledSURF);
		timeRes.open(resultsScaledSURFTime);
	}
	else
	{
		matchesRes.open(resultsScaledSIFT);
		timeRes.open(resultsScaledSIFTTime);
	}

	matchesRes << "Picture № " << ";";
	timeRes << "Picture № " << ";";

	for (double j = 0.1; j <= 1; j += 0.05)
	{
		matchesRes << j << ";";
		timeRes << j << ";";
	}

	matchesRes << endl;
	timeRes << endl;

	cout << "Entered Func SCALALA" << endl;
	for (int i = 1; i <= NO_IMAGES; i++)
	{
		string str = (pathReference)+to_string(i) + ".jpg";

		Mat img_1 = imread(str, IMREAD_COLOR);

		int parameter = 10; //degrees

		matchesRes << i << ";";
		timeRes << i << ";";
		for (int j = 1; j <= NO_SCALED; j++)
		{
			string str1 = (pathScaled)+to_string(i) + "_" + to_string(j) + ".jpg";
			cout << str << endl;
			cout << str1 << endl << endl;

			Mat img_2 = imread(str1, IMREAD_COLOR);

			int timer_before = 0;
			int timer_after = 0;
			int matches = 0;

			if (isSURF)
			{
				try
				{
					timer_before = getTickCount();
					matches = surfMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch(Exception e)
				{
					matches = 0;
				}
			}
			else
			{
				try
				{
					timer_before = getTickCount();
					matches = siftMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch (Exception e)
				{
					matches = 0;
				}
			}

			float time = (timer_after - timer_before) / (getTickFrequency() * 1.0000) / 2;

			matchesRes << matches << ";";
			timeRes << time << ";";
			cout << " Matches: " << matches << endl << endl;

			parameter += 10;
		}
		matchesRes << endl;
		timeRes << endl;
	}

	matchesRes.close();
	timeRes.close();
}

void matchPerspectived(bool isSURF)
{
	ofstream matchesRes;
	ofstream timeRes;


	//Если сравниваем по SURF
	if (isSURF)
	{
		matchesRes.open(resultsPerspectiveSURF);
		timeRes.open(resultsPerspectiveSURFTime);
	}
	else
	{
		matchesRes.open(resultsPerspectiveSIFT);
		timeRes.open(resultsPerspectiveSIFTTime);
	}

	matchesRes << "Picture № " << ";";
	timeRes << "Picture № " << ";";

	for (float j = 0.15; j <= 1; j += 0.05)
	{
		matchesRes << j << ";";
		timeRes << j << ";";
	}

	matchesRes << endl;
	timeRes << endl;

	cout << "Entered Func PERSPECTIVA" << endl;
	for (int i = 26; i <= 26; i++)
	{
		string str = (pathReference)+to_string(i) + ".jpg";

		Mat img_1 = imread(str, IMREAD_COLOR);

		int parameter = 10; //degrees

		matchesRes << i << ";";
		timeRes << i << ";";
		for (int j = 1; j <= NO_PERSPECTIVE; j++)
		{
			string str1 = (pathPerspective)+to_string(i) + "_" + to_string(j) + ".jpg";
			cout << str << endl;
			cout << str1 << endl << endl;

			Mat img_2 = imread(str1, IMREAD_COLOR);

			int timer_before = 0;
			int timer_after = 0;
			int matches = 0;

			if (isSURF)
			{
				try
				{
					timer_before = getTickCount();
					matches = surfMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch (Exception e)
				{
					matches = 0;
				}
			}
			else
			{
				try
				{
					timer_before = getTickCount();
					matches = siftMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch (Exception e)
				{
					matches = 0;
				}
			}

			float time = (timer_after - timer_before) / (getTickFrequency() * 1.0000) / 2;

			matchesRes << matches << ";";
			timeRes << time << ";";
			cout << " Matches: " << matches << endl << endl;

			parameter += 10;
		}
		matchesRes << endl;
		timeRes << endl;
	}

	matchesRes.close();
	timeRes.close();
}

void matchNoised(bool isSURF)
{
	ofstream matchesRes;
	ofstream timeRes;


	if (isSURF)
	{
		matchesRes.open(resultsNoisedSURF);
		timeRes.open(resultsNoisedSURFTime);
	}
	else
	{
		matchesRes.open(resultsNoisedSIFT);
		timeRes.open(resultsNoisedSIFTTime);
	}

	matchesRes << "Picture № " << ";";
	timeRes << "Picture № " << ";";

	for (float j = 0; j <= 300; j += 30)
	{
		matchesRes << j << ";";
		timeRes << j << ";";
	}

	matchesRes << endl;
	timeRes << endl;

	cout << "Entered Func NOISU" << endl;
	for (int i = 26; i <= 26; i++)
	{
		string str = (pathReference)+to_string(i) + ".jpg";

		Mat img_1 = imread(str, IMREAD_COLOR);

		int parameter = 10; //degrees

		matchesRes << i << ";";
		timeRes << i << ";";
		for (int j = 1; j <= NO_NOISED; j++)
		{
			string str1 = (pathNoised)+to_string(i) + "_" + to_string(j) + ".jpg";
			cout << str << endl;
			cout << str1 << endl << endl;

			Mat img_2 = imread(str1, IMREAD_COLOR);

			int timer_before = 0;
			int timer_after = 0;
			int matches = 0;

			if (isSURF)
			{
				try
				{
					timer_before = getTickCount();
					matches = surfMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch (Exception e)
				{
					matches = 0;
				}
			}
			else
			{
				try
				{
					timer_before = getTickCount();
					matches = siftMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch (Exception e)
				{
					matches = 0;
				}
			}

			float time = (timer_after - timer_before) / (getTickFrequency() * 1.0000) / 2;

			matchesRes << matches << ";";
			timeRes << time << ";";
			cout << " Matches: " << matches << endl << endl;

			parameter += 10;
		}
		matchesRes << endl;
		timeRes << endl;
	}

	matchesRes.close();
	timeRes.close();
}

void matchBlured(bool isSURF)
{
	ofstream matchesRes;
	ofstream timeRes;


	if (isSURF)
	{
		matchesRes.open(resultsBluredSURF);
		timeRes.open(resultsBluredSURFTime);
	}
	else
	{
		matchesRes.open(resultsBluredSIFT);
		timeRes.open(resultsBluredSIFTTime);
	}

	matchesRes << "Picture № " << ";";
	timeRes << "Picture № " << ";";

	for (int j = 1; j < 20; j = j + 2)
	{
		matchesRes << j << ";";
		timeRes << j << ";";
	}

	matchesRes << endl;
	timeRes << endl;

	cout << "Entered Func BLURA" << endl;
	for (int i = 26; i <= 26; i++)
	{
		string str = (pathReference)+to_string(i) + ".jpg";

		Mat img_1 = imread(str, IMREAD_COLOR);

		int parameter = 10; //degrees

		matchesRes << i << ";";
		timeRes << i << ";";
		for (int j = 1; j <= NO_BLURED; j++)
		{
			string str1 = (pathBlured)+to_string(i) + "_" + to_string(j) + ".jpg";
			cout << str << endl;
			cout << str1 << endl << endl;

			Mat img_2 = imread(str1, IMREAD_COLOR);

			int timer_before = 0;
			int timer_after = 0;
			int matches = 0;

			if (isSURF)
			{
				try
				{
					timer_before = getTickCount();
					matches = surfMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch (Exception e)
				{
					matches = 0;
				}
			}
			else
			{
				try
				{
					timer_before = getTickCount();
					matches = siftMatching(img_1, img_2);
					timer_after = getTickCount();
				}
				catch (Exception e)
				{
					matches = 0;
				}
			}

			float time = 0;
			if (isSURF)
				time = (timer_after - timer_before) / (getTickFrequency() * 1.0000);
			else
				time = (timer_after - timer_before) / (getTickFrequency() * 1.0000) / 2;

			matchesRes << matches << ";";
			timeRes << time << ";";
			cout << " Matches: " << matches << endl << endl;

			parameter += 10;
		}
		matchesRes << endl;
		timeRes << endl;
	}

	matchesRes.close();
	timeRes.close();
}

int main(int argc, char *argv[])
{
	cout << "It works! Finally" << endl;

	matchRotated(true);
	matchScaled(true);
	matchPerspectived(true);
	matchNoised(true);
	matchBlured(true);
	cout << "SURF matching completed" << endl;

	matchRotated(false);
	matchScaled(false);
	matchPerspectived(false);
	matchNoised(false);
	matchBlured(false);
	cout << "SIFT matching completed" << endl;
	
	cvWaitKey(0);
	return 0;
}
