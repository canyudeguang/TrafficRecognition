#include <opencv\cv.h>
#include <opencv\highgui.h>

using namespace cv;

#define THRESHOLD_UNDER 100
#define THRESHOLD_UPPER 256
#define MORPH_SIZE 3	//Top-Hat Kernel Size
#define DISTANCE_WITH_LIGHT 60

/*///////////////////////////////////////////////////////////////////////////////////////////*/
					/* Morphology Setting */

int morph_size = MORPH_SIZE;
Mat element_tophat = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

/*///////////////////////////////////////////////////////////////////////////////////////////*/



/*///////////////////////////////////////////////////////////////////////////////////////////*/
					/* Labeling Setting */

int *label;
int labelWidth;
int labelHeight;
int objectN = 0;

struct Object	//Labeled Object
{
public:
	int count;
	int centerX;
	int centerY;
	int width;
	int height;
	bool checking;
	bool isDeleted;
	Object() : count(0), centerX(0), centerY(0), width(0), height(0), checking(false), isDeleted(false) {};
};

//Labeling Direction
const int dx[] = { +1, 0, -1, 0 };
const int dy[] = { 0, +1, 0, -1 };
/*///////////////////////////////////////////////////////////////////////////////////////////*/



/* 색깔끼리의 거리(유사도)를 체크하는 함수(유클리드 Norm사용) */
double calcDistance(int b1, int g1, int r1, int b2, int g2, int r2)
{
	return sqrt((b1 - b2)*(b1 - b2) + (g1 - g2)*(g1 - g2) + (r1 - r2)*(r1 - r2));
}

/* recursive로 구현된 Labeing함수 */
void labeling(int x, int y,int current_label, Mat& sourceImage) {
	if (x < 0 || x > sourceImage.cols - 1)
	{
		return;
	}// out of bounds
	if (y < 0 || y > sourceImage.rows - 1)
	{
		return;
	}// out of bounds
	if (label[y*sourceImage.cols + x] > 0 || sourceImage.at<uchar>(y, x) == 0)
	{
		return;
	}// already labeled or not marked with 1 in m

	// mark the current cell
	label[y*sourceImage.cols + x] = current_label;

	// recursively mark the neighbors
	for (int direction = 0; direction < 4; ++direction)
	{
		labeling(x + dx[direction], y + dy[direction], current_label, sourceImage);
	}
}

/* Object를 Reject하기위해 넓이와 높이 계산하는 함수 */
void checkArea(Object object[], Mat& src, Mat& original)
{
	for (int i = labelWidth / 10-1; i < labelWidth / 10 * 9+1; i++)
	{
		for (int j = 0; j < labelHeight / 3 * 2+1; j++)
		{
			if (label[j*labelWidth + i] > 0)
			{
				object[label[j*labelWidth + i]-1].count++;
				object[label[j*labelWidth + i]-1].centerX += i;
				object[label[j*labelWidth + i]-1].centerY += j;
				if (object[label[j*labelWidth + i]-1].checking)
					continue;
				else
				{
					object[label[j*labelWidth + i]-1].width++;
					object[label[j*labelWidth + i]-1].checking = true;
				}
			}
		}
		
		for (int k = 0; k < objectN; k++)
			if(object[k].checking) object[k].checking = false;
	}

	for (int i = 0; i < labelHeight / 3 * 2; i++)
	{
		for (int j = labelWidth / 10 - 1; j < labelWidth / 10 * 9 + 1; j++)
		{
			if (label[i*labelWidth + j] > 0)
			{
				if (object[label[i*labelWidth + j]-1].checking)
					continue;
				else
				{
					object[label[i*labelWidth + j]-1].height++;
					object[label[i*labelWidth + j]-1].checking = true;
				}
			}
		}

		for (int k = 0; k < objectN; k++)
			if (object[k].checking) object[k].checking = false;
	}

	for (int k = 0; k < objectN; k++)
	{
		object[k].centerX /= object[k].count;
		object[k].centerY /= object[k].count;
		
		if (object[k].count > 2)
		{
			int maxLenght = max(object[k].height, object[k].width);
			if (!(maxLenght * maxLenght >= object[k].count && 0.5890485 * maxLenght * maxLenght <= object[k].count))
			{
				object[k].isDeleted = true;
			}
		}
	}
}

/* Lebeing 시작하는 함수 */
int find_components(Mat& sourceImage) {
	int component = 0;

	for (int i = 0; i < sourceImage.cols; ++i)
	{
		for (int j = 0; j < sourceImage.rows; ++j)
		{
			if (label[j*sourceImage.cols + i] == 0 && sourceImage.at<uchar>(j, i) > 0)
			{
				labeling(i, j, ++component, sourceImage);
			}
		}
	}

	return component;
}

/* 화면 바깥쪽의 잡티를 제거하고 신호등이 될수있는 색을 밝게 해주는 함수 */
void makeWhite(Mat& dst, Mat& src,Mat& tophatImage)
{
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (i>dst.rows / 3 * 2)
			{
				dst.at<uchar>(i, j) = 0;
				continue;
			}
			
			if (j<dst.cols / 10 || j>dst.cols / 10 * 9)
			{
				dst.at<uchar>(i, j) = 0;
				continue;
			}

			int blue = src.at<Vec3b>(i, j)[0];
			int green = src.at<Vec3b>(i, j)[1];
			int red = src.at<Vec3b>(i, j)[2];

			double distanceYello = calcDistance(blue, green, red, 100, 220, 255);
			double distanceGreen = calcDistance(blue, green, red, 143, 250, 77);
			double distanceRed = calcDistance(blue, green, red, 250, 190, 76);
			
			if (tophatImage.at<uchar>(i, j) > 30)
			{
				if (distanceYello < DISTANCE_WITH_LIGHT || distanceGreen < DISTANCE_WITH_LIGHT || distanceRed < DISTANCE_WITH_LIGHT)
				{
					dst.at<uchar>(i, j) = 255;
				}
				else if (distanceYello < DISTANCE_WITH_LIGHT + 30 || distanceGreen < DISTANCE_WITH_LIGHT + 30 || distanceRed < DISTANCE_WITH_LIGHT + 30)
				{
					dst.at<uchar>(i, j) += 80;
				}
				else if (distanceYello < DISTANCE_WITH_LIGHT + 60 || distanceGreen < DISTANCE_WITH_LIGHT + 60 || distanceRed < DISTANCE_WITH_LIGHT + 60)
				{
					dst.at<uchar>(i, j) += 50;
				}
			}

		}
	}
}

//
//void regionGrowing()
//{
//
//}
//
//void checkRegion(Object object[],Mat& grayScaleImage)
//{
//
//	for (int k = 0; k < objectN; k++)
//	{
//		Mat temp(labelHeight, labelWidth, CV_8UC1, Scalar(0));
//		regionGrowing();
//		
//	}
//}


void checkSurround(Object object[], Mat& src)
{
	for (int k = 0; k < objectN; k++)
	{
		if (!object[k].isDeleted)
		{
			int blackNum = 0;
			int areaWindow = 0;
			int multipleNum = 3;
			if (object[k].count < 20) multipleNum = 5;
			int left = object[k].centerX - multipleNum * 3 * object[k].width;
			int right = object[k].centerX + multipleNum * object[k].width / 2;
			int up = object[k].centerY - multipleNum * object[k].height;
			int down = object[k].centerY + multipleNum * object[k].height;

			if (left < 0 || right>labelWidth || up < 0 || down >labelHeight) continue;

			areaWindow = (right - left)*(down - up);

			for (int i = left; i < right; i++)
			{
				for (int j = up; j < down; j++)
				{
					int blue = src.at<Vec3b>(j, i)[0];
					int green = src.at<Vec3b>(j, i)[1];
					int red = src.at<Vec3b>(j, i)[2];
					if (calcDistance(blue, green, red, 20, 20, 20) < 40) blackNum++;
				}
			}

			if (blackNum < areaWindow*0.5) object[k].isDeleted = true;
			else continue;

			blackNum = 0;
			areaWindow = 0;
			left = object[k].centerX - multipleNum * object[k].width / 2;
			right = object[k].centerX + multipleNum * 3 * object[k].width;
			up = object[k].centerY - multipleNum * object[k].height;
			down = object[k].centerY + multipleNum * object[k].height;

			if (left < 0 || right>labelWidth || up < 0 || down >labelHeight) continue;

			areaWindow = (right - left)*(down - up);

			for (int i = left; i < right; i++)
			{
				for (int j = up; j < down; j++)
				{
					int blue = src.at<Vec3b>(j, i)[0];
					int green = src.at<Vec3b>(j, i)[1];
					int red = src.at<Vec3b>(j, i)[2];
					if (calcDistance(blue, green, red, 30, 30, 30) < 40) blackNum++;
				}
			}

			if (blackNum < areaWindow*0.5) object[k].isDeleted = true;
		}
	}
}


void main(void)
{
	/* Setting */
	Mat image = imread("test2.jpg");
	Mat grayScaleImage(image.rows, image.cols, CV_8UC1);
	Mat topHatImage(image.rows, image.cols, CV_8UC1);
	Mat thresholdedImage(image.rows, image.cols, CV_8UC1);
	Mat temp(image.rows, image.cols, CV_8UC1, Scalar(0));

	label = new int[image.cols*image.rows];
	labelWidth = image.cols;
	labelHeight = image.rows;
	for (int i = 0; i < image.cols*image.rows; i++) label[i] = 0;

	/* GrayScale */
	cvtColor(image, grayScaleImage, CV_BGR2GRAY);

	/* White Top Hat */
	morphologyEx(grayScaleImage, topHatImage, CV_MOP_TOPHAT, element_tophat);

	/* Make Spot Bright and extra color black */
	makeWhite(topHatImage, image, topHatImage);

	/* Thresholding */
	threshold(topHatImage, thresholdedImage, THRESHOLD_UNDER, THRESHOLD_UPPER, CV_THRESH_BINARY);
	
	/* First Filter(Area) */
	objectN = find_components(thresholdedImage);
	Object* object = new Object[objectN];
	checkArea(object, thresholdedImage, image);


	//지워진거 삭제해주는 임시코드
	for (int k = 0; k < objectN; k++)
	{
		for (int i = 0; i < image.cols; i++)
		{
			for (int j = 0; j < image.rows; j++)
			{
				if (!object[k].isDeleted)
				{
					if (k + 1 == label[j*image.cols + i])
					{
						temp.at<uchar>(j, i) = 255;
					}
				}
			}
		}
	}

	/* Second Filter(Region Growing */
	//checkRegion(object, thresholdedImage, grayScaleImage);

	/* Final Filter(Is Background Black?) */
	//checkSurround(object, image);


	imshow("image", image);
	imshow("grayScaleImage", grayScaleImage);
	imshow("topHatImage", topHatImage);
	imshow("thresholdedImage", thresholdedImage);
	imshow("temp", temp);
	waitKey(0);

	delete[] label;
	//delete[] object;
}

