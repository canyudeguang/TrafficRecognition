#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <stack>

using namespace cv;
using namespace std;

#define THRESHOLD_LIGHT 80
#define TOP_HAT_SIZE 3
#define DISTANCE_WITH_LIGHT 50

enum LightColor {Nothing, Red, Green};

int morph_size = TOP_HAT_SIZE;
Mat element_tophat = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

int labelWidth;
int labelHeight;
int objectN = 0;
int blackObjectN = 0;

LightColor lightColor = LightColor::Nothing;

struct Object
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

/* recursive로 구현된 Labeing함수 */
void labeling(int x, int y,int current_label, Mat& sourceImage, int label[]) {
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
		labeling(x + dx[direction], y + dy[direction], current_label, sourceImage,label);
	}
}

/* Object를 Reject하기위해 넓이와 높이 계산하는 함수 */
void checkArea(Object object[], Mat& src, Mat& original, int label[])
{
	for (int i = labelWidth / 10 -1; i < labelWidth / 10 * 9 +1; i++)
	{
		for (int j = 0; j < labelHeight / 3 * 2 +1; j++)
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

	for (int i = 0; i < labelHeight / 3 * 2 +1; i++)
	{
		for (int j = labelWidth / 10 +1; j < labelWidth / 10 * 9 +1 ; j++)
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

		int height = object[k].height;
		int width = object[k].width;

		int maxLenght = max(height, width);
		int minLenght = min(height, width);
		double dimentionRatio = (double)maxLenght / (double)minLenght;
		if (dimentionRatio > 1.2)
		{
			object[k].isDeleted = true;
		}
		if (height * width > 300 || height * width < 16)
		{
			object[k].isDeleted = true;
		}
		double fillingRatio = (double)object[k].count / height * width;
		if (fillingRatio > 0.7)
		{
			object[k].isDeleted;
		}

	}
}

/* Lebeing 시작하는 함수 */
int find_components(Mat& sourceImage, int label[]) {
	int component = 0;

	for (int i = 0; i < sourceImage.cols; ++i)
	{
		for (int j = 0; j < sourceImage.rows; ++j)
		{
			if (label[j*sourceImage.cols + i] == 0 && sourceImage.at<uchar>(j, i) > 0)
			{
				labeling(i, j, ++component, sourceImage, label);
			}
		}
	}

	return component;
}

/* 화면 바깥쪽의 잡티를 제거하고 신호등이 될수있는 색을 밝게 해주는 함수 */
void makeWhite(Mat& dst, Mat& src,Mat& tophatImage)
{
	Mat source;
	cvtColor(src, source, CV_RGB2HSV);
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
		}

	}
}

int regionGrowing(int x, int y, Mat& grayScaleImage, Mat& labelImage, int preSize, int count)
{
	if (x < 1 || x > grayScaleImage.cols - 2)
	{
		return 0;
	}
	if (y < 1 || y > grayScaleImage.rows - 2)
	{
		return 0;
	}

	int addGrowing = 0;
	
	for (int direction = 0; direction < 4; ++direction)
	{
		if (labelImage.at<uchar>(y + dy[direction], x + dx[direction]) == 0 && abs(grayScaleImage.at<uchar>(y + dy[direction], x + dx[direction]) - grayScaleImage.at<uchar>(y,x)) < 50)
		{
			labelImage.at<uchar>(y + dy[direction], x + dx[direction]) = 255;
			count++;
			if (count > preSize * 1.5)
			{
				break;
			}
			addGrowing += (regionGrowing(x + dx[direction], y + dy[direction], grayScaleImage, labelImage, preSize, count) + 1);
		}
	}

	return addGrowing;
}

void checkRegion(Mat& grayScaleImage, Object object[])
{
	Mat labelImage(grayScaleImage.rows, grayScaleImage.cols, CV_8UC1, Scalar(0));
	for (int k = 0; k < objectN; k++)
	{
		if (!object[k].isDeleted)
		{
			int count = 0;
			labelImage.at<uchar>(object[k].centerY, object[k].centerY) = 255;
			int componentSize = regionGrowing(object[k].centerX, object[k].centerY, grayScaleImage, labelImage, object[k].count, count);
			labelImage = Scalar(0);

			if (componentSize > object[k].count * 1.5)
			{
				object[k].isDeleted = true;
			}
		}
	}
}

void erosion(Mat& image)
{
	Mat temp(image.rows, image.cols, CV_8UC1, Scalar(0));

	for (int i = 1; i < image.rows - 1; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{
			if (image.at<uchar>(i, j) == 255)
			{
				if (image.at<uchar>(i - 1, j) != 255)
				{
					continue;
				}
				if (image.at<uchar>(i + 1, j) != 255)
				{
					continue;
				}
				if (image.at<uchar>(i, j - 1) != 255)
				{
					continue;
				}
				if (image.at<uchar>(i, j + 1) != 255)
				{
					continue;
				}
				temp.at<uchar>(i, j) = 255;
			}
		}
	}

	temp.copyTo(image);
}

void dilation(Mat& image)
{
	Mat temp(image.rows, image.cols, CV_8UC1, Scalar(0));

	for (int i = 1; i < image.rows - 1; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{
			if (image.at<uchar>(i, j) == 255)
			{
				temp.at<uchar>(i, j) = 255;
				temp.at<uchar>(i - 1, j) = 255;
				temp.at<uchar>(i + 1, j) = 255;
				temp.at<uchar>(i, j - 1) = 255;
				temp.at<uchar>(i, j + 1) = 255;
			}
		}
	}
	temp.copyTo(image);
}

void opening(Mat& image)
{
	erosion(image);
	dilation(image);
}

void closing(Mat& image)
{
	dilation(image);
	erosion(image);
}

void templateMatching(Mat& image, Object object[], Mat& greenLightTemplate, Mat& redLightTemplate)
{
	Mat tempTemplate(greenLightTemplate.rows, greenLightTemplate.cols,CV_8UC3);
	LightColor tempLightColor = LightColor::Nothing;
	for (int k = 0; k < objectN; k++)
	{
		if (!object[k].isDeleted)
		{
			/* 색깔로 파란불인지 빨간불인지 인식 */
			int b = image.at<Vec3b>(object[k].centerY, object[k].centerX)[0];
			int g = image.at<Vec3b>(object[k].centerY, object[k].centerX)[1];
			int r = image.at<Vec3b>(object[k].centerY, object[k].centerX)[2];

			int i = 0;
			int j = 0;

			double distanceWithGreen = sqrt((b - 180)*(b - 180) + (g - 180)*(g - 180) + (r - 150)*(r - 150));
			double distanceWithRed = sqrt((b - 240)*(b - 240) + (g - 230)*(g - 230) + (r - 255)*(r - 255));

			Scalar color;

			if (distanceWithGreen > distanceWithRed)
			{
				color = Scalar(228, 223, 230);
				tempLightColor = LightColor::Red;
				redLightTemplate.copyTo(tempTemplate);
			}
			else
			{
				color = Scalar(205, 209, 149);
				tempLightColor = LightColor::Green;
				greenLightTemplate.copyTo(tempTemplate);
			}
			
			/* 템플릿 제작 */
			double componentRatio = (double)(object[k].width * 1.2)/tempTemplate.cols;
			resize(tempTemplate, tempTemplate, Size(), componentRatio, componentRatio, INTER_CUBIC);

			if (tempLightColor == LightColor::Red)
			{
				i = object[k].centerX - object[k].width * 0.6;
				j = object[k].centerY - object[k].height * 0.6;
			}
			else
			{
				i = object[k].centerX - object[k].width * 0.6;
				j = object[k].centerY - tempTemplate.rows + object[k].height * 0.6;
			}

			if (i < 0 || i + tempTemplate.cols > image.cols - 1)
			{
				object[k].isDeleted = true;
				break;
			}
			
			if (j < 0 || j + tempTemplate.rows > image.rows - 1)
			{
				object[k].isDeleted = true;
				break;
			}

			int count = 0;
			int colorDistance = 0;

			/* 매칭 */
			for (int x = 0; x < tempTemplate.cols; x++)
			{
				for (int y = 0; y < tempTemplate.rows; y++)
				{
					colorDistance = sqrt((image.at<Vec3b>(j + y, i + x)[0] - tempTemplate.at<Vec3b>(y, x)[0])*(image.at<Vec3b>(j + y, i + x)[0] - tempTemplate.at<Vec3b>(y, x)[0]) +
						(image.at<Vec3b>(j + y, i + x)[1] - tempTemplate.at<Vec3b>(y, x)[1])*(image.at<Vec3b>(j + y, i + x)[1] - tempTemplate.at<Vec3b>(y, x)[1]) +
						(image.at<Vec3b>(j + y, i + x)[2] - tempTemplate.at<Vec3b>(y, x)[2])*(image.at<Vec3b>(j + y, i + x)[2] - tempTemplate.at<Vec3b>(y, x)[2]));
					
					if (colorDistance < DISTANCE_WITH_LIGHT)
					{
						count++;
					}
				}
			}

			if ((double)count / (tempTemplate.cols * tempTemplate.rows) > 0.5)
			{
				lightColor = tempLightColor;
			}
			else
			{
				object[k].isDeleted = true;
			}

			line(image, Point(object[k].centerX, object[k].centerY), Point(50, 50), color, 2);
			circle(image, Point(50, 50), 30, color, -1, 8);
			circle(image, Point(50, 50), 30, Scalar(0,0,0), 2, 8);

		}
	}
}

void main(void)
{
	/* Setting */
	Mat image = imread("test2.jpg");
	Mat greenLightTemplate = imread("GreenLight.jpg");
	Mat redLightTemplate = imread("RedLight.jpg");
	Mat grayScaleImage(image.rows, image.cols, CV_8UC1);
	Mat topHatImage(image.rows, image.cols, CV_8UC1);
	Mat thresholdedImage(image.rows, image.cols, CV_8UC1);
	Mat temp(image.rows, image.cols, CV_8UC1, Scalar(0));

	int *label = new int[image.cols*image.rows];
	labelWidth = image.cols;
	labelHeight = image.rows;

	for (int i = 0; i < image.cols*image.rows; i++)	label[i] = 0;

	/* GrayScale */
	cvtColor(image, grayScaleImage, CV_BGR2GRAY);

	/* White Top Hat */
	morphologyEx(grayScaleImage, topHatImage, CV_MOP_TOPHAT, element_tophat);

	/* Make Spot Bright and extra color black */
	makeWhite(topHatImage, image, topHatImage);

	threshold(topHatImage, thresholdedImage, THRESHOLD_LIGHT, 255, THRESH_BINARY);
	opening(thresholdedImage);
	closing(thresholdedImage);

	/* First Filter(Area) */
	objectN = find_components(thresholdedImage, label);
	Object* object = new Object[objectN];
	checkArea(object, thresholdedImage, image, label);

	/* regionGrowing */
	checkRegion(grayScaleImage, object);

	/* Template Matching */
	templateMatching(image, object, greenLightTemplate, redLightTemplate);

	for (int k = 0; k < objectN; k++)
	{
		if (!object[k].isDeleted)
		{
			for (int i = 0; i < image.rows; i++)
			{
				for (int j = 0; j < image.cols; j++)
				{
					if(label[i*image.cols + j] == k+1)
						temp.at<uchar>(i, j) = 255;
				}
			}
		}
	}
	
	imshow("image", image);
	waitKey(0);

	delete[] label;
	delete[] object;
}




