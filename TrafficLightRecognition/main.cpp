#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <stack>


using namespace cv;
using namespace std;

#define THRESHOLD_LIGHT 80
#define TOP_HAT_SIZE 3
#define TOP_OPENING_SIZE 1
#define TOP_CLOSING_SIZE 1
#define DISTANCE_WITH_LIGHT 50
#define VIDEO_WIDTH 640
#define VIDEO_HEIGHT 480

enum LightColor {Nothing, Red, Green};

int morph_size = TOP_HAT_SIZE;
Mat element_tophat = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
int open_size = TOP_OPENING_SIZE;
Mat element_openhat = getStructuringElement(MORPH_RECT, Size(2 * open_size + 1, 2 * open_size + 1), Point(open_size, open_size));
int close_size = TOP_CLOSING_SIZE;
Mat element_closehat = getStructuringElement(MORPH_RECT, Size(2 * close_size + 1, 2 * close_size + 1), Point(close_size, close_size));

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
void checkArea(Object object[], Mat& src, int label[])
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
	Mat tempTemplate(greenLightTemplate.rows, greenLightTemplate.cols, CV_8UC3);
	LightColor tempLightColor = LightColor::Nothing;

	for (int k = 0; k < objectN; k++)
	{
		if (!object[k].isDeleted)
		{
			/* 색깔로 파란불인지 빨간불인지 인식 */
			int b = image.at<Vec3b>(object[k].centerY, object[k].centerX)[0];
			int g = image.at<Vec3b>(object[k].centerY, object[k].centerX)[1];
			int r = image.at<Vec3b>(object[k].centerY, object[k].centerX)[2];

			double distanceWithGreen = sqrt((b - 180)*(b - 180) + (g - 180)*(g - 180) + (r - 150)*(r - 150));
			double distanceWithRed = sqrt((b - 240)*(b - 240) + (g - 230)*(g - 230) + (r - 255)*(r - 255));
			
			Scalar color;


			if (distanceWithRed < distanceWithGreen)
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
			double componentRatio = (double)(object[k].width*1.2) / tempTemplate.cols;
			resize(tempTemplate, tempTemplate, Size(), componentRatio, componentRatio, INTER_CUBIC);

			int i = 0;
			int j = 0;

			if (tempLightColor == LightColor::Red)
			{
				i = object[k].centerX - object[k].width * 1.3;
				j = object[k].centerY - object[k].height * 1.3;
			}
			else
			{
				i = object[k].centerX - object[k].width * 1.3;
				j = object[k].centerY - tempTemplate.rows;
			}

			if (i < 0 || i + tempTemplate.cols >= image.cols - 1)
			{
				object[k].isDeleted = true;
				continue;
			}

			if (j < 0 || j + tempTemplate.rows >= image.rows - 1)
			{
				object[k].isDeleted = true;
				continue;
			}

			double min, max;
			Point leftTop;
			Mat crop = image(Rect(i, j, tempTemplate.cols * 1.5 + object[k].width * 2, tempTemplate.rows * 1.5 + object[k].height * 2));
			line(image, Point(i, j), Point(i, j + tempTemplate.rows * 1.5 + object[k].height * 2), color, 2);
			line(image, Point(i, j), Point(i + tempTemplate.cols * 1.5 + object[k].width * 2, j), color, 2);
			line(image, Point(i + tempTemplate.cols * 1.5 + object[k].width * 2, j), Point(i + tempTemplate.cols * 1.5 + object[k].width * 2, j + tempTemplate.rows * 1.5 + object[k].height * 2), color, 2);
			line(image, Point(i, j + tempTemplate.rows * 1.5 + object[k].height * 2), Point(i + tempTemplate.cols * 1.5 + object[k].width * 2, j + tempTemplate.rows * 1.5 + object[k].height * 2), color, 2);

			Mat result(crop.rows, crop.cols, CV_32FC1);
			matchTemplate(crop, tempTemplate, result, CV_TM_CCOEFF_NORMED);
			minMaxLoc(result, &min, &max, NULL, &leftTop);

			
			if (max > 0.8f)
			{
				lightColor = tempLightColor;
				circle(image, Point(50, 50), 30, color, -1, 8);
				line(image, Point(50, 50), Point(object[k].centerX, object[k].centerY), color, 2);
				circle(image, Point(50, 50), 30, Scalar(0,0,0), 2, 8);
				crop = 0;
			}
		}
	}
}

int main(void)
{
	/* Setting */
	VideoCapture cap("test.avi"); // open the video file for reading
	Mat frame;
	Mat greenLightTemplate = imread("GreenLight.jpg");
	Mat redLightTemplate = imread("RedLight.jpg");
	Mat grayScaleImage(VIDEO_HEIGHT, VIDEO_WIDTH, CV_8UC1);
	Mat topHatImage(VIDEO_HEIGHT, VIDEO_WIDTH, CV_8UC1);
	Mat thresholdedImage(VIDEO_HEIGHT, VIDEO_WIDTH, CV_8UC1);

	Object* object;
	int *label = new int[VIDEO_HEIGHT * VIDEO_WIDTH];
	labelWidth = VIDEO_WIDTH;
	labelHeight = VIDEO_HEIGHT;

	for (int i = 0; i < VIDEO_HEIGHT * VIDEO_WIDTH; i++)	label[i] = 0;

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		delete[] label;

		return -1;
	}

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

	while (1)
	{

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}

		cvtColor(frame, grayScaleImage, CV_BGR2GRAY);

		morphologyEx(grayScaleImage, topHatImage, CV_MOP_TOPHAT, element_tophat);

		makeWhite(topHatImage, frame, topHatImage);

		threshold(topHatImage, thresholdedImage, THRESHOLD_LIGHT, 255, THRESH_BINARY);
		//opening(thresholdedImage);
		//closing(thresholdedImage);
		morphologyEx(grayScaleImage, topHatImage, CV_MOP_OPEN, element_openhat);
		morphologyEx(grayScaleImage, topHatImage, CV_MOP_CLOSE, element_closehat);

		objectN = find_components(thresholdedImage, label);
		object = new Object[objectN];
		checkArea(object, thresholdedImage, label);

		checkRegion(grayScaleImage, object);
		
		templateMatching(frame, object, greenLightTemplate, redLightTemplate);
		imshow("frame.jpg", frame); //show the frame in "MyVideo" window

		for (int i = 0; i < VIDEO_HEIGHT * VIDEO_WIDTH; i++)	label[i] = 0;
		delete[] object;

		if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	delete[] label;

	return 0;
}



