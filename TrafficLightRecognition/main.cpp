#include <opencv\cv.h>
#include <opencv\highgui.h>

#define THRESHOLD_UNDER 80
#define THRESHOLD_UPPER 256
#define MORPH_SIZE 5;

using namespace cv;
int *label;

/* Morphology Kernel */
int morph_size = MORPH_SIZE;
Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));


/* Labeling Move Direction */
const int dx[] = { +1, 0, -1, 0 };
const int dy[] = { 0, +1, 0, -1 };

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

void find_components(Mat& sourceImage) {
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

	printf("%d", component);
}

double calcDistance(int b1,int g1,int r1,int b2,int g2, int r2)
{
	return sqrt((b1 - b2)*(b1 - b2) + (g1 - g2)*(g1 - g2) + (r1 - r2)*(r1 - r2));
}

void makeWhite(Mat& dst, Mat& src,Mat& tophatImage)
{
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			int blue = src.at<Vec3b>(i, j)[0];
			int green = src.at<Vec3b>(i, j)[1];
			int red = src.at<Vec3b>(i, j)[2];

			if (tophatImage.at<uchar>(i,j) > 50)
			{
				if (calcDistance(blue, green, red, 0, 255, 0) < 255 || calcDistance(blue, green, red, 0, 0, 255) < 255 || calcDistance(blue, green, red, 0, 255, 255) < 255)
				{
					dst.at<uchar>(i, j) = 255;
				}
			}

		}
	}
}


void main(void)
{
	/* Image */
	Mat image = imread("test3.jpg");
	Mat grayScaleImage(image.rows, image.cols, CV_8UC1);
	Mat topHatImage(image.rows, image.cols, CV_8UC1);
	Mat thresholdedImage(image.rows, image.cols, CV_8UC1);
	label = new int[image.cols*image.rows];
	for (int i = 0; i < image.cols*image.rows; i++) label[i] = 0;

	/* GrayScale */
	cvtColor(image, grayScaleImage, CV_BGR2GRAY);

	/* White Top Hat */
	morphologyEx(grayScaleImage, topHatImage, CV_MOP_TOPHAT, element);
	//makeWhite(topHatImage, image, topHatImage);

	/* Thresholding */
	threshold(topHatImage, thresholdedImage, THRESHOLD_UNDER, THRESHOLD_UPPER, CV_THRESH_BINARY);

	/* First Filter */
	// 1. 넓이,높이로 필터
	// 2. 구멍이 있나 없나
	// 3. 컨벡스한가 아닌가	
	find_components(thresholdedImage);


	/* Second Filter */
	// 1. Region Growing Image 생성
	// 2. filteredImage와 비교하여 큰 차이가 보이는 경우 제거

	

	imshow("image", image);
	imshow("grayScaleImage", grayScaleImage);
	imshow("topHatImage", topHatImage);
	imshow("thresholdedImage", thresholdedImage);
	waitKey(0);

	delete[] label;
}

