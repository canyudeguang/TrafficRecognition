#include <opencv\cv.h>
#include <opencv\highgui.h>

using namespace cv;

/* Image */
Mat image = imread("test1.jpg");
Mat grayScaleImage(image.cols, image.rows, CV_8UC1);
Mat topHatImage(image.cols, image.rows, CV_8UC1);
Mat thresholdedImage(image.cols, image.rows, CV_8UC1);

/* Morphology Kernel */
int morph_size = 10;
Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

void main(void)
{
	/* GrayScale */
	cvtColor(image, grayScaleImage, CV_BGR2GRAY);

	/* White Top Hat */
	morphologyEx(grayScaleImage, topHatImage, CV_MOP_TOPHAT, element);

	/* Thresholding */
	threshold(topHatImage, thresholdedImage, 100, 256, CV_THRESH_BINARY);

	/* First Filter */
	// 라벨링 먼저 하기
	// 1. 넓이,높이로 필터
	// 2. 구멍이 있나 없나
	// 3. 컨벡스한가 아닌가

	/* Second Filter */
	// 1. Region Growing Image 생성
	// 2. filteredImage와 비교하여 큰 차이가 보이는 경우 제거

	

	imshow("grayScaleImage", grayScaleImage);
	imshow("topHatImage", topHatImage);
	imshow("thresholdedImage", thresholdedImage);
	waitKey(0);
}
