#include <opencv\cv.h>
#include <opencv\highgui.h>

using namespace cv;

void main(void)
{
	Mat image = imread("test1.jpg");
	
	imshow("window", image);
	waitKey(0);
}