#include <opencv\cv.h>
#include <opencv\highgui.h>

using namespace cv;

void main(){

	Mat img = imread("Ball.jpg");
	Mat gray(img.rows, img.cols, CV_8UC1);	//��� ���� ã�����ؼ� ������� �ٲ۰�(�����Ұ� Tresholding�̶� ���ȭ�� �ٸ�����)
	Mat result(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));	//�̰� ���������� �׸��׸���, 3ä��¥�� ���������� �ʱ�ȭ
	
	cvtColor(img, gray, CV_BGR2GRAY);	//���� ã������(HoughCircle�� ��������) ������� �ٲ�)

	GaussianBlur(gray, gray, Size(9, 9), 2, 2);	//�̰��ϸ� ���� �� ��Ȯ�ϰ� ã��(���ذ��ε� ���� ����þȺ�������)

	vector<Vec3f> circles;	//HoughCircle�� 3¥�����Ϳ� x,y,r ������ ����־ ������� ������ �����Ѱ�

	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows / 8, 200, 100, 0, 0);	//�� �Ķ��Ʈ�� ���� ������ ���ͳݰ˻��ؼ� ����

	for (size_t i = 0; i < circles.size(); i++)	//���� ã�� ������ŭ
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));	//���� �߽� x,y
		int radius = cvRound(circles[i][2]);	//���� ������ r
		
		circle(result, center, radius, Scalar(255,255,255), CV_FILLED); // �̰��� ���������� �׷��� �׸��� ã�� ������ �Ͼ�� �׷��ִ� �ڵ�

		//������ bgr�� ����Ǿ����� bgr�� �о����(��� ���⼱ ���� �߽��� ã�Ҵµ� ������� �� ��ü�� ����̳� Ȯ���� ����Ͽ� ������)
		int b = img.at<Vec3b>(center.y, center.x)[0];
		int g = img.at<Vec3b>(center.y, center.x)[1];
		int r = img.at<Vec3b>(center.y, center.x)[2];

		circle(result, center, radius, Scalar(b, g, r), 3, 8, 0);	//�̰��� �׵θ� ���� �׸��� �ڵ��ε� ������ �Ʊ� ã�� bgr�� ���
	}

	imshow("display", result);
	waitKey(0);
}