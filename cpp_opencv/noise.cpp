#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iomanip>

void printProgressBar(int progress, int total) {
    const int barWidth = 50;

    std::cout << "[";
    int pos = static_cast<int>(static_cast<float>(barWidth * progress) / total);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(static_cast<float>(progress) / total * 100.0) << " %\r";
    std::cout.flush();
}

void bilateralFilter(const std::vector<std::vector<double>>& src, std::vector<std::vector<double>>& dst, int d, double sigmaColor, double sigmaSpace) {
    int height = src.size();
    int width = src[0].size();
    int half = d / 2;
    double colorCoef = -0.5 / (sigmaColor * sigmaColor);
    double spaceCoef = -0.5 / (sigmaSpace * sigmaSpace);

    dst = src;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0, norm = 0.0;

            for (int j = -half; j <= half; ++j) {
                for (int i = -half; i <= half; ++i) {
                    if (y + j >= 0 && y + j < height && x + i >= 0 && x + i < width) {
                        double colorDiff = src[y + j][x + i] - src[y][x];
                        double spaceDiff = sqrt(i * i + j * j);
                        double weight = exp(colorDiff * colorDiff * colorCoef + spaceDiff * spaceDiff * spaceCoef);
                        sum += weight * src[y + j][x + i];
                        norm += weight;
                    }
                }
            }

            dst[y][x] = sum / norm;
        }
    }
}

void bilateralFilterBGR(const cv::Mat& src, cv::Mat& dst, int d, double sigmaColor, double sigmaSpace) {
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);

    for (int c = 0; c < 3; ++c) {
        cv::Mat temp;
        channels[c].convertTo(temp, CV_64F);
        std::vector<std::vector<double>> img(temp.rows, std::vector<double>(temp.cols));
        std::vector<std::vector<double>> result(temp.rows, std::vector<double>(temp.cols));

        for (int y = 0; y < temp.rows; ++y)
            for (int x = 0; x < temp.cols; ++x)
                img[y][x] = temp.at<double>(y, x);

        bilateralFilter(img, result, d, sigmaColor, sigmaSpace);

        for (int y = 0; y < temp.rows; ++y)
            for (int x = 0; x < temp.cols; ++x)
                temp.at<double>(y, x) = result[y][x];

        temp.convertTo(channels[c], CV_8U);
    }

    cv::merge(channels, dst);
}

void nonLocalMeansFilterDouble(const cv::Mat& src, cv::Mat& dst, double h, double sigma, int patchSize, int filterHeight, int filterWidth) {
    int height = src.rows;
    int width = src.cols;
    int halfPatchSize = patchSize / 2;

    src.copyTo(dst);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printProgressBar(y*width+x, width*height);
            cv::Mat patch1 = cv::Mat::zeros(patchSize, patchSize, src.type());
            for (int j = -halfPatchSize; j <= halfPatchSize; ++j) {
                for (int i = -halfPatchSize; i <= halfPatchSize; ++i) {
                    if (y + j >= 0 && y + j < height && x + i >= 0 && x + i < width) {
                        patch1.at<cv::Vec3d>(j+halfPatchSize, i+halfPatchSize) = src.at<cv::Vec3d>(y+j, x+i);
                    }
                }
            }

            int filterYstart = std::max(0, y - filterHeight/2);
            int filterXstart = std::max(0, x - filterWidth/2);
            if (filterYstart + filterHeight > height)
                filterYstart = height - filterHeight;
            if (filterXstart + filterWidth > width)
                filterXstart = width - filterWidth;

            cv::Mat w = cv::Mat::zeros(filterHeight, filterWidth, CV_64F);
            double sumW = 0.0;
            for (int yj = filterYstart; yj < filterYstart + filterHeight; yj++) {
                for (int xj = filterXstart; xj < filterXstart + filterWidth; xj++) {
                    cv::Mat patch2 = cv::Mat::zeros(patchSize, patchSize, src.type());
                    for (int j = -halfPatchSize; j <= halfPatchSize; ++j) {
                        for (int i = -halfPatchSize; i <= halfPatchSize; ++i) {
                            if (yj + j >= 0 && yj + j < height && xj + i >= 0 && xj + i < width) {
                                patch2.at<cv::Vec3d>(j+halfPatchSize, i+halfPatchSize) = src.at<cv::Vec3d>(yj+j, xj+i);
                            }
                        }
                    }
                    cv::Mat patchDiff;
                    cv::subtract(patch1, patch2, patchDiff);
                    patchDiff = patchDiff.reshape(1, patchSize*patchSize*3);
                    double norm = cv::norm(patchDiff, cv::NORM_L2);
                    double d = norm*norm/(patchSize*patchSize*3) - 2*sigma*sigma;
                    w.at<double>(yj-filterYstart, xj-filterXstart) =  exp(-std::max(d, 0.0)/(h*h));
                    sumW += exp(-std::max(d, 0.0)/(h*h));
                }
            }

            cv::Vec3d sumPix = cv::Vec3d(0,0,0);
            for (int yj = filterYstart; yj < filterYstart + filterHeight; yj++) {
                for (int xj = filterXstart; xj < filterXstart + filterWidth; xj++) {
                    sumPix += src.at<cv::Vec3d>(yj, xj) * w.at<double>(yj-filterYstart, xj-filterXstart)  / sumW ;
                }
            }
            dst.at<cv::Vec3d>(y, x) = sumPix;
        }
    }
}

void nonLocalMeansFilter(const cv::Mat& src, cv::Mat& dst, double h, double sigma, int patchSize, int filterHeight, int filterWidth) {
    cv::Mat srcDouble;
    cv::Mat dstDouble;

    src.convertTo(srcDouble, CV_64FC3);
    nonLocalMeansFilterDouble(srcDouble, dstDouble, h, sigma, patchSize, filterHeight, filterWidth);

    dstDouble.convertTo(dst, CV_8UC3);
}


int main() {
    cv::Mat src = cv::imread("input2.jpg");
    if (src.empty()) {
        std::cerr << "Failed to open image file." << std::endl;
        return -1;
    }

    cv::Mat dst_bilateral;
    int d = 15;
    double sigmaColor = 30.0;
    double sigmaSpace = 10.0;
    bilateralFilterBGR(src, dst_bilateral, d, sigmaColor, sigmaSpace);
    cv::imwrite("output_bilateral.jpg", dst_bilateral);

    cv::Mat dst_nonlocalmeans;
    double h = 15;
    double sigma = 0.2;
    int patchSize = 15;
    int filterHeight = 100;
    int filterWidth = 100;
    nonLocalMeansFilter(src, dst_nonlocalmeans, h, sigma, patchSize, filterHeight, filterWidth);
    cv::imwrite("output_nonlocalmeans.jpg", dst_nonlocalmeans);

    return 0;
}