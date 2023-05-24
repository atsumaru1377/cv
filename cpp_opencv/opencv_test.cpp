#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>


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

double l2norm(const std::vector<std::vector<double>>& mat) {
    int height = mat.size();
    int width = mat[0].size();
    double sumSquared = 0.0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            sumSquared += mat[y][x] * mat[y][x];
        }
    }
    return sqrt(sumSquared);
}

void subMat(const std::vector<std::vector<double>>& src1, const std::vector<std::vector<double>>& src2, std::vector<std::vector<double>>& dst) {
    int height = src1.size();
    int width = src1[0].size();
    dst = src1;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dst[y][x] = src1[y][x] - src2[y][x];
        }
    }
}

void nonLocalMeansFilter(const std::vector<std::vector<double>>& src, std::vector<std::vector<double>>& dst, double h, double sigma, int patchSize, int filterHeight, int filterWidth) {
    int height = src.size();
    int width = src[0].size();
    int halfPatchSize = patchSize / 2;

    dst = src;
    std::cout << "start" << std::endl;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            std::vector<std::vector<double>> patch1(patchSize, std::vector<double>(patchSize, 0.0));
            for (int j = -halfPatchSize; j <= halfPatchSize; ++j) {
                for (int i = -halfPatchSize; i <= halfPatchSize; ++i) {
                    if (y + j >= 0 && y + j < height && x + i >= 0 && x + i < width) {
                        patch1[j+halfPatchSize][i+halfPatchSize] = src[y+j][x+i];
                    } else {
                        patch1[j+halfPatchSize][i+halfPatchSize] = 0;
                    }
                }
            }
            int filterYstart;
            if (filterHeight == height) {
                filterYstart = 0;
            } else {
                if (y - filterHeight/2 < 0) {
                    filterYstart = 0;
                } else if (y + filterHeight/2 > height) {
                    filterYstart = height - filterHeight - 1;
                } else {
                    filterYstart = y;
                }
            }
            int filterXstart;
            if (filterWidth == width) {
                filterXstart = 0;
            } else {
                if (x - filterWidth/2 < 0) {
                    filterXstart = 0;
                } else if (x + filterWidth/2 > width) {
                    filterXstart = width - filterWidth - 1;
                } else {
                    filterXstart = x;
                }
            }
            std::cout << y << "," << x << std::endl;

            std::vector<std::vector<double>> w(filterHeight, std::vector<double>(filterWidth, 0.0));
            double sumW = 0.0;
            for (int yj = filterYstart; yj < filterYstart + filterHeight; yj++) {
                for (int xj = filterXstart; xj < filterXstart + filterWidth; xj++) {
                    std::vector<std::vector<double>> patch2(patchSize, std::vector<double>(patchSize, 0.0));
                    for (int j = -halfPatchSize; j <= halfPatchSize; ++j) {
                        for (int i = -halfPatchSize; i <= halfPatchSize; ++i) {
                            if (yj + j >= 0 && yj + j < height && xj + i >= 0 && xj + i < width) {
                                patch2[j+halfPatchSize][i+halfPatchSize] = src[yj+j][xj+i];
                            } else {
                                patch2[j+halfPatchSize][i+halfPatchSize] = 0;
                            }
                        }
                    }
                    std::vector<std::vector<double>> patchDiff;
                    subMat(patch1, patch2, patchDiff);
                    double norm = l2norm(patchDiff);
                    w[yj-filterYstart][xj-filterXstart] =  exp(-std::max(norm/(patchSize*patchSize*3)-2*sigma*sigma, 0.0));
                    sumW += exp(-std::max(norm/(patchSize*patchSize*3)-2*sigma*sigma, 0.0));
                }
            }

            for (int yj = filterYstart; yj < filterYstart + filterHeight; yj++) {
                for (int xj = filterXstart; xj < filterXstart + filterWidth; xj++) {
                    dst[y][x] += src[yj][xj] * w[yj][xj] / sumW;
                }
            }
        }
    }
}

void nonLocalMeansBGR(const cv::Mat& src, cv::Mat& dst, double h, double sigma, int patchSize, int filterHeight, int filterWidth) {
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

        // nonLocalMeansFilter(img, result, h, sigma, patchSize, filterHeight, filterWidth);
        nonLocalMeansFilter(img, result, h, sigma, patchSize, temp.rows, temp.cols);

        for (int y = 0; y < temp.rows; ++y)
            for (int x = 0; x < temp.cols; ++x)
                temp.at<double>(y, x) = result[y][x];

        temp.convertTo(channels[c], CV_8U);
    }

    cv::merge(channels, dst);
}


int main() {
    cv::Mat src = cv::imread("input3.png");
    if (src.empty()) {
        std::cerr << "Failed to open image file." << std::endl;
        return -1;
    }

    cv::Mat dst_bilateral;
    int d = 15;
    double sigmaColor = 100.0;
    double sigmaSpace = 100.0;
    bilateralFilterBGR(src, dst_bilateral, d, sigmaColor, sigmaSpace);
    cv::imwrite("output_bilateral.jpg", dst_bilateral);

    cv::Mat dst_nonlocalmeans;
    double h = 1;
    double sigma = 10;
    int patchSize = 5;
    int filterHeight = 100;
    int filterWidth = 100;
    nonLocalMeansBGR(src, dst_nonlocalmeans, h, sigma, patchSize, filterHeight, filterWidth);


    // 結果を保存
    cv::imwrite("output_nonlocalmeans.jpg", dst_nonlocalmeans);

    return 0;
}