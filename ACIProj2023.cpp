#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

void calculateHistogram(const cv::Mat& image, cv::Mat& hist) {
    // Initialize the histogram matrix with all zeros, one row, and 256 columns
    hist = cv::Mat::zeros(1, 256, CV_32F);

    // Iterate through each pixel in the image
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            // Get the intensity value of the pixel
            int pixelValue = image.at<uchar>(i, j);

            // Increment the corresponding bin in the histogram
            hist.at<float>(pixelValue) += 1.0;
        }
    }
    // Normalize the histogram by dividing each bin by the total number of pixels in the image
    hist /= (image.rows * image.cols);
}

void calculateCumulativeHistogram(const cv::Mat& hist, cv::Mat& histCumulative) {
    // Clone the original histogram to store the cumulative distribution
    histCumulative = hist.clone();

    // Iterate through the bins of the cumulative histogram
    for (int i = 1; i < histCumulative.cols; ++i) {
        // Sum the current bin with the previous bin
        histCumulative.at<float>(i) += histCumulative.at<float>(i - 1);
    }
    // Normalize the cumulative histogram by dividing each bin by the total sum (last bin)
    histCumulative /= histCumulative.at<float>(255);
}

cv::Mat automaticContrastEnhancementHE(cv::Mat image) {
    // Clone the input image to avoid modifying the original
    cv::Mat imageClone = image.clone();

    // Convert the image to grayscale if it is in color
    if (imageClone.channels() == 3) {
        cv::cvtColor(imageClone, imageClone, cv::COLOR_BGR2GRAY);
    }

    // Calculate the histogram of the input image
    cv::Mat hist, histCumulative;
    calculateHistogram(imageClone, hist);

    // Calculate the cumulative distribution of the histogram
    calculateCumulativeHistogram(hist, histCumulative);

    // Adjust pixel values in the image based on the cumulative distribution
    for (int i = 0; i < imageClone.rows; ++i) {
        for (int j = 0; j < imageClone.cols; ++j) {
            // Get the intensity value of the current pixel
            int pixelValue = imageClone.at<uchar>(i, j);

            // Calculate the new intensity value based on the cumulative distribution
            float newPixelValue = histCumulative.at<float>(pixelValue) * 255;

            // Update the pixel value in the output image
            imageClone.at<uchar>(i, j) = static_cast<uchar>(newPixelValue);
        }
    }
    // Return the contrast-enhanced image
    return imageClone;
}

cv::Mat automaticContrastEnhancementCLAHE(const cv::Mat& image, float clipLimit = 2.0) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit);
    cv::Mat imageClahe;
    clahe->apply(image, imageClahe);
    return imageClahe;
}

cv::Mat automaticContrastEnhancementContrastStretching(const cv::Mat& image) {
    cv::Mat imageStretch = image.clone();

    double minVal, maxVal;
    cv::minMaxLoc(imageStretch, &minVal, &maxVal);

    // Apply contrast stretching to each pixel in the image
    for (int i = 0; i < imageStretch.rows; ++i) {
        for (int j = 0; j < imageStretch.cols; ++j) {
            int pixelValue = imageStretch.at<uchar>(i, j);

            // Normalize pixel value to the range [0, 1]
            double normalizedPixelValue = (pixelValue - minVal) / (maxVal - minVal);

            // Scale normalized value to the range [0, 255]
            double stretchedPixelValue = 255.0 * normalizedPixelValue;

            // Update the pixel value in the image
            imageStretch.at<uchar>(i, j) = static_cast<uchar>(stretchedPixelValue);
        }
    }
    return imageStretch;
}

void retinexSingleScale(const cv::Mat& image, cv::Mat& result, float sigma) {
    // Convert the image to float for calculations
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);

    // Apply Gaussian blur to the original image
    cv::Mat blurredImage;
    cv::GaussianBlur(imageFloat, blurredImage, cv::Size(0, 0), sigma);

    // Subtract blurred image from the original image
    result = imageFloat - blurredImage;

    // Convert the result back to the original image type
    result.convertTo(result, image.type());

    // Normalize the result to the range [0, 255]
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8U);
}

cv::Mat automaticContrastEnhancementRetinex(const cv::Mat& image, float sigma) {
    cv::Mat imageRetinex;
    retinexSingleScale(image, imageRetinex, sigma);

    return imageRetinex;
}
cv::Mat automaticContrastEnhancementAdaptive(const cv::Mat& image, int blockSize = 16) {
    // Clone the input image to avoid modifying the original
    cv::Mat imageAdaptive = image.clone();

    // Iterate through image in blocks defined by the blockSize
    for (int i = 0; i < imageAdaptive.rows; i += blockSize) {
        for (int j = 0; j < imageAdaptive.cols; j += blockSize) {
            // Define the region of interest (ROI) for the current block
            cv::Rect roi(j, i, std::min(blockSize, imageAdaptive.cols - j), std::min(blockSize, imageAdaptive.rows - i));

            // Extract the current block from the image
            cv::Mat block = imageAdaptive(roi);

            // Apply histogram equalization to the current block
            cv::Mat blockEqualized;
            cv::equalizeHist(block, blockEqualized);

            // Copy the equalized block back to the original image
            blockEqualized.copyTo(imageAdaptive(roi));
        }
    }

    // Return the contrast-enhanced image
    return imageAdaptive;
}

int main() {
    std::string pathName = "C://Users//Ovidiu//Desktop//ACI//pixel";
    std::string fileExtension = ".png";
    cv::Mat image = cv::imread(pathName + fileExtension, cv::IMREAD_GRAYSCALE);

    cv::imwrite(pathName + "_gray" + fileExtension, image);

    // Histogram Equalization (HE)
    cv::Mat imageHE = automaticContrastEnhancementHE(image);
    cv::imwrite(pathName + "_HE" + fileExtension, imageHE);

    // CLAHE
    for (int i = 0; i <= 40; i += 4)
    {
        cv::Mat imageCLAHE = automaticContrastEnhancementCLAHE(image, i);
        cv::imwrite(pathName + "_CLAHE_" + std::to_string(i) + fileExtension, imageCLAHE);
    }

    // Contrast Stretching
    cv::Mat imageStretch = automaticContrastEnhancementContrastStretching(image);
    cv::imwrite(pathName + "_ContrastStretching" + fileExtension, imageStretch);


    // Single-Scale Retinex
    float sigma = 500;
       cv::Mat imageRetinex = automaticContrastEnhancementRetinex(image, sigma);
       cv::imwrite(pathName + "_RetinexFixed_"+std::to_string(sigma) + fileExtension, imageRetinex);

       // Adaptive Contrast Enhancement(ACE)
    int blockSize = 8;
    cv::Mat imageAdaptive = automaticContrastEnhancementAdaptive(image, blockSize);
    cv::imwrite(pathName + "_Adaptive_" + std::to_string(blockSize) + fileExtension, imageAdaptive);


    return 0;
}