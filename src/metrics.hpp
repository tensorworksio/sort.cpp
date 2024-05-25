#pragma once

#include <opencv2/opencv.hpp>

inline float iou(const cv::Rect& bbox1, const cv::Rect& bbox2) {
    cv::Rect intersection = bbox1 & bbox2;
    cv::Rect union_ = bbox1 | bbox2;
    return static_cast<float>(intersection.area()) / union_.area();
}