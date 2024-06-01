#pragma once

#include <opencv2/opencv.hpp>

constexpr float EPSILON = 1e-6;

inline float iou(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2) {
    float in = (bbox1 & bbox2).area();
    float un = bbox1.area() + bbox2.area() - in;
    
    if (un < EPSILON) {
        return 0.f;
    }
    
    return in / un;
}