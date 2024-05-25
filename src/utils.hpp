#pragma once

#include <opencv2/opencv.hpp>

struct Detection {
    int id;
    std::string category;
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Rect> trajectory{};
};

struct Frame {
    int idx;
    cv::Mat image;
    std::vector<Detection> detected_objects;
};