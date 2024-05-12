#pragma once

#include <opencv2/opencv.hpp>

struct Detection {
    int id;
    std::string category;
    cv::Rect2f bbox;
    float confidence;
};

struct Frame {
    int idx;
    cv::Mat image;
    std::vector<Detection> detected_objects;
};