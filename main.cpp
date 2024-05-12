#include "sort.hpp"
#include "utils.hpp"

int main() {
    Frame frame;
    frame.idx = 0;
    frame.image = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
    frame.detected_objects = {
        {0, "car", cv::Rect2f(100, 100, 50, 50), 0.9},
        {1, "car", cv::Rect2f(200, 200, 50, 50), 0.9},
        {2, "car", cv::Rect2f(300, 300, 50, 50), 0.9},
        {3, "car", cv::Rect2f(400, 400, 50, 50), 0.9},
    };

    SortTracker tracker;
    tracker.process(frame);
    return 0;
}