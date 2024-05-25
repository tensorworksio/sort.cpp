#pragma once

#include <opencv2/opencv.hpp>

class Track {
public:
    static int kf_count;
    std::vector<cv::Rect> m_history{};
    size_t m_time_since_update = 0;
    size_t m_age = 0;
    int m_id = -1;


    Track(cv::Rect bbox, float process_noise_scale = 1.f, float measurement_noise_scale = 1.f);
    ~Track();

    cv::Rect predict();
    void update(cv::Rect bbox);
    cv::Rect get_state();
    cv::Rect get_bbox(float cx, float cy, float s, float r);

private:
    cv::KalmanFilter kf{};
    cv::Mat measurement{};
    void init_kf(cv::Rect bbox, float process_noise_scale, float measurement_noise_scale);
};