#pragma once

#include <opencv2/opencv.hpp>

class Track {
public:
    static int kf_count;
    size_t m_time_since_update = 0;
    size_t m_age = 0;
    int m_id = -1;


    Track(cv::Rect2f bbox, float process_noise_scale = 1.f, float measurement_noise_scale = 1.f);
    ~Track();

    cv::Rect2f predict();
    void update(cv::Rect2f bbox);
    cv::Rect2f get_state();
    cv::Rect2f get_bbox(float cx, float cy, float s, float r);

private:
    cv::KalmanFilter kf{};
    cv::Mat measurement{};
    std::vector<cv::Rect2f> m_history{};
    void init_kf(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale);
};