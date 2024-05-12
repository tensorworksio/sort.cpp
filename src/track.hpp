#pragma once

#include <opencv2/opencv.hpp>

class Track {
private:
    void init_kf(cv::Rect2f bbox);
    cv::KalmanFilter kf;
    cv::Mat measurement;
    std::vector<cv::Rect2f> m_history;

public:
    static int kf_count;
    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;

    Track(cv::Rect2f bbox) {
        init_kf(bbox);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;
    }

    ~Track() {
        m_history.clear();
    }

    cv::Rect2f predict();
    void update(cv::Rect2f bbox);
    cv::Rect2f get_state();
    cv::Rect2f get_rect_xysr(float cx, float cy, float s, float r);
};