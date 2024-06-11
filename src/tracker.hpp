#pragma once

#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp>


enum class KFTrackerType {
    CONSTANT_VELOCITY,
    CONSTANT_ACCELERATION
};

class KFTracker {
public:
    static size_t kf_count;
    size_t m_time_since_update = 0;
    size_t m_age = 0;
    size_t m_id = 0;

    KFTracker(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step) {
        (void)bbox;
        
        if (process_noise_scale < 0) {
            throw std::invalid_argument("process_noise_scale must be greater than or equal to 0");
        }
        if (measurement_noise_scale < 0) {
            throw std::invalid_argument("measurement_noise_scale must be greater than or equal to 0");
        }
        if (!time_step) {
            throw std::invalid_argument("time_step must be greater than 0");
        }
        m_id = ++kf_count;
    }

    virtual ~KFTracker() {
        m_history.clear();
    }

    virtual cv::Rect2f predict() = 0;
    virtual void update(cv::Rect2f bbox) = 0;
    virtual cv::Rect2f get_state() = 0;

protected:
    cv::KalmanFilter kf{};
    cv::Mat measurement{};
    std::vector<cv::Rect2f> m_history{};
    virtual void init_kf(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step) = 0;
};

class KFTrackerConstantVelocity : public KFTracker {
public:
    KFTrackerConstantVelocity(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step) 
        : KFTracker(bbox, process_noise_scale, measurement_noise_scale, time_step) {
            init_kf(bbox, process_noise_scale, measurement_noise_scale, time_step);
        }

    cv::Rect2f predict() override;
    void update(cv::Rect2f bbox) override;
    cv::Rect2f get_state() override;
    cv::Rect2f get_bbox(float cx, float cy, float s, float r);

private:
    void init_kf(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step) override;
};

class KFTrackerConstantAcceleration : public KFTracker {
public:
    KFTrackerConstantAcceleration(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step) 
        : KFTracker(bbox, process_noise_scale, measurement_noise_scale, time_step) {
            init_kf(bbox, process_noise_scale, measurement_noise_scale, time_step);
        }

    cv::Rect2f predict() override;
    void update(cv::Rect2f bbox) override;
    cv::Rect2f get_state() override;

private:
    void init_kf(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step) override;
};

class KFTrackerFactory {
public:
    static std::unique_ptr<KFTracker> create(KFTrackerType type, cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step) {
        switch (type) {
        case KFTrackerType::CONSTANT_VELOCITY:
            return std::make_unique<KFTrackerConstantVelocity>(bbox, process_noise_scale, measurement_noise_scale, time_step);
        case KFTrackerType::CONSTANT_ACCELERATION:
            return std::make_unique<KFTrackerConstantAcceleration>(bbox, process_noise_scale, measurement_noise_scale, time_step);
        default:
            throw std::invalid_argument("Invalid KFTrackerType");
        }
    }
};