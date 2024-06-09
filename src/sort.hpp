#pragma once

#include "tracker.hpp"
#include "utils.hpp"

class SortTracker {
public:
    SortTracker(KFTrackerType type = KFTrackerType::CONSTANT_VELOCITY, 
                size_t max_age = 3, 
                float iou_threshold = 0.3, 
                float process_noise_scale = 1.f, 
                float measurement_noise_scale = 1.f, 
                size_t time_step = 1);
    void process(Frame& frame);
    void assign(std::vector<Detection>& detections, 
                std::set<std::pair<size_t, size_t>>& matches, 
                std::set<size_t>& unmatched_detections, 
                std::set<size_t>& unmatched_tracks);

    std::vector<std::unique_ptr<KFTracker>> tracks{};
private:
    const KFTrackerType tracker_type;
    const size_t max_age;
    const int iou_threshold;
    const float process_noise_scale;
    const float measurement_noise_scale;
    const size_t time_step;
    static constexpr float PRECISION = 1E6f;
};