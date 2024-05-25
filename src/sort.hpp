#pragma once

#include "track.hpp"
#include "utils.hpp"

class SortTracker {
public:
    SortTracker(size_t max_age = 3, float iou_threshold = 0.3, float process_noise_scale = 1.f, float measurement_noise_scale = 1.f);
    void process(Frame& frame);
    void assign(std::vector<Detection>& detections, 
                std::set<std::pair<size_t, size_t>>& matches, 
                std::set<size_t>& unmatched_detections, 
                std::set<size_t>& unmatched_tracks);

private:
    std::vector<Track> tracks{};
    const size_t max_age;
    const int iou_threshold;
    const float process_noise_scale;
    const float measurement_noise_scale;
    static constexpr float PRECISION = 1E6f;
};