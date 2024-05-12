#pragma once

#include <track.hpp>
#include <utils.hpp>

class SortTracker {
private:
    std::vector<Track> tracks{};
    const int max_age;
    const int min_hits;
    const float iou_threshold;

public:
    SortTracker(int max_age = 1, int min_hits = 3, float iou_threshold = 0.3)
        : max_age(max_age), min_hits(min_hits), iou_threshold(iou_threshold) {}

    void process(Frame& frame);
    void assign(std::vector<Detection>& detections, 
                std::vector<std::pair<int, int>>& matches, 
                std::vector<int>& unmatched_detections, 
                std::vector<int>& unmatched_tracks);

};