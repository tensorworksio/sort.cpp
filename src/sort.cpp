#include <sort.hpp>
#include <dlib/optimization/max_cost_assignment.h>


float iou(cv::Rect2f bb_test, cv::Rect2f bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    return in / un;
}


void SortTracker::assign(std::vector<Detection>& detections, 
                         std::vector<std::pair<int, int>>& matches, 
                         std::vector<int>& unmatched_detections, 
                         std::vector<int>& unmatched_tracks)
{
    if (tracks.empty()) {
        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_detections.push_back(static_cast<int>(i));
        }
        return;
    }

    if (detections.empty()) {
        for (size_t i = 0; i < tracks.size(); i++) {
            unmatched_tracks.push_back(static_cast<int>(i));
        }
        return;
    }

    // Create a cost matrix
    dlib::matrix<float> cost_matrix(detections.size(), tracks.size());
    for (size_t i = 0; i < detections.size(); i++) {
        for (size_t j = 0; j < tracks.size(); j++) {
            cost_matrix(i, j) = 1.f - iou(detections[i].bbox, tracks[j].get_state());
        }
    }

    // Solve the linear assignment problem
    std::vector<long> assignment = dlib::max_cost_assignment(cost_matrix);

    // Keep matches with IoU > threshold
    for (size_t i = 0; i < detections.size(); ++i) {
        if (cost_matrix(i, assignment[i]) < iou_threshold) {
            unmatched_detections.push_back(static_cast<int>(i));
            unmatched_tracks.push_back(static_cast<int>(assignment[i]));
        } else {
            matches.emplace_back(static_cast<int>(i), static_cast<int>(assignment[i]));
        }
    }

    // If there are more tracks than detections
    for (size_t j = 0; j < tracks.size(); ++j) {
        if (std::find(assignment.begin(), assignment.end(), j) == assignment.end()) {
            unmatched_tracks.push_back(static_cast<int>(j));
        }
    }
}

void SortTracker::process(Frame& frame) {
    std::vector<Detection>& detections = frame.detected_objects;

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_detections;
    std::vector<int> unmatched_tracks;

    // Propagate tracks
    for (auto track = tracks.begin(); track != tracks.end(); ) {
        track->predict();
        auto state = track->get_state();
        if (state.x < 0 || state.y < 0 || state.x + state.width > frame.image.cols || state.y + state.height > frame.image.rows) {
            track = tracks.erase(track);
        } else {
            ++track;
        }
    }

    assign(detections, matches, unmatched_detections, unmatched_tracks);

    // Update tracks
    for (size_t i = 0; i < matches.size(); i++) {
        int detection_idx = matches[i].first;
        int track_idx = matches[i].second;
        tracks[track_idx].update(detections[detection_idx].bbox);

        // Update detection id with verified tracks
        if ((tracks[track_idx].m_hit_streak >= min_hits) || frame.idx <= min_hits)  {
            detections[detection_idx].id = tracks[track_idx].m_id;
        }
    }

    // Create new tracks
    for (size_t i = 0; i < unmatched_detections.size(); i++) {
        Track new_track(detections[unmatched_detections[i]].bbox);
        tracks.push_back(new_track);
    }

    // Remove lost tracks
    for (size_t i = 0; i < unmatched_tracks.size(); i++) {
        if (tracks[unmatched_tracks[i]].m_time_since_update > max_age) {
            tracks.erase(tracks.begin() + unmatched_tracks[i]);
        }
    }
}