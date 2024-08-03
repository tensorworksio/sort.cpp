#include "sort/sort.hpp"
#include "sort/metrics.hpp"
#include <dlib/optimization/max_cost_assignment.h>

SORT::SORT(const SORTConfig& config) : 
  SORT(config.tracker, 
              config.max_age, 
              config.iou_threshold, 
              config.process_noise_scale, 
              config.measurement_noise_scale, 
              config.time_step) {}


SORT::SORT(KFTrackerType type,
                         size_t max_age, 
                         float iou_threshold, 
                         float process_noise_scale, 
                         float measurement_noise_scale, 
                         size_t time_step)
    : tracker_type(type),
      max_age(max_age), 
      iou_threshold(static_cast<int>(iou_threshold * PRECISION)), 
      process_noise_scale(process_noise_scale), 
      measurement_noise_scale(measurement_noise_scale),
      time_step(time_step) {}


void SORT::assign(std::vector<Detection>& detections, 
                         std::set<std::pair<size_t, size_t>>& matches, 
                         std::set<size_t>& unmatched_detections, 
                         std::set<size_t>& unmatched_tracks)
{   
    // By default detections are unmatched 
    for (size_t i = 0; i < detections.size(); i++) {
        unmatched_detections.insert(i);
    }

    // By default tracks are unmatched
    for (size_t i = 0; i < tracks.size(); i++) {
        unmatched_tracks.insert(i);
    }

    if (tracks.empty() || detections.empty()) {
        return;
    }

    // Create a cost matrix
    size_t size = std::max(detections.size(), tracks.size());
    dlib::matrix<int> cost_matrix = dlib::zeros_matrix<int>(size, size);
    for (size_t i = 0; i < detections.size(); i++) {
        for (size_t j = 0; j < tracks.size(); j++) {
            cost_matrix(i, j) = static_cast<int>(PRECISION * iou(detections[i].bbox, tracks[j]->get_state()));
        }
    }

    // Solve the linear assignment problem
    std::vector<long> assignment = dlib::max_cost_assignment(cost_matrix);

    // Find matches
    for (size_t i = 0; i < detections.size(); i++) {
        if (cost_matrix(i, assignment[i]) < iou_threshold) {
            continue;
        }
        unmatched_detections.erase(i);
        unmatched_tracks.erase(assignment[i]);
        matches.emplace(i, assignment[i]);
    }
}

void SORT::process(Frame& frame) {
    std::vector<Detection>& detections = frame.detected_objects;

    std::set<std::pair<size_t, size_t>> matches;
    std::set<size_t> unmatched_detections;
    std::set<size_t> unmatched_tracks;

    // Propagate tracks
    for (auto& track : tracks) {
        cv::Rect2f predicted_bbox = track->predict();

        // Clamp the predicted bounding box to the image size
        predicted_bbox.x = std::max(predicted_bbox.x, 0.f);
        predicted_bbox.y = std::max(predicted_bbox.y, 0.f);
        predicted_bbox.width = std::min(predicted_bbox.width, frame.image.cols - predicted_bbox.x - 1);
        predicted_bbox.height = std::min(predicted_bbox.height, frame.image.rows - predicted_bbox.y - 1);
    }

    // Assign detections to tracks
    assign(detections, matches, unmatched_detections, unmatched_tracks);

    // Update tracks
    for (const auto& [det_idx, track_idx] : matches) {
        tracks.at(track_idx)->update(detections[det_idx].bbox);

        // Update detection info with verified tracks
        detections[det_idx].id = tracks[track_idx]->m_id;
    }

    // Create new tracks
    for (const auto& det_idx : unmatched_detections) {
        auto new_track = KFTrackerFactory::create(tracker_type, detections[det_idx].bbox, process_noise_scale, measurement_noise_scale, time_step);
        tracks.push_back(std::move(new_track));
    }

    // Remove lost tracks
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), 
        [this](const auto& track) { return track->m_time_since_update > max_age; }), tracks.end());
}
