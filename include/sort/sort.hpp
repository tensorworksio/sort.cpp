#pragma once

#include "sort/tracker.hpp"
#include "sort/utils.hpp"
#include <nlohmann/json.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

struct SORTConfig {
    // type of propagation model for the tracker
    KFTrackerType tracker = KFTrackerType::CONSTANT_VELOCITY;
    // track lost streak before delete
    size_t max_age = 3;
    // min iou for the association matrix
    float iou_threshold = 0.3f;
    // the higher the less you trust the propagation model
    float process_noise_scale = 1.f;
    // the higher the less you trust the object detector
    float measurement_noise_scale = 1.f;
    // propagation model time step
    size_t time_step = 1;
    
    static SORTConfig loadFromJson(const std::string& filename) {
        boost::property_tree::ptree root;
        boost::property_tree::read_json(filename, root);
        SORTConfig config;

        if (root.count("tracker")) {
            config.tracker = static_cast<KFTrackerType>(root.get<int>("tracker.value"));
        }
        if (root.count("iou_threshold")) {
            config.iou_threshold = root.get<float>("iou_threshold.value");
        }
        if (root.count("process_noise_scale")) {
            config.process_noise_scale = root.get<float>("process_noise_scale.value");
        }
        if (root.count("measurement_noise_scale")) {
            config.measurement_noise_scale = root.get<float>("measurement_noise_scale.value");
        }
        if (root.count("time_step")) {
            config.time_step = root.get<size_t>("time_step.value");
        }

        return config;
    }

    friend std::ostream& operator<<(std::ostream& os, const SORTConfig& config) {
        os << "tracker                  :" << static_cast<int>(config.tracker) << "\n"
           << "max_age                  :" << config.max_age << "\n"
           << "iou_threshold            :" << config.iou_threshold << "\n"
           << "process_noise_scale      :" << config.process_noise_scale << "\n"
           << "measurement_noise_scale  :" << config.measurement_noise_scale << "\n"
           << "time_step                :" << config.time_step;
        return os;
    }
};

class SORT {
public:
    SORT(const SORTConfig& config);
    SORT(KFTrackerType type = KFTrackerType::CONSTANT_VELOCITY, 
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
