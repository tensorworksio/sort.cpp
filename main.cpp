#include <iostream>
#include <iomanip> 
#include <fstream>
#include <string>

#include "sort.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "SORT multiple object tracker")
        ("path,p", po::value<std::string>()->required(), "Path to MOT sequence folder")
        ("config,c", po::value<std::string>(), "Path to SORT config.json")
        ("gt", po::bool_switch()->default_value(false), "Ground truth mode")
        ("display", po::bool_switch()->default_value(false), "Display frames");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }
    
    // SORT config
    SORTConfig config;
    if (vm.count("config")) {
        fs::path configPath(vm["config"].as<std::string>());
        config = SORTConfig::loadFromJson(configPath.string());
    }

    std::cout << "SORT config:" << std::endl;
    std::cout << config << std::endl;

    // Display
    bool display = vm["display"].as<bool>();

    // Ground truth
    bool gt = vm["gt"].as<bool>();
    
    // MOT path
    fs::path path(vm["path"].as<std::string>());
    fs::path outPath = path / "out.txt";

    fs::path detPath = path / "det" / "det.txt";
    if (!fs::exists(detPath)) {
        std::cerr << "Detection file does not exist: " << detPath.string() << std::endl;
        return 1;
    }
    fs::path gtPath = path / "gt" / "gt.txt";
    if (gt && !fs::exists(gtPath)) {
        std::cerr << "Ground truth file does not exist: " << gtPath.string() << std::endl;
        return 1;
    }

    fs::path imgPath = path / "img1";
    if (display && !fs::is_directory(imgPath)) {
        std::cerr << "Image directory does not exist: " << imgPath.string() << std::endl;
        return 1;
    }
    
    std::ifstream infile(gt ? gtPath.string() : detPath.string());
    std::ofstream outfile(outPath.string());

    std::istringstream iss;
    std::ostringstream oss;
    std::string line;
    std::string next_line;
    
    SORT tracker = SORT(config);
    Detection detection;
    Frame frame;
    
    std::vector<fs::path> imageFiles;
    for (const auto& entry : fs::directory_iterator(imgPath)) {
        imageFiles.push_back(entry.path());
    }

    std::sort(imageFiles.begin(), imageFiles.end());

    for (const auto& path : imageFiles) {
        frame.idx++;
        frame.image = cv::imread(path.string());
        frame.detected_objects.clear();
        
        // Read detections from file
        while (true) {
            if (!next_line.empty()) {
                line = next_line;
                next_line.clear();
            } else if (!std::getline(infile, line)) {
                break;
            }
            iss.str(line);
            iss >> detection;

            iss.str("");
            iss.clear();

            if (detection.frame == frame.idx) {
                frame.detected_objects.push_back(detection);
            } else {
                next_line = line;
                break;
            }
        }

        // Process frame
        tracker.process(frame);

        // Write detections to file
        for (const auto& detection : frame.detected_objects) {
            outfile << detection << std::endl;
        }

        // Display frame with detections
        for (const auto& detection : frame.detected_objects) {
            auto color = detection.getColor();
            cv::rectangle(frame.image, detection.bbox, color, 2);
            cv::putText(frame.image, std::to_string(detection.id), cv::Point(detection.bbox.x, detection.bbox.y), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
        }

        if (display) {
            cv::imshow("Frame", frame.image);
            if (cv::waitKey(30) == 27) {
                break;
            }
        }
    }

    cv::destroyAllWindows();
    outfile.close();
    return 0;
}
