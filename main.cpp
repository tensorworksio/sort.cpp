#include <iostream>
#include <iomanip> 
#include <fstream>
#include <string>
#include <filesystem> 

#include "sort.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "MOT tracker using SORT algorithm")
        ("path", po::value<std::string>(), "path to MOT sequence folder")
        ("display", po::bool_switch()->default_value(false), "display frames");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::string path;
    std::string detPath;
    std::string gtPath;
    std::string outPath;
    std::filesystem::path imgPath ;
    std::vector<std::filesystem::path> imageFiles;

    if (vm.count("path")) {
        path = vm["path"].as<std::string>();
        detPath = path + "/det/det.txt";
        gtPath = path + "/gt/gt.txt";
        outPath = path + "/out.txt";
        imgPath = path + "/img1";
    } else {
        std::cerr << "Path to MOT sequence was not set.\n";
        return 1;
    }

    bool display;
    if (vm["display"].as<bool>()) {
        display = true;
    }

    std::ifstream infile(detPath);
    std::ofstream outfile(outPath);

    std::istringstream iss;
    std::ostringstream oss;
    std::string line;
    std::string next_line;
    
    SortTracker tracker = SortTracker(KFTrackerType::CONSTANT_VELOCITY);
    Detection detection;
    Frame frame;

    for (const auto& entry : std::filesystem::directory_iterator(imgPath)) {
        imageFiles.push_back(entry.path());
    }

    std::sort(imageFiles.begin(), imageFiles.end());

    for (const auto& path : imageFiles) {
        frame.idx++;
        frame.image = cv::imread(path);
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