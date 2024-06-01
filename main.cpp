#include <iostream>
#include <iomanip> 
#include <fstream>
#include <string>
#include <filesystem> 

#include "sort.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_sequence>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    std::string detPath = path + "/det/det.txt";
    std::string gtPath = path + "/gt/gt.txt";
    std::string outPath = path + "/out.txt";

    std::filesystem::path imgPath = path + "/img1";
    std::vector<std::filesystem::path> imageFiles;

    std::ifstream infile(detPath);
    std::ofstream outfile(outPath);

    std::istringstream iss;
    std::ostringstream oss;
    std::string line;
    std::string next_line;
    
    SortTracker tracker = SortTracker();
    Detection detection;
    Frame frame;

    for (const auto& entry : std::filesystem::directory_iterator(imgPath)) {
        imageFiles.push_back(entry.path());
    }

    std::sort(imageFiles.begin(), imageFiles.end());

    for (const auto& path : imageFiles) {
        std::cout << path << std::endl;
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

            // Draw the trajectory
            std::cout << "Trajectory size " << detection.trajectory.size() << std::endl;
            for (const auto& rect : detection.trajectory) {
                cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
                cv::circle(frame.image, center, 2, color, -1);
            }
        }

        cv::imshow("Frame", frame.image);
        if (cv::waitKey(0) == 27) {
            continue;
        }
    }

    cv::destroyAllWindows();
    outfile.close();
    return 0;
}

// input: Path to sequence
// read sequence: update Frame
// when frame is ready (next id != id), call tracker.process(frame)
// output: write detection of frame to file
// display: draw detection on frame.image