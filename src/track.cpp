#include "track.hpp"

int Track::kf_count = 0;

void Track::init_kf(cv::Rect bbox, float process_noise_scale, float measurement_noise_scale) 
{
    int stateNum = 7; // [cx,cy,s,r,d(cx),d(cy),d(s)]
    int measureNum = 4; // [x,y,s,r]
    kf = cv::KalmanFilter(stateNum, measureNum, 0);
    measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

    kf.measurementMatrix = (cv::Mat_<float>(measureNum, stateNum) <<
        1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0);

    kf.processNoiseCov = (cv::Mat_<float>(stateNum, stateNum) <<
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1);

    kf.processNoiseCov *= process_noise_scale;
    kf.processNoiseCov.at<float>(stateNum - 1, stateNum - 1) *= 0.01f;
    kf.processNoiseCov.rowRange(4, stateNum).colRange(4, stateNum) *= 0.01f;
    
    kf.measurementNoiseCov = cv::Mat_<float>::eye(measureNum, measureNum);
    kf.measurementNoiseCov *= measurement_noise_scale;
    kf.measurementNoiseCov.rowRange(2, measureNum).colRange(2, measureNum) *= 0.01f;

    kf.errorCovPre = cv::Mat_<float>::ones(stateNum, stateNum);
    kf.errorCovPre *= 10.f;
    kf.errorCovPre.rowRange(4, stateNum).colRange(4, stateNum) *= 100.f;

    // initialize state vector with bounding box in [cx,cy,s,r] style
    kf.statePost = cv::Mat_<float>::zeros(stateNum, 1);
    update(bbox);  
}

Track::Track(cv::Rect bbox, float process_noise_scale, float measurement_noise_scale)
{
    m_id = kf_count++;
    init_kf(bbox, process_noise_scale, measurement_noise_scale);
}

Track::~Track()
{
    m_history.clear();
}

cv::Rect Track::predict()
{
    cv::Mat prediction = kf.predict();
    m_age++;
    m_time_since_update++;
    return get_bbox(prediction.at<float>(0, 0), prediction.at<float>(1, 0), prediction.at<float>(2, 0), prediction.at<float>(3, 0));

}

void Track::update(cv::Rect bbox)
{
    m_time_since_update = 0;
    m_history.clear();

    // update measurement
    measurement.at<float>(0, 0) = bbox.x + bbox.width / 2;
    measurement.at<float>(1, 0) = bbox.y + bbox.height / 2;
    measurement.at<float>(2, 0) = bbox.area();
    measurement.at<float>(3, 0) = bbox.width / bbox.height;

    // update
    kf.correct(measurement);
}

cv::Rect Track::get_state()
{
    cv::Mat state = kf.statePost;
    return get_bbox(state.at<float>(0, 0), state.at<float>(1, 0), state.at<float>(2, 0), state.at<float>(3, 0));
}

cv::Rect Track::get_bbox(float cx, float cy, float s, float r)
{
    float w = sqrt(s * r);
    float h = s / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);

    return cv::Rect(x, y, w, h);
}