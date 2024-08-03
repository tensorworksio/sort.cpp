#include "sort/tracker.hpp"

size_t KFTracker::kf_count = 0;

void KFTrackerConstantVelocity::init_kf(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step)
{
    size_t stateNum = 7;   // [cx,cy,s,r,dcx,dcy,ds]
    size_t measureNum = 4; // [x,y,s,r]
    kf = cv::KalmanFilter(stateNum, measureNum, 0);
    measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    float dt = static_cast<float>(time_step);

    kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) << 1, 0, 0, 0, dt, 0, 0,
                           0, 1, 0, 0, 0, dt, 0,
                           0, 0, 1, 0, 0, 0, dt,
                           0, 0, 0, 1, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 0, 1);

    kf.measurementMatrix = (cv::Mat_<float>(measureNum, stateNum) << 1, 0, 0, 0, 0, 0, 0,
                            0, 1, 0, 0, 0, 0, 0,
                            0, 0, 1, 0, 0, 0, 0,
                            0, 0, 0, 1, 0, 0, 0);

    kf.processNoiseCov = (cv::Mat_<float>(stateNum, stateNum) << 1, 0, 0, 0, 1, 0, 0,
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
    kf.statePost.at<float>(0, 0) = bbox.x + bbox.width / 2;
    kf.statePost.at<float>(1, 0) = bbox.y + bbox.height / 2;
    kf.statePost.at<float>(2, 0) = bbox.area();
    kf.statePost.at<float>(3, 0) = bbox.width / bbox.height;
}

cv::Rect2f KFTrackerConstantVelocity::predict()
{
    m_age++;
    m_time_since_update++;
    cv::Mat prediction = kf.predict();
    auto bbox = get_bbox(prediction.at<float>(0, 0), prediction.at<float>(1, 0), prediction.at<float>(2, 0), prediction.at<float>(3, 0));
    m_history.push_back(bbox);
    return bbox;
}

void KFTrackerConstantVelocity::update(cv::Rect2f bbox)
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

cv::Rect2f KFTrackerConstantVelocity::get_state()
{
    cv::Mat state = kf.statePost;
    return get_bbox(state.at<float>(0, 0), state.at<float>(1, 0), state.at<float>(2, 0), state.at<float>(3, 0));
}

cv::Rect2f KFTrackerConstantVelocity::get_bbox(float cx, float cy, float s, float r)
{
    float w = sqrt(s * r);
    float h = s / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);

    return cv::Rect2f(x, y, w, h);
}

void KFTrackerConstantAcceleration::init_kf(cv::Rect2f bbox, float process_noise_scale, float measurement_noise_scale, size_t time_step)
{

    size_t measureNum = 4;            // [x,y,w,h]
    size_t stateNum = measureNum * 3; // [x,y,w,h,dx,dy,dw,dh,ddx,ddy,ddw,ddh]
    kf = cv::KalmanFilter(stateNum, measureNum, 0);
    measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    float dt = static_cast<float>(time_step);
    cv::Mat dynamicsMatrix = (cv::Mat_<float>(3, 3) << 1, dt, dt * dt * 0.5,
                              0, 1, dt,
                              0, 0, 1);

    cv::Mat covMatrix = (cv::Mat_<float>(3, 3) << powf(dt, 6) / 36.f, powf(dt, 5) / 24.f, powf(dt, 4) / 6.f,
                         powf(dt, 5) / 24.f, powf(dt, 4) / 4.f, powf(dt, 3) / 2.f,
                         powf(dt, 4) / 6.f, powf(dt, 3) / 2.f, powf(dt, 2));

    covMatrix *= process_noise_scale;

    kf.transitionMatrix = cv::Mat::zeros(stateNum, stateNum, CV_32F);
    kf.processNoiseCov = cv::Mat::zeros(stateNum, stateNum, CV_32F);

    for (size_t i = 0; i < stateNum; i += 3)
    {
        dynamicsMatrix.copyTo(kf.transitionMatrix.rowRange(i, i + 3).colRange(i, i + 3));
        covMatrix.copyTo(kf.processNoiseCov.rowRange(i, i + 3).colRange(i, i + 3));
    }

    kf.measurementMatrix = cv::Mat::zeros(measureNum, stateNum, CV_32F);
    for (size_t i = 0; i < measureNum; ++i)
    {
        kf.measurementMatrix.at<float>(i, i * 3) = 1.f;
    }

    kf.measurementNoiseCov = cv::Mat::eye(measureNum, measureNum, CV_32F);
    kf.measurementNoiseCov *= measurement_noise_scale;

    kf.errorCovPre = cv::Mat_<float>::ones(stateNum, stateNum);

    // initialize state vector with bounding box in // [x, y, w, h] style
    kf.statePost = cv::Mat_<float>::zeros(stateNum, 1);
    kf.statePost.at<float>(0, 0) = bbox.x;
    kf.statePost.at<float>(3, 0) = bbox.y;
    kf.statePost.at<float>(6, 0) = bbox.width;
    kf.statePost.at<float>(9, 0) = bbox.height;
}

cv::Rect2f KFTrackerConstantAcceleration::predict()
{
    m_age++;
    m_time_since_update++;
    cv::Mat prediction = kf.predict();
    auto bbox = cv::Rect2f(prediction.at<float>(0, 0), prediction.at<float>(3, 0), prediction.at<float>(6, 0), prediction.at<float>(9, 0));
    m_history.push_back(bbox);
    return bbox;
}

void KFTrackerConstantAcceleration::update(cv::Rect2f bbox)
{
    m_time_since_update = 0;
    m_history.clear();

    // update measurement
    measurement.at<float>(0, 0) = bbox.x;
    measurement.at<float>(1, 0) = bbox.y;
    measurement.at<float>(2, 0) = bbox.width;
    measurement.at<float>(3, 0) = bbox.height;

    // update
    kf.correct(measurement);
}

cv::Rect2f KFTrackerConstantAcceleration::get_state()
{
    cv::Mat state = kf.statePost;
    return cv::Rect2f(state.at<float>(0, 0), state.at<float>(3, 0), state.at<float>(6, 0), state.at<float>(9, 0));
}