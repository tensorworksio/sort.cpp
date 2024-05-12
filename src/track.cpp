#include <track.hpp>

int Track::kf_count = 0;

void Track::init_kf(cv::Rect2f bbox) 
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

    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    // initialize state vector with bounding box in [cx,cy,s,r] style
    kf.statePost.at<float>(0, 0) = bbox.x + bbox.width / 2;
    kf.statePost.at<float>(1, 0) = bbox.y + bbox.height / 2;
    kf.statePost.at<float>(2, 0) = bbox.area();
    kf.statePost.at<float>(3, 0) = bbox.width / bbox.height;    
}

cv::Rect2f Track::predict()
{
    cv::Mat prediction = kf.predict();
    m_age++;

    if (m_time_since_update > 0)
    {
        m_hit_streak = 0;
    }

    m_time_since_update++;
    cv::Rect2f predicted_bbox = get_rect_xysr(prediction.at<float>(0, 0), prediction.at<float>(1, 0), prediction.at<float>(2, 0), prediction.at<float>(3, 0));
    m_history.push_back(predicted_bbox);

    return predicted_bbox;
}

void Track::update(cv::Rect2f bbox)
{
    m_time_since_update = 0;
    m_hits++;
    m_hit_streak++;
    m_history.clear();

    // update measurement
    measurement.at<float>(0, 0) = bbox.x + bbox.width / 2;
    measurement.at<float>(1, 0) = bbox.y + bbox.height / 2;
    measurement.at<float>(2, 0) = bbox.area();
    measurement.at<float>(3, 0) = bbox.width / bbox.height;

    // update
    kf.correct(measurement);
}

cv::Rect2f Track::get_state()
{
    cv::Mat state = kf.statePost;
    return get_rect_xysr(state.at<float>(0, 0), state.at<float>(1, 0), state.at<float>(2, 0), state.at<float>(3, 0));
}

cv::Rect2f Track::get_rect_xysr(float cx, float cy, float s, float r)
{
    float w = sqrt(s * r);
    float h = s / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);

    if (x < 0 && cx > 0)
        x = 0;
    if (y < 0 && cy > 0)
        y = 0;

    return cv::Rect2f(x, y, w, h);
}