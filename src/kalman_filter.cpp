#include "kalman_filter.h"
#define SmallValue 0.0001

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
// https://discussions.udacity.com/t/rmse-too-high-with-simulator-v2/250330/5
  
//  double rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
//  double theta = atan(x_(1) / x_(0));
//  double rho_dot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
  
  double p_x = x_(0);
  double p_y = x_(1);
  double v_x = x_(2);
  double v_y = x_(3);
  double rho = sqrt(pow(p_x, 2) + pow(p_y, 2));
  
  if(fabs(p_x) < SmallValue) {
    p_x = SmallValue;
  }
  if (fabs(rho) < SmallValue){
    rho = SmallValue;
  }
  
  double theta = atan2(p_y, p_x);
  double rho_dot = (p_x * v_x + p_y * v_y) / (rho);
  
  VectorXd h = VectorXd(3); // h(x_)
  h << rho, theta, rho_dot;
  
  VectorXd y = z - h;
  
  if(fabs(y[1]) > M_PI){
    y[1] = atan2(sin(y[1]), cos(y[1]));
  }
  
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
