/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	 
	// Number of particles
	num_particles = 50;
	is_initialized = true;

	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// loop to create particles
	for (int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(particle.weight);

		// cout << "Sample " << i + 1 << ": " << particle.x << endl;
	}


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	
	default_random_engine gen;
	
	// loop trough all particles
	for (int i = 0; i < num_particles; ++i) {
		// If yaw rate differs from zero
		if (abs(yaw_rate) >= 0.001) {
			particles[i].x += velocity / yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		// If yaw_rate is close to zero
		else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
			particles[i].theta += yaw_rate * delta_t;
		}

		// Adding Gaussian noise
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
		// particles with noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		// cout << i << ", " << particles[i].y << endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	
	// loop trough observations
	for (int i=0; i< observations.size(); ++i) {
		// min distance set to inf
		double min_dist = 1.7e308;
		double X = observations[i].x;
		double Y = observations[i].y;
		// loop trough predictions of landmarks
		for (int j = 0; j < predicted.size(); ++j) {
			double x = predicted[j].x;
			double y = predicted[j].y;
			double dis = dist(X, Y, x, y);
			
			// if it is closer than previous min
			if (dis < min_dist) {
				observations[i].id = predicted[j].id;
				min_dist = dis;
			}
		}
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
		  
		  
	double std_r = std_landmark[0];
	double std_theta = std_landmark[1];
	double norm_factor = 0.0;

	for (int i=0; i < num_particles; ++i){
		Particle& particle_i = particles[i];
		vector<LandmarkObs> predicted;
		// loop trough landmarks
		for (int i=0; i< map_landmarks.landmark_list.size(); ++i) { 
			LandmarkObs landmark;
			landmark = map_landmarks.landmark_list[i];
			if (dist(particle_i.x, particle_i.y, landmark.x_f, landmark.y_f) < sensor_range) {
				predicted.push_back(landmark);
			}
		}
		// transformed observations for a particle
		vector<LandmarkObs> transformed_obs;
		for (int i=0; i< observations.size(); ++i) {
			LandmarkObs obs; 
			obs = observations[i];
			LandmarkObs trans_obs;
			trans_obs.x = cos(particle_i.theta) * obs.x - sin(particle_i.theta) * obs.y + particle_i.x;
			trans_obs.y = sin(particle_i.theta) * obs.x + cos(particle_i.theta) * obs.y + particle_i.y;

			transformed_obs.push_back(trans_obs);
		  }
		  dataAssociation(predicted, transformed_obs);

		double particle_weight = 1.0;

		double landmark_x, landmark_y;
		for (int i=0; i< transformed_obs.size(); ++i) {
			LandmarkObs obs;
			obs = observations[i];
			// finding corresponding landmark
			for (int i=0; i< predicted.size(); ++i) { 
				LandmarkObs landmark;
				landmark = map_landmarks.landmark_list[i];
				if (obs.id == landmark.id) {
					landmark_x = landmark.x;
					landmark_y = landmark.y;
					break;
				}
			}
			double prob = exp( -( pow(obs.x - landmark_x, 2) / (2 * std_x * std_x) + pow(obs.y - landmark_y, 2) / (2 * std_y * std_y) ) );

			particle_weight *= prob;

			particle_i.weight = particle_weight;
		}
		weights[i] = particle_i.weight;
		norm_factor += particle_i.weight;
	  }
	  
	// Normalize weights
	if (norm_factor>0.001){
		for (int i=0; i < num_particles; ++i){
			Particle& particle_i = particles[i];
			particle_i.weight /= norm_factor;
		}
	}	  
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());
	vector<Particle> particles_res[num_particles];

	for (int n=0; n < num_particles; ++n) {
		particles_res[n] = particles[d(gen)];
	}

	// Assigning resampled particles
	particles = particles_res;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
