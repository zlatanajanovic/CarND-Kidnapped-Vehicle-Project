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
	num_particles = 10;
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
		  
		  
	double weight_sum = 0;

	for (int i=0; i<num_particles; ++i) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		std::vector<LandmarkObs> predicted;
		
		// Calculate map_landmarks in vehicle's cooridnate assuming particle's state.
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			double landmark_x = map_landmarks.landmark_list[j].x_f;
			double landmark_y = map_landmarks.landmark_list[j].y_f;
			if (dist(x, y, landmark_x, landmark_y) < sensor_range) {
				LandmarkObs landmark;
				landmark.x = cos(-theta) * (landmark_x - x) - sin(-theta) * (landmark_y - y);
				landmark.y = sin(-theta) * (landmark_x - x) + cos(-theta) * (landmark_y - y);
				landmark.id = map_landmarks.landmark_list[j].id_i;
				predicted.push_back(landmark);
			}
		}

		// Associate observation with map_landmark (estimated in vehicle coordinate).
		dataAssociation(predicted, observations);

		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		// Associate particle with each observation.
		for (int j = 0; j < observations.size(); ++j) {
			double o_x = observations[j].x;
			double o_y = observations[j].y;

			sense_x.push_back(cos(theta) * o_x - sin(theta) * o_y + x);
			sense_y.push_back(sin(theta) * o_x + cos(theta) * o_y + y);
			associations.push_back(observations[j].id);
		}
		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);

		// Calculate weights.
		particles[i].weight = 1.0;
		for (int j = 0; j < observations.size(); ++j) {
			double sig_x2 = std_landmark[0]*std_landmark[0];
			double sig_y2 = std_landmark[1]*std_landmark[1];
			double dx = 0, dy = 0;

			// Search associated landmark and calculate difference.
			for (int k = 0; k < predicted.size(); ++k) {
				if (observations[j].id == predicted[k].id) {
					dx = observations[j].x - predicted[k].x;
					dy = observations[j].y - predicted[k].y;
					break;
				}
				else if (k == predicted.size() - 1) {
					cout << i << " " << j << " Association not found!" << endl;
				}
			}

			particles[i].weight *= exp(-0.5 * (1.0/sig_x2*dx*dx + 1.0/sig_y2*dy*dy)) / sqrt(pow(2.0*M_PI,observations.size())*sig_x2*sig_y2);
		}

		// cout << i << ", " << particles[i].weight << endl;

		weight_sum += particles[i].weight;
	}

	for (int i = 0; i < num_particles; ++i) {
		particles[i].weight /= weight_sum;
	}	  
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
		
	default_random_engine gen
	
	for (int i = 0; i < weights.size(); ++i) {
		weights[i] = particles[i].weight;
	}

	discrete_distribution<> dist(weights.begin(), weights.end());

	// Copy current particles
	std::vector<Particle> old_particles;
	copy(particles.begin(), particles.end(), back_inserter(old_particles));

	// Resample according to weights
	for (int i = 0; i < particles.size(); ++i) {
		particles[i] = old_particles[dist(gen)];
	}
	
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
