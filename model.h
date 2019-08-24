#pragma once
#include<vector>
#include<memory>
#include<random>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Mesh.h"
#include "Eigen/Dense"
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>



using json = nlohmann::json;
namespace py = pybind11;

class Model {
public:

    struct pos{
        double r[3];
        pos(double x = 0, double y = 0, double z = 0){
            r[0]=x;r[1]=y;r[2]=z;
        }
    };
    
    struct particle {
        Eigen::Vector3d r, F, vel;
        Eigen::Vector2d local_r;
        int meshFaceIdx;
        bool free;
        
        particle(){
            r.fill(0);
            local_r.fill(0);
            meshFaceIdx = 0;
        }
        
        particle(double x, double y, double z, int idx){
            r(0)=x;r(1)=y;r(2)=z;
            meshFaceIdx = idx;
        }
        
        particle(double q1,double q2, int idx){
            local_r(0) = q1;
            local_r(1) = q2;
            meshFaceIdx = idx;
        }
    };
    typedef std::shared_ptr<particle> particle_ptr;
    typedef std::vector<particle_ptr> state;
    typedef std::vector<std::shared_ptr<pos>> posArray;
   
    Model(){}
    Model(std::string fileName, int seed0);
    virtual ~Model() {trajOs.close();
    opOs.close(); osTarget.close();
    }

    virtual void moveOnMeshV2(int p_idx);
    py::array_t<double> findClosestPosition(double x, double y, double z, double thresh);
    int findClosestFace(double x, double y, double z, double thresh);
    void step_given_field(int steps, double x, double y, double z);
    virtual void run();
    virtual void run(int steps);
    virtual void createInitialState();
    py::array_t<double> getPosition();
    void setPosition(int faceIdx, double q1, double q2);
    void readConfigFile();
     
    std::shared_ptr<Mesh> mesh;
    state particles;
    json config;
protected:
    virtual void calForces();
    virtual void calForcesHelper(int i, int j, Eigen::Vector3d &F);
    int dimP, randomSeed;
    std::string configName, meshName;
    static const double kb, T, vis;
    int numP, numObstacles;
    double radius, radius_nm;
    double LJ,rm;
    double Bpp; //2.29 is Bpp/a/kT
    double Kappa; // here is kappa*radius
    double Os_pressure;
    double L_dep; // 0.2 of radius size, i.e. 200 nm
    double combinedSize;
    std::vector<double> velocity={0.0,2.0e-6,5.0e-6};
    std::vector<double> externalField={0.0,0.0,0.0};
    bool trajOutputFlag, randomMoveFlag;
    posArray obstacles; 
    std::vector<int> control;
    std::string iniFile;
    double dt_, cutoff, mobility, diffusivity_r, diffusivity_t, fieldStrength;
    std::default_random_engine rand_generator;
    std::shared_ptr<std::normal_distribution<double>> rand_normal;
    int trajOutputInterval;
    int timeCounter,fileCounter;
    std::ofstream trajOs, opOs, osTarget;
    std::string filetag;
    virtual void outputTrajectory(std::ostream& os);
    
};


