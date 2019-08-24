#include "model.h"
#include <omp.h>
#include <unordered_map>
double const Model::T = 293.0;
double const Model::kb = 1.38e-23;
double const Model::vis = 1e-3;





Model::Model(std::string fileName, int seed0){
    randomSeed = seed0;
    configName = fileName;
    std::ifstream ifile(this->configName);
    ifile >> config;
    ifile.close();

    readConfigFile();
    
    
    rand_normal = std::make_shared<std::normal_distribution<double>>(0.0, 1.0);

    mesh = std::make_shared<Mesh>();
    mesh->readMeshFile(meshName);
    mesh->initialize();
    


    for(int i = 0; i < numP; i++){
        particles.push_back(particle_ptr(new Model::particle));
   }
    
}

void Model::readConfigFile() {
    
    meshName = config["meshName"];
    filetag = config["filetag"];
    numP = config["N"];
    dimP = 3;
    radius = config["radius"];
    dt_ = config["dt"];
    diffusivity_t = config["diffusivity"];// this corresponds the diffusivity of 1um particle
    diffusivity_t /= pow(radius,2); // diffusivity will be in unit a^2/s
    Bpp = config["Bpp"];
    Bpp = Bpp * kb * T * 1e9; //2.29 is Bpp/a/kT
    Kappa = config["kappa"]; // here is kappa*radius
    Os_pressure = 0.0 * kb * T * 1e9;
    L_dep = 0.2; // 0.2 of radius size, i.e. 200 nm
    radius_nm = radius*1e9;
    combinedSize = (1+L_dep)*radius_nm;
    mobility = diffusivity_t/kb/T;
    trajOutputInterval = config["trajOutputInterval"];
    fieldStrength = config["fieldStrength"];
    fileCounter = 0;
    cutoff = config["cutoff"];
    trajOutputFlag = config["trajOutputFlag"];
    this->rand_generator.seed(randomSeed);
    srand(randomSeed);

    randomMoveFlag = config["randomMoveFlag"];

}


void Model::setPosition(int faceIdx, double q1, double q2) {

    particles[0]->meshFaceIdx = faceIdx;
    particles[0]->local_r[0] = q1;
    particles[0]->local_r[1] = q2;
    for (int i = 0; i < numP; i++) {
        particles[i]->r = mesh->coord_l2g[particles[i]->meshFaceIdx](particles[i]->local_r);        
    }

}

py::array_t<double> Model::getPosition() {

    std::vector<double> positions(3);

    positions[0] = particles[0]->r[0];
    positions[1] = particles[0]->r[1];
    positions[2] = particles[0]->r[2];

    py::array_t<double> result(3, positions.data());
    return result;
}

py::array_t<double> Model::findClosestPosition(double x, double y, double z, double thresh){
    
    while (1) {
        int faceIdx = rand() / mesh->numF;
        Eigen::Vector3d P1 = mesh->V.row(mesh->F(faceIdx,0)).transpose().eval();
        Eigen::Vector3d P2 = mesh->V.row(mesh->F(faceIdx,1)).transpose().eval();
        Eigen::Vector3d P3 = mesh->V.row(mesh->F(faceIdx,2)).transpose().eval();
        Eigen::Vector3d Center = (P1 + P2 + P3) / 3.0;
        Eigen::Vector3d Target{x, y, z};
        double dist = (Center - Target).norm();
        if (dist < thresh) {
            std::vector<double> positions(3);

            positions[0] = Center(0);
            positions[1] = Center(1);
            positions[2] = Center(2);

            py::array_t<double> result(3, positions.data());
            return result;            
        }   
    }    
}

int Model::findClosestFace(double x, double y, double z, double thresh){
    
    while (1) {
        int faceIdx = rand() % mesh->numF;
        Eigen::Vector3d P1 = mesh->V.row(mesh->F(faceIdx,0)).transpose().eval();
        Eigen::Vector3d P2 = mesh->V.row(mesh->F(faceIdx,1)).transpose().eval();
        Eigen::Vector3d P3 = mesh->V.row(mesh->F(faceIdx,2)).transpose().eval();
        Eigen::Vector3d Center = (P1 + P2 + P3) / 3.0;
        Eigen::Vector3d Target{x, y, z};
        double dist = (Center - Target).norm();
        if (dist < thresh) {
            return faceIdx;            
        }   
    }    
}

void Model::run() {
    if (this->timeCounter == 0 || ((this->timeCounter + 1) % trajOutputInterval == 0)) {
        this->outputTrajectory(this->trajOs);
    }

    calForces(); 
    for (int i = 0; i < numP; i++) {            

        for (int j = 0; j < 3; j++){
           particles[i]->vel(j) = diffusivity_t * particles[i]->F(j);
           if (randomMoveFlag) {
            particles[i]->vel(j) += sqrt(2.0 * diffusivity_t/dt_) *(*rand_normal)(rand_generator);
           }
        }           
        this->moveOnMeshV2(i);
    }
    this->timeCounter++;
    
}

void Model::run(int steps){
    for (int i = 0; i < steps; i++){
	run();
    }
}

void Model::step_given_field(int steps, double x, double y, double z) {
    
    externalField[0] = x;
    externalField[1] = y;
    externalField[2] = z;
    
    for (int i = 0; i < steps; i++){
	
        if (trajOutputFlag) {
            if (this->timeCounter == 0 || ((this->timeCounter + 1) % trajOutputInterval == 0)) {
                this->outputTrajectory(this->trajOs);
            }
        }
        calForces(); 
        for (int i = 0; i < numP; i++) {            
            particles[i]->F(0) += fieldStrength * x / diffusivity_t;
            particles[i]->F(1) += fieldStrength * y / diffusivity_t;
            particles[i]->F(2) += fieldStrength * z / diffusivity_t;


            for (int j = 0; j < 3; j++){
               particles[i]->vel(j) = diffusivity_t * particles[i]->F(j);
               if (randomMoveFlag) {
               particles[i]->vel(j) += sqrt(2.0 * diffusivity_t/dt_) *(*rand_normal)(rand_generator);
               }
            }           
            this->moveOnMeshV2(i);
        }
        this->timeCounter++;
    }
}



void Model::moveOnMeshV2(int p_idx){
    // first calculate the tangent velocity
    Eigen::Vector3d velocity = particles[p_idx]->vel;
    int meshIdx = this->particles[p_idx]->meshFaceIdx;
    Eigen::Vector3d normal = mesh->F_normals.row(meshIdx).transpose();
    Eigen::Vector3d tangentV = velocity - normal*(normal.dot(velocity));
    // local velocity representation
    Eigen::Vector2d localV = mesh->Jacobian_g2l[meshIdx]*tangentV;
    Eigen::Vector2d localQ_new;
    
    double t_residual = this->dt_;
    // the while loop will start with particle lying on the surface
    // need to do wraping do ensure new particle position is finally lying on a surface
    
    if (!mesh->inTriangle(particles[p_idx]->local_r)){
        std::cerr << this->timeCounter << " not in triangle before the loop!" << std::endl;
        std::cout <<  particles[p_idx]->local_r << std::endl;
    }
    double positionPrecision = 1e-8;
    
    while(t_residual > 1e-8){
        // move with local tangent speed
        localQ_new = particles[p_idx]->local_r + localV * t_residual;
        if (mesh->inTriangle(localQ_new)){
            t_residual = 0.0;
            // to avoid tiny negative number
            localQ_new(0) = abs(localQ_new(0));
            localQ_new(1) = abs(localQ_new(1));
            particles[p_idx]->local_r = localQ_new;
            break;
        } else {
            // if localQ_new(0) + localQ_new(1) > 1.0, then must hit edge 1
            // if localQ_new(0) < 0 and localQ_new(1) > 0, then must hit edge 2
            // if localQ_new(0) > 0 and localQ_new(1) < 0, then must hit edge 0
            // if localQ_new(0) < 0 and localQ_new(1) < 0, then we might hit edge 0 and 2
            
            
            // as long as the velocity vector is not parallel to the three edges, there will a collision
            Eigen::Vector3d t_hit;
            // t_hit will be negative if it move away from the edge
            t_hit(2) = -particles[p_idx]->local_r(0) / localV(0); // this second edge
            t_hit(0) = -particles[p_idx]->local_r(1) / localV(1); // the zero edge
            t_hit(1) = (1 - particles[p_idx]->local_r.sum()) / localV.sum(); // the first edge
            
            double t_min = t_residual;
            int min_idx = -1;
            // the least positive t_hit will hit
            for (int i = 0; i < 3; i++){
                if (t_hit(i) > 1e-12 && t_hit(i) <= t_min){
                    t_min = t_hit(i);
                    min_idx = i;
                }            
            }

            if (min_idx < 0){
                // based on above argument, at least one edge will be hitted
                std::cerr << this->timeCounter << "\t t_hit is not determined!" << std::endl;
                std::cerr << t_hit(0) << "\t" << t_hit(1) << "\t" << t_hit(2) << "\t tmin " << t_min << std::endl;
                std::cout <<  particles[p_idx]->local_r << std::endl;
                std::cout << localQ_new << std::endl;
                std::cout << localV << std::endl;
                
                break;
            }
            
            t_residual -= t_min;

            // here update the local coordinate
            particles[p_idx]->local_r += localV * t_min;
            // here is the correction step to make simulation stable
            if( min_idx == 0){
                // hit the first edge
                particles[p_idx]->local_r(1) = 0.0;                
            } else if( min_idx == 1){
                // hit the second edge, local_r(0) + local_r(1) = 1.0
                particles[p_idx]->local_r(0) = 1.0 - particles[p_idx]->local_r(1);
            } else{
                // hit the third edge
                particles[p_idx]->local_r(0) = 0.0;
            }
            
            int meshIdx = particles[p_idx]->meshFaceIdx;
            int newMeshIdx = mesh->TT(meshIdx, min_idx);
            
               // if hit the boundary of the mesh surface, then we should stop
            if (newMeshIdx < 0) {
                // we do some safety adjustment
                if (abs(particles[p_idx]->local_r(0)) < positionPrecision) {
                    particles[p_idx]->local_r(0) = 2 * positionPrecision;
                } else if (abs(particles[p_idx]->local_r(1)) < positionPrecision) {
                    particles[p_idx]->local_r(1) = 2 * positionPrecision;
                } else if (abs((1 - particles[p_idx]->local_r.sum())) < positionPrecision){
                    particles[p_idx]->local_r(1) = 1.0 - positionPrecision - particles[p_idx]->local_r(0);
                    particles[p_idx]->local_r(0) = 1.0 - positionPrecision - particles[p_idx]->local_r(1);
                }
                break;
            }


#ifdef DEBUG
            
            int reverseIdx = mesh->TTi(meshIdx,min_idx);
            Eigen::Vector2d newV = mesh->localtransform_v2v[meshIdx][min_idx]*localV;
            Eigen::Vector2d oldV = mesh->localtransform_v2v[mesh->TT(meshIdx,min_idx)][reverseIdx]*newV;
            
            double diff1 = (localV - oldV).norm();
            
          
            if (diff1 > 1e-6) {
                std::cerr << this->timeCounter << " speed transformation error! " << diff1 << std::endl;
            }
           
            
#endif   
            // transform to the local tangent speed in the new plane
            localV = mesh->localtransform_v2v[meshIdx][min_idx]*localV;

            // transform to local coordinate in the new plane
            // because the local coordinate is on the edge of the old plane; it also must be in the edge of new plane
            
            particles[p_idx]->local_r = mesh->localtransform_p2p[meshIdx][min_idx](particles[p_idx]->local_r);
            particles[p_idx]->meshFaceIdx = newMeshIdx;
            
            
            if (abs(particles[p_idx]->local_r(0)) < positionPrecision) {
                particles[p_idx]->local_r(0) = 2 * positionPrecision;
            } else if (abs(particles[p_idx]->local_r(1)) < positionPrecision) {
                particles[p_idx]->local_r(1) = 2 * positionPrecision;
            } else if (abs((1 - particles[p_idx]->local_r.sum())) < positionPrecision){
                particles[p_idx]->local_r(1) = 1.0 - positionPrecision - particles[p_idx]->local_r(0);
                particles[p_idx]->local_r(0) = 1.0 - positionPrecision - particles[p_idx]->local_r(1);
                
            } else {
                std::cerr << this->timeCounter << " not in triangle after wrapping!" << std::endl;
                std::cout <<  particles[p_idx]->local_r << std::endl;
            }
#ifdef DEBUG
            if (!mesh->inTriangle(particles[p_idx]->local_r)){
                std::cerr << "not in triangleafter wrapping and adjustment " << std::endl;
            }
#endif            

        }
        
#ifdef DEBUG3
        particles[p_idx]->r = mesh->coord_l2g[particles[p_idx]->meshFaceIdx](particles[p_idx]->local_r);
        this->outputTrajectory(this->trajOs);
#endif        
        
        

    }



    if (!mesh->inTriangle(particles[p_idx]->local_r)){
        std::cerr << this->timeCounter << " not in triangle after the loop!" << std::endl;
        std::cout <<  particles[p_idx]->local_r << std::endl;
    }
    particles[p_idx]->r = mesh->coord_l2g[particles[p_idx]->meshFaceIdx](particles[p_idx]->local_r);
}


void Model::calForcesHelper(int i, int j, Eigen::Vector3d &F) {
    double dist;
    Eigen::Vector3d r;

    dist = 0.0;
    F.fill(0);
    r = particles[j]->r - particles[i]->r;
    dist = r.norm();
            
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t" << j << "\t"<< this->timeCounter << "dist: " << dist << "\t" << this->timeCounter <<std::endl;

        dist = 2.06;
    }
    if (dist < cutoff) {
        // the unit of force is kg m s^-2
        // kappa here is kappa*a a non-dimensional number
        
        double Fpp = -4.0/3.0*
        Os_pressure*M_PI*(-3.0/4.0*pow(combinedSize,2.0)+3.0*dist*dist/16.0*radius_nm*radius_nm);
        Fpp = -Bpp * Kappa * exp(-Kappa*(dist-2.0));
//        Fpp += -9e-13 * exp(-kappa* (dist - 2.0));
        F = Fpp*r/dist;
    }
}

void Model::calForces() {
     
    for (int i = 0; i < numP; i++) {
        particles[i]->F.fill(0);
    }
    Eigen::Vector3d F;
    for (int i = 0; i < numP - 1; i++) {
        for (int j = i + 1; j < numP; j++) {
            calForcesHelper(i, j, F);
            particles[i]->F += F;
            particles[j]->F -= F;

        }
               
    }

    
}
    


void Model::createInitialState(){

    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    ss << this->fileCounter++;
    if (trajOs.is_open()) trajOs.close();
//    if (opOs.is_open()) opOs.close();
    if (trajOutputFlag) {
        this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
    }
//    this->opOs.open(filetag + "op" + ss.str() + ".txt");
    this->timeCounter = 0;

              
}

void Model::outputTrajectory(std::ostream& os) {

    for (int i = 0; i < numP; i++) {
        os << i << "\t";
        for (int j = 0; j < 2; j++){
            os << particles[i]->local_r[j] << "\t";
        }
        os << particles[i]->meshFaceIdx << "\t";
//        particles[i]->r = mesh->coord_l2g[particles[i]->meshFaceIdx](particles[i]->local_r); 
        for (int j = 0; j < 3; j++){
            os << particles[i]->r(j) << "\t";
        }
        for (int j = 0; j < 3; j++){
            os << externalField[j] << "\t";
        }
        os << this->timeCounter*this->dt_ << "\t";
        os << std::endl;
    }
}


