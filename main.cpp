#include <igl/readOFF.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>
#include <igl/adjacency_list.h>
#include "igl/unique_edge_map.h"
#include <iostream>
#include "igl/all_edges.h"
#include "igl/edge_flaps.h"
#include "igl/edges.h"
#include "igl/triangle_triangle_adjacency.h"
#include "igl/boundary_facets.h"
#include "igl/grad.h"
#include "Mesh.h"
#include "common.h"
#include "model.h"
#include <omp.h>
#include <cstdlib>
Parameter parameter;
Parameter_cell parameter_cell;
void readParameter();
void testIO();
void testMyMesh();
void testModel();
void testDiffusionStat(int step);
void testMultiPModel();


int main(int argc, char *argv[])
{
    testModel();
//    testMyMesh();
//    testMoveOnSphere();
//    testMultiPModel();
////    testModel();
//    if (argc <= 1) 
//    {
//        std::cout << "no argument" << std::endl;
//        exit(1);
//    }
//    
//    
//   
//    std::string option = std::string(argv[1]);
//    
//    std::cout << "option is: " << option << std::endl;
//    if (option.compare("singleP") == 0){
//        testModel();
//    } else if (option.compare("multiP") == 0) {
//        testMultiPModel();
//    } else if (option.compare("Diffusion") == 0) {
//        int steps = strtol(argv[2], nullptr, 0);
//        testDiffusionStat(steps);
//    } else if (option.compare("SphereMove") == 0) {
//        testMoveOnSphere();
//    }
//    else {
//        std::cout << "option unknown! " << option << std::endl;
//    }
    
    
    
//    testIO();
//    testMyMesh();
//    testPDESolver();
    
//    testModel_cell_on2d();
//    testOpenMP();
//    testModelOpenMP();
//    testConfigGeneration();
    return 0;
}


void testModel(){
    Model m("config.json", 1);
    
    for (int c_idx = 0; c_idx < 1; c_idx++){
        m.createInitialState();
        int idx = m.findClosestFace(8.428, 0.0, -21.13, 100);
        m.setPosition(idx, 0.25, 0.25);
//        m.MCRelaxation();
        for (int i = 0; i < 100; i++){
            Eigen::Vector3d field = Eigen::Vector3d::Random();
            m.step_given_field(1, field(0), field(1), field(2));    
        }
    }

}



void testMyMesh(){
    Eigen::Vector3d v;
    std::cout << v << std::endl;
    
    Mesh mesh;
    mesh.readMeshFile("square/square.off");
    mesh.initialize();
    for (int i = 0; i < mesh.numF; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << mesh.TT(i, j) << std::endl;
        }
    }

}

void testIO() {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd F_normals;
    Eigen::MatrixXi E, EF, EI, TT, TTi;
    Eigen::MatrixXi uE;
    //Eigen::MatrixXi EMap;
    Eigen::VectorXi EMap;
    std::vector<std::vector<int>> uE2E;
    // Load a mesh in OFF format
    igl::readOFF("2dmesh/circle.off", V, F);

    // Print the vertices and faces matrices
    std::cout << "Vertices: " << std::endl << V << std::endl;
    std::cout << "Faces:    " << std::endl << F << std::endl;
    std::cout << "num of Vertices: " << V.rows() << std::endl;
    std::cout << "num of faces: " << F.rows() << std::endl;

    std::vector<std::vector<int> > adj_list;
    // adjacency is used for obtain the adjacency relationship for vertices
    igl::adjacency_list(F, adj_list);

    // E contains unordered directed edges    
    igl::all_edges(F, E);
    std::cout << "directed EDGES: " << E << std::endl;
    //   EMAP #F*3 list of indices into uE, mapping each directed edge to unique
    //     undirected edge
    igl::unique_edge_map(F, E, uE, EMap, uE2E);



    // Inputs:
    //   F  #F by simplex_size list of mesh faces (must be triangles)
    // Outputs:
    //   TT   #F by #3 adjacent matrix, the element i,j is the id of the triangle adjacent to the j edge of triangle i
    //   TTi  #F by #3 adjacent matrix, the element i,j is the id of edge of the triangle TT(i,j) that is adjacent with triangle i
    // NOTE: the first edge of a triangle is [0,1] the second [1,2] and the third [2,3]. it is zero based.
    //       this convention is DIFFERENT from cotmatrix_entries.h
    igl::triangle_triangle_adjacency(F, TT, TTi);
    std::cout << "TT: " << std::endl << TT << std::endl;
    std::cout << "TTi:    " << std::endl << TTi << std::endl;



    igl::edge_flaps(F, E, EMap, EF, EI);

    // undirected edges does not contained repeated edges
    std::cout << "undirected Edges: " << uE << std::endl;
    std::cout << "Emap:  " << EMap << std::endl;
    std::cout << "uE to E" << std::endl;
    for (int i = 0; i < uE2E.size(); i++) {
        std::cout << "uE idx: " << i << std::endl;
        for (int j = 0; j < uE2E[i].size(); j++) {
            std::cout << uE2E[i][j] << std::endl;
        }

    }
    std::cout << "EF: " << std::endl << EF << std::endl;
    std::cout << "EI:    " << std::endl << EI << std::endl;

    for (int j = 0; j < adj_list.size(); j++) {
        std::cout << "vertex idx: " << j << std::endl;
        for (int i = 0; i < adj_list[j].size(); i++) {
            std::cout << adj_list[j][i] << std::endl;

        }
    }
    // calculate normals of faces
    igl::per_face_normals(V, F, F_normals);
    std::cout << "Faces normals:    " << std::endl << F_normals << std::endl;

    // Save the mesh in OBJ format
    igl::writeOBJ("2triangles.obj", V, F);


    // obtain the boundary edges/faces for triangles/tetrahedron
     Eigen::MatrixXi bF;
     igl::boundary_facets(F,bF);
     std::cout << "boundary edge vertices: " << std::endl << bF << std::endl;
    
    // calculate the numerical gradient at every face of a triangle mesh
    // G = grad(V,F,X)
    //
    // Compute the numerical gradient at every face of a triangle mesh.
    //
    // Inputs:
    // V #vertices by 3 list of mesh vertex positions
    // F #faces by 3 list of mesh face indices
    // X # vertices list of scalar function values
    // Outputs:
    // G #faces by 3 list of gradient values
     
    Eigen::SparseMatrix<double> G;
    igl::grad(V,F,G);
//    std::cout << "gradient operator: " << std::endl << G << std::endl; 
    // 
    // Compute the numerical gradient operator 
    // 
    // Inputs: 
    // V #vertices by 3 list of mesh vertex positions 
    // F #faces by 3 list of mesh face indices 
    // Outputs: 
    // G #faces*dim by #V Gradient operator 

}


