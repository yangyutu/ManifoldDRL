CC = gcc
CXX = g++

HOME=/home/yangyutu/
IGL_INCLUDE=-I./libigl/include

DEBUGFLAG=-DDEBUG -g3
RELEASEFLAG= -O3 
CXXFLAGS=  -std=c++0x $(BOOST_INCLUDE) $(IGL_INCLUDE)  -D__LINUX -fopenmp `python-config --cflags` `/home/yangyutu/anaconda3/bin/python -m pybind11 --includes`

LDFLAG= -fopenmp -lpthread  `python-config --ldflags`

OBJ=main.o Mesh.o model.o
all:test.exe 
test.exe: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAG) 
	
test_static: $(OBJ)
	$(CXX) -o $@ $^ -static $(LDFLAG) -lgomp -lm -ldl 
	
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $(DEBUGFLAG) $^
	


clean:
	rm *.o *.exe
	
