#!/bin/bash

cd alpha0_1
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff



cd ../alpha0_01
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff



cd ../alpha0_001
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff


cd ../alpha0_0001
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff


cd ../alpha1
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff


cd ../alpha1e-05
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff



cd ../alpha1e-06
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff



cd ../alpha1e-07
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff



cd ../alpha1e-08
python3 ComputeAnalyticExpressions.py -parameters_hpp ./parameters.hpp
cmake .
make
./adv_diff




