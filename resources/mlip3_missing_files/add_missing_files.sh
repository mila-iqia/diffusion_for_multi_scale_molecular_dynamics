#!/bin/bash

cp mlip3_missing_files/CMakeLists.txt mlip-3/CMakeLists.txt
cp mlip3_missing_files/src_CMakeLists.txt mlip-3/src/CMakeLists.txt
cp mlip3_missing_files/test_CMakeLists.txt mlip-3/test/CMakeLists.txt
cp mlip3_missing_files/src_mlp_CMakeLists.txt mlip-3/src/mlp/CMakeLists.txt
cp mlip3_missing_files/src_common_CMakeLists.txt mlip-3/src/common/CMakeLists.txt
cp mlip3_missing_files/src_drivers_CMakeLists.txt mlip-3/src/drivers/CMakeLists.txt
cp mlip3_missing_files/src_external_CMakeLists.txt mlip-3/src/external/CMakeLists.txt
cp mlip3_missing_files/test_selftest_CMakeLists.txt mlip-3/test/self-test/CMakeLists.txt
cp mlip3_missing_files/test_examples_CMakeLists.txt mlip-3/test/examples/CMakeLists.txt
cp mlip3_missing_files/test_examples_00_CMakeLists.txt mlip-3/test/examples/00.convert_vasp_outcar/CMakeLists.txt
cp mlip3_missing_files/test_examples_01_CMakeLists.txt mlip-3/test/examples/01.train/CMakeLists.txt
cp mlip3_missing_files/test_examples_02_CMakeLists.txt mlip-3/test/examples/02.check_errors/CMakeLists.txt
cp mlip3_missing_files/test_examples_03_CMakeLists.txt mlip-3/test/examples/03.calculate_efs/CMakeLists.txt
cp mlip3_missing_files/test_examples_04_CMakeLists.txt mlip-3/test/examples/04.calculate_grade/CMakeLists.txt
cp mlip3_missing_files/test_examples_05_CMakeLists.txt mlip-3/test/examples/05.cut_extrapolative_neighborhood/CMakeLists.txt
cp mlip3_missing_files/test_examples_06_CMakeLists.txt mlip-3/test/examples/06.select_add/CMakeLists.txt
cp mlip3_missing_files/test_examples_07_CMakeLists.txt mlip-3/test/examples/07.relax/CMakeLists.txt
cp mlip3_missing_files/test_examples_08_CMakeLists.txt mlip-3/test/examples/08.relax_preselect/CMakeLists.txt
cp -r mlip3_missing_files/cmake mlip-3/
cp mlip3_missing_files/mlp_commands.cpp mlip-3/src/mlp/mlp_commands.cpp

