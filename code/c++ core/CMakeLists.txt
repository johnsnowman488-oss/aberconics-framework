cmake_minimum_required(VERSION 3.16)
project(gfe_core VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GFE_BUILD_TESTS "Build local smoke tests" ON)
option(GFE_BUILD_APPS "Build CLI tools" ON)
option(GFE_ENABLE_SPACE_CLEAN_TARGETS "Enable cache/artifact cleanup targets" ON)

add_library(gfe_core
    src/gfe_core.cpp
    src/gfe_backend.cpp
    src/gfe_c_api.cpp
    src/gfe_dynamics.cpp
    src/gfe_gst.cpp
    src/gfe_assumptions.cpp
    src/gfe_energy.cpp
    src/gfe_verification.cpp
    src/abersoe_config.cpp
    src/abersoe_diagnostics.cpp
    src/abersoe_hooks.cpp
    src/abersoe_learning.cpp
    src/abersoe_runtime.cpp
    src/abersoe_scenarios.cpp
    src/abersoe_regression.cpp
    src/hierarchical_min.cpp
    src/hierarchical_min_coupling.cpp
    src/hierarchical_min_diagnostics.cpp
    src/hierarchical_min_renorm.cpp
)
add_library(gfe::gfe_core ALIAS gfe_core)

target_include_directories(gfe_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(gfe_core PROPERTIES
    OUTPUT_NAME gfe_core
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(gfe_core PRIVATE GFE_CORE_EXPORTS)
else()
    target_compile_definitions(gfe_core PUBLIC GFE_CORE_STATIC)
endif()

if(GFE_BUILD_TESTS)
    enable_testing()
    add_executable(gfe_smoke_test tests/smoke_test.cpp)
    target_link_libraries(gfe_smoke_test PRIVATE gfe_core)
    add_test(NAME gfe_smoke_test COMMAND gfe_smoke_test)

    add_executable(gst_smoke_test tests/gst_smoke_test.cpp)
    target_link_libraries(gst_smoke_test PRIVATE gfe_core)
    add_test(NAME gst_smoke_test COMMAND gst_smoke_test)

    add_executable(gfe_foundation_smoke_test tests/gfe_foundation_smoke_test.cpp)
    target_link_libraries(gfe_foundation_smoke_test PRIVATE gfe_core)
    add_test(NAME gfe_foundation_smoke_test COMMAND gfe_foundation_smoke_test)

    add_executable(gfe_backend_smoke_test tests/gfe_backend_smoke_test.cpp)
    target_link_libraries(gfe_backend_smoke_test PRIVATE gfe_core)
    add_test(NAME gfe_backend_smoke_test COMMAND gfe_backend_smoke_test)

    add_executable(gfe_c_api_smoke_test tests/gfe_c_api_smoke_test.cpp)
    target_link_libraries(gfe_c_api_smoke_test PRIVATE gfe_core)
    add_test(NAME gfe_c_api_smoke_test COMMAND gfe_c_api_smoke_test)

    add_executable(abersoe_smoke_test tests/abersoe_smoke_test.cpp)
    target_link_libraries(abersoe_smoke_test PRIVATE gfe_core)
    add_test(NAME abersoe_smoke_test COMMAND abersoe_smoke_test)

    add_executable(hierarchical_min_smoke_test tests/hierarchical_min_smoke_test.cpp)
    target_link_libraries(hierarchical_min_smoke_test PRIVATE gfe_core)
    add_test(NAME hierarchical_min_smoke_test COMMAND hierarchical_min_smoke_test)

    add_executable(hierarchical_min_integration_test tests/hierarchical_min_integration_test.cpp)
    target_link_libraries(hierarchical_min_integration_test PRIVATE gfe_core)
    add_test(NAME hierarchical_min_integration_test COMMAND hierarchical_min_integration_test)

    add_executable(hierarchical_min_unit_test tests/hierarchical_min_unit_test.cpp)
    target_link_libraries(hierarchical_min_unit_test PRIVATE gfe_core)
    add_test(NAME hierarchical_min_unit_test COMMAND hierarchical_min_unit_test)

    add_executable(hierarchical_min_renorm_test tests/hierarchical_min_renorm_test.cpp)
    target_link_libraries(hierarchical_min_renorm_test PRIVATE gfe_core)
    add_test(NAME hierarchical_min_renorm_test COMMAND hierarchical_min_renorm_test)

    add_executable(hierarchical_min_renorm_integration_test tests/hierarchical_min_renorm_integration_test.cpp)
    target_link_libraries(hierarchical_min_renorm_integration_test PRIVATE gfe_core)
    add_test(NAME hierarchical_min_renorm_integration_test COMMAND hierarchical_min_renorm_integration_test)

    add_executable(hierarchical_min_consistency_test tests/hierarchical_min_consistency_test.cpp)
    target_link_libraries(hierarchical_min_consistency_test PRIVATE gfe_core)
    add_test(NAME hierarchical_min_consistency_test COMMAND hierarchical_min_consistency_test)

    add_executable(hierarchical_min_benchmark_test tests/hierarchical_min_benchmark_test.cpp)
    target_link_libraries(hierarchical_min_benchmark_test PRIVATE gfe_core)
    add_test(NAME hierarchical_min_benchmark_test COMMAND hierarchical_min_benchmark_test)
endif()

if(GFE_BUILD_APPS)
    add_executable(gfe_cli apps/gfe_cli.cpp)
    target_link_libraries(gfe_cli PRIVATE gfe_core)
    install(TARGETS gfe_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_executable(gfe_ou_experiment apps/gfe_ou_experiment.cpp)
    target_link_libraries(gfe_ou_experiment PRIVATE gfe_core)
    install(TARGETS gfe_ou_experiment RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_executable(abersoe_cli apps/abersoe_cli.cpp)
    target_link_libraries(abersoe_cli PRIVATE gfe_core)
    install(TARGETS abersoe_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_executable(hierarchical_min_cli apps/hierarchical_min_cli.cpp)
    target_link_libraries(hierarchical_min_cli PRIVATE gfe_core)
    install(TARGETS hierarchical_min_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(
    TARGETS gfe_core
    EXPORT gfe_coreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT gfe_coreTargets
    FILE gfe_coreTargets.cmake
    NAMESPACE gfe::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gfe_core
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gfe_coreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gfe_coreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gfe_core
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gfe_coreConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/gfe_coreConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/gfe_coreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gfe_core
)

if(GFE_ENABLE_SPACE_CLEAN_TARGETS)
    add_custom_target(space_clean
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            ${CMAKE_BINARY_DIR}/CMakeFiles
            ${CMAKE_BINARY_DIR}/Testing
            ${CMAKE_BINARY_DIR}/_deps
        COMMAND ${CMAKE_COMMAND} -E rm -f
            ${CMAKE_BINARY_DIR}/CMakeCache.txt
            ${CMAKE_BINARY_DIR}/cmake_install.cmake
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_BINARY_DIR}/install_manifest.txt
        COMMENT "Remove CMake cache/temporary artifacts to conserve disk space"
    )
endif()
